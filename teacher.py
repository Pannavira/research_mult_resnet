"""
teacher.py — Teacher model: 6-branch Deep 1D-ResNet + Transformer Late Fusion.

Architecture
------------
For each of the 6 modalities:
    sensor_i ──► Deep1DResNet(1→32→64→128) ──► GAP ──► token_i   (B, D)

Stack 6 tokens → (B, 6, D)
     ──► nn.TransformerEncoder (Self-Attention across sensors)
     ──► mean pooling over 6 tokens → (B, D)
     ──► Classifier FC → logits (B, num_classes)

The forward pass returns ``(logits, attn_map)`` where ``attn_map`` is the
averaged attention weight matrix from the last TransformerEncoder layer,
shape ``(B, num_heads, N_sensors, N_sensors)`` = ``(B, 4, 6, 6)``.
This is consumed by KDLoss for feature alignment with Student SE weights.

Why Late Fusion + TransformerEncoder instead of MulT Cross-Attention?
    - MulT cross-attention on temporal sequences is O(T²) in memory.
      For 6 modalities it would require 30 directional attention matrices.
    - Here we first reduce each modality to a single D-dim token via GAP,
      then apply self-attention over only 6 tokens: O(6²) = trivially cheap.
    - The fusion step still captures inter-sensor dependencies globally.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# ════════════════════════════════════════════════════════════════════════════
#  Building Blocks — Residual Block for 1-D signals
# ════════════════════════════════════════════════════════════════════════════

class ResidualBlock1D(nn.Module):
    """Single residual block: two Conv1d layers with InstanceNorm and skip.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution (used for downsampling).
        dropout: Dropout probability on inter-layer activations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride,
            padding=3, bias=False,
        )
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1,
            padding=3, bias=False,
        )
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.dropout = nn.Dropout1d(p=dropout)

        # Skip connection projection
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.InstanceNorm1d(out_channels, affine=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


# ════════════════════════════════════════════════════════════════════════════
#  Deep 1D-ResNet backbone (one per modality branch)
# ════════════════════════════════════════════════════════════════════════════

class Deep1DResNet(nn.Module):
    """Deep 1-D ResNet backbone that maps a single modality (B,1,L) → (B,D).

    Structure:
        Stem Conv → 3 Stages × (blocks_per_stage ResidualBlock1D) → GAP

    The final Global Average Pooling collapses the temporal dimension so that
    each modality produces a single D-dimensional token.

    Args:
        in_channels: Number of input channels (1 for each sensor).
        stage_channels: Output channels per stage, e.g. [32, 64, 128].
        blocks_per_stage: Residual blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        stage_channels: list[int] | None = None,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        if stage_channels is None:
            stage_channels = [32, 64, 128]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stage_channels[0], kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.InstanceNorm1d(stage_channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Stages
        stages = []
        in_ch = stage_channels[0]
        for ch in stage_channels:
            blocks = []
            for i in range(blocks_per_stage):
                stride = 2 if (i == 0 and ch != stage_channels[0]) else 1
                blocks.append(ResidualBlock1D(in_ch, ch, stride=stride))
                in_ch = ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        # Global Average Pooling → single token per modality
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape ``(B, 1, seq_len)``

        Returns:
            Token vector, shape ``(B, D)`` where D = stage_channels[-1]
        """
        out = self.stem(x)        # (B, 32, L//4)
        out = self.stages(out)    # (B, 128, T')
        out = self.gap(out)       # (B, 128, 1)
        return out.squeeze(-1)    # (B, 128)


# ════════════════════════════════════════════════════════════════════════════
#  Attention-extracting TransformerEncoder wrapper
# ════════════════════════════════════════════════════════════════════════════

class TransformerWithAttn(nn.Module):
    """Wraps nn.TransformerEncoder to also return the last-layer attention map.

    PyTorch's built-in TransformerEncoder doesn't expose attention weights
    out of the box, so we manually run the layer stack and hook the last block.

    Args:
        d_model: Token embedding dimension.
        n_heads: Number of self-attention heads.
        num_layers: Number of TransformerEncoderLayer blocks.
        dim_feedforward: FFN hidden dim (defaults to 4×d_model).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Build individual layers so we can extract attn from the last one
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN for training stability
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape ``(B, N_tokens, D)``

        Returns:
            out: ``(B, N_tokens, D)`` — fused token representations
            attn_map: ``(B, n_heads, N_tokens, N_tokens)`` — last-layer
                      self-attention weight matrix
        """
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)

        # Last layer — extract attention weights manually
        last_layer = self.layers[-1]
        # Pre-LN (norm_first=True)
        out_norm = last_layer.norm1(out)
        # We call the underlying MHA directly with need_weights=True
        attn_out, attn_weights = last_layer.self_attn(
            out_norm, out_norm, out_norm,
            need_weights=True,
            average_attn_weights=False,   # keep per-head → (B, heads, N, N)
        )
        # Residual after attn
        out = out + last_layer.dropout1(attn_out)
        # FFN sub-layer (same as the layer's forward internals)
        out = out + last_layer.dropout2(
            last_layer.linear2(
                last_layer.dropout(
                    last_layer.activation(last_layer.linear1(last_layer.norm2(out)))
                )
            )
        )

        out = self.norm(out)
        return out, attn_weights   # attn_weights: (B, heads, N, N)


# ════════════════════════════════════════════════════════════════════════════
#  Teacher Model
# ════════════════════════════════════════════════════════════════════════════

class TeacherModel(nn.Module):
    """Late-Fusion Teacher: 6 independent ResNet branches + TransformerEncoder.

    Input shape:  ``(B, 6, seq_len)`` — 6 modalities stacked along channel dim.
    Output:       ``(logits, attn_map)``
        - logits:   ``(B, num_classes)``
        - attn_map: ``(B, n_heads, 6, 6)`` — last Transformer layer attention

    Modality ordering (must match data_loader):
        0: chest_ecg  1: chest_eda  2: chest_resp
        3: wrist_bvp  4: wrist_eda  5: wrist_temp
    """

    def __init__(self, cfg: CFG | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = CFG()
        self.cfg = cfg

        n_modalities = len(cfg.teacher_modalities)  # 6
        d_model = cfg.attn_dim                       # 128

        # 6 independent ResNet branches (one per sensor)
        self.branches = nn.ModuleList([
            Deep1DResNet(
                in_channels=1,
                stage_channels=cfg.resnet_channels,
                blocks_per_stage=cfg.resnet_blocks_per_stage,
            )
            for _ in range(n_modalities)
        ])

        # Learnable modality type embedding (so the Transformer knows which
        # token corresponds to which sensor)
        self.modality_embed = nn.Embedding(n_modalities, d_model)

        # Transformer fusion over 6 sensor tokens
        self.transformer = TransformerWithAttn(
            d_model=d_model,
            n_heads=cfg.attn_heads,
            num_layers=cfg.transformer_layers,
            dropout=cfg.transformer_dropout,
        )

        # Classification head: mean-pooled token → FC
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, cfg.num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape ``(B, N_modalities, seq_len)`` — 6 modalities

        Returns:
            logits:   ``(B, num_classes)``
            attn_map: ``(B, n_heads, 6, 6)``
        """
        B, N, L = x.shape  # N == 6

        # ── Per-branch feature extraction ────────────────────────────────
        # Each branch maps (B, 1, L) → (B, D)
        tokens = []
        for i, branch in enumerate(self.branches):
            feat = branch(x[:, i:i+1, :])  # (B, D)
            tokens.append(feat)

        # Stack → (B, N, D)
        tokens = torch.stack(tokens, dim=1)  # (B, 6, D)

        # ── Add modality-type embeddings ─────────────────────────────────
        idx = torch.arange(N, device=x.device)       # (6,)
        mod_emb = self.modality_embed(idx)            # (6, D)
        tokens = tokens + mod_emb.unsqueeze(0)        # (B, 6, D)

        # ── Transformer sensor-level fusion ──────────────────────────────
        fused, attn_map = self.transformer(tokens)    # fused: (B,6,D), attn: (B,H,6,6)

        # ── Mean pooling over sensor tokens → classify ───────────────────
        pooled = fused.mean(dim=1)                    # (B, D)
        logits = self.classifier(pooled)              # (B, num_classes)

        return logits, attn_map


# ════════════════════════════════════════════════════════════════════════════
#  Quick sanity check
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = CFG()
    model = TeacherModel(cfg)
    dummy = torch.randn(2, 6, cfg.seq_len)
    logits, attn_map = model(dummy)
    print(f"logits   : {logits.shape}")       # (2, 2)
    print(f"attn_map : {attn_map.shape}")     # (2, 4, 6, 6)
    total = sum(p.numel() for p in model.parameters())
    print(f"Teacher params: {total:,}")
