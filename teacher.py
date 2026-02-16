"""
teacher.py — Teacher model: Deep 1D-ResNet feature extractors + MulT cross-attention fusion.

Architecture
------------
ECG ──► Deep1DResNet(1→64→128→256) ──► feat_ecg  (B, 256, T')
                                              │
                                Cross-Attention (Q=ECG, K/V=EDA) ──► attn_weights_ecg2eda
                                              │
EDA ──► Deep1DResNet(1→64→128→256) ──► feat_eda  (B, 256, T')
                                              │
                                Cross-Attention (Q=EDA, K/V=ECG) ──► attn_weights_eda2ecg
                                              │
                                Cat + AdaptivePool + FC ──► logits (B, 3)

The forward pass returns ``(logits, attn_ecg2eda, attn_eda2ecg)`` so that
attention weights can be used for feature-based knowledge distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# ════════════════════════════════════════════════════════════════════════════
#  Building Blocks — Residual Block for 1-D signals
# ════════════════════════════════════════════════════════════════════════════

class ResidualBlock1D(nn.Module):
    """Single residual block: two Conv1d layers with BatchNorm and skip.

    If ``in_channels != out_channels`` a 1×1 projection is added to the skip
    path so dimensions match.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution (used for downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride,
            padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1,
            padding=3, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection projection
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape ``(B, C_in, L)``

        Returns:
            shape ``(B, C_out, L')`` where ``L' = L // stride``
        """
        identity = self.skip(x)       # shape: (B, C_out, L')
        out = F.relu(self.bn1(self.conv1(x)))  # shape: (B, C_out, L')
        out = self.bn2(self.conv2(out))        # shape: (B, C_out, L')
        return F.relu(out + identity)          # shape: (B, C_out, L')


# ════════════════════════════════════════════════════════════════════════════
#  Deep 1D-ResNet backbone (one per modality)
# ════════════════════════════════════════════════════════════════════════════

class Deep1DResNet(nn.Module):
    """Deep 1-D ResNet backbone for a single modality channel.

    Structure:
        Stem Conv → 3 Stages × (blocks_per_stage ResidualBlock1D)

    Args:
        in_channels: Number of input channels (1 for ECG or EDA).
        stage_channels: List of output channels per stage, e.g. ``[64, 128, 256]``.
        blocks_per_stage: Number of residual blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        stage_channels: list[int] | None = None,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        if stage_channels is None:
            stage_channels = [64, 128, 256]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stage_channels[0], kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(stage_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )  # output shape: (B, 64, L//4)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape ``(B, 1, seq_len)``

        Returns:
            Feature map, shape ``(B, C_last, T')`` where ``T'`` is the
            temporally downsampled length.
        """
        out = self.stem(x)        # shape: (B, 64, seq_len//4)
        out = self.stages(out)    # shape: (B, 256, T')
        return out


# ════════════════════════════════════════════════════════════════════════════
#  Cross-Attention Layer (MulT-style)
# ════════════════════════════════════════════════════════════════════════════

class CrossAttentionLayer(nn.Module):
    """Multi-head cross-attention: query from one modality, key/value from another.

    Args:
        d_model: Feature dimension (must match the last ResNet channel).
        n_heads: Number of attention heads.
        dropout: Dropout on attention weights.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: shape ``(B, T_q, D)``  — from modality A
            key_value: shape ``(B, T_kv, D)`` — from modality B

        Returns:
            output: shape ``(B, T_q, D)``
            attn_weights: shape ``(B, T_q, T_kv)``
        """
        # Multi-head cross-attention
        attn_out, attn_weights = self.attn(
            query, key_value, key_value, need_weights=True,
            average_attn_weights=True,
        )  # attn_out: (B, T_q, D), attn_weights: (B, T_q, T_kv)

        # Add & Norm
        out = self.norm(query + attn_out)    # shape: (B, T_q, D)

        # Feed-forward + Add & Norm
        out = self.norm2(out + self.ffn(out))  # shape: (B, T_q, D)

        return out, attn_weights


# ════════════════════════════════════════════════════════════════════════════
#  Teacher Model
# ════════════════════════════════════════════════════════════════════════════

class TeacherModel(nn.Module):
    """MulT + Deep 1D-ResNet teacher for multimodal stress detection.

    Returns:
        logits: ``(B, num_classes)``
        attn_ecg2eda: ``(B, T', T')`` — attention weights where ECG queries EDA
        attn_eda2ecg: ``(B, T', T')`` — attention weights where EDA queries ECG
    """

    def __init__(self, cfg: CFG | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = CFG()
        self.cfg = cfg

        # Independent ResNet branches
        self.resnet_ecg = Deep1DResNet(
            in_channels=1,
            stage_channels=cfg.resnet_channels,
            blocks_per_stage=cfg.resnet_blocks_per_stage,
        )
        self.resnet_eda = Deep1DResNet(
            in_channels=1,
            stage_channels=cfg.resnet_channels,
            blocks_per_stage=cfg.resnet_blocks_per_stage,
        )

        # Cross-attention layers
        self.cross_attn_ecg2eda = CrossAttentionLayer(
            d_model=cfg.attn_dim, n_heads=cfg.attn_heads,
        )
        self.cross_attn_eda2ecg = CrossAttentionLayer(
            d_model=cfg.attn_dim, n_heads=cfg.attn_heads,
        )

        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.attn_dim * 2, cfg.attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(cfg.attn_dim, cfg.num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor, shape ``(B, 2, seq_len)``
                Channel 0 = ECG, Channel 1 = EDA.

        Returns:
            logits: ``(B, num_classes)``
            attn_ecg2eda: ``(B, T', T')``
            attn_eda2ecg: ``(B, T', T')``
        """
        # Split modalities — each (B, 1, seq_len)
        ecg_in = x[:, 0:1, :]   # shape: (B, 1, seq_len)
        eda_in = x[:, 1:2, :]   # shape: (B, 1, seq_len)

        # Feature extraction
        feat_ecg = self.resnet_ecg(ecg_in)  # shape: (B, 256, T')
        feat_eda = self.resnet_eda(eda_in)  # shape: (B, 256, T')

        # Transpose to (B, T', D) for attention
        feat_ecg_t = feat_ecg.permute(0, 2, 1)  # shape: (B, T', 256)
        feat_eda_t = feat_eda.permute(0, 2, 1)  # shape: (B, T', 256)

        # Cross-attention: ECG queries EDA
        fused_ecg, attn_ecg2eda = self.cross_attn_ecg2eda(
            query=feat_ecg_t, key_value=feat_eda_t,
        )  # fused_ecg: (B, T', 256), attn_ecg2eda: (B, T', T')

        # Cross-attention: EDA queries ECG
        fused_eda, attn_eda2ecg = self.cross_attn_eda2ecg(
            query=feat_eda_t, key_value=feat_ecg_t,
        )  # fused_eda: (B, T', 256), attn_eda2ecg: (B, T', T')

        # Back to (B, D, T') for pooling
        fused_ecg = fused_ecg.permute(0, 2, 1)  # shape: (B, 256, T')
        fused_eda = fused_eda.permute(0, 2, 1)  # shape: (B, 256, T')

        # Global average pooling
        pool_ecg = self.pool(fused_ecg).squeeze(-1)  # shape: (B, 256)
        pool_eda = self.pool(fused_eda).squeeze(-1)  # shape: (B, 256)

        # Concatenate and classify
        combined = torch.cat([pool_ecg, pool_eda], dim=1)  # shape: (B, 512)
        logits = self.classifier(combined)                  # shape: (B, num_classes)

        return logits, attn_ecg2eda, attn_eda2ecg


# ════════════════════════════════════════════════════════════════════════════
#  Quick sanity check
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = CFG()
    model = TeacherModel(cfg)
    dummy = torch.randn(2, 2, cfg.seq_len)
    logits, a1, a2 = model(dummy)
    print(f"logits : {logits.shape}")     # (2, 3)
    print(f"attn_e2d: {a1.shape}")        # (2, T', T')
    print(f"attn_d2e: {a2.shape}")        # (2, T', T')
    total = sum(p.numel() for p in model.parameters())
    print(f"Teacher params: {total:,}")
