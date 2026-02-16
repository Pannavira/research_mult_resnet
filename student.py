"""
student.py — Lightweight Student model for edge/TinyML deployment.

Architecture
------------
Input (B, 2, seq_len)
  │
  DepthwiseSeparable1D(2 → 32)  → SE(32) → MaxPool
  DepthwiseSeparable1D(32 → 64) → SE(64) → MaxPool
  DepthwiseSeparable1D(64 → 128)→ SE(128)→ MaxPool
  │
  AdaptiveAvgPool1d(1) → FC → logits (B, 3)

The forward pass returns ``(logits, se_weights_list)`` where
``se_weights_list`` is a list of channel-attention vectors from each SE
block — these are aligned with the Teacher's cross-attention weights
during feature-based knowledge distillation.

The model is robust to missing modalities because one input channel can
be entirely zeroed out and the SE blocks will adaptively re-weight the
remaining channel information.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# ════════════════════════════════════════════════════════════════════════════
#  Depthwise Separable Convolution 1D
# ════════════════════════════════════════════════════════════════════════════

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1-D convolution.

    Factorises a standard convolution into:
        1. *Depthwise*  — one filter per input channel (``groups = in_ch``).
        2. *Pointwise*  — 1×1 convolution to mix channels.

    This dramatically reduces parameter count compared to a standard Conv1d.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the depthwise convolution.
        stride: Stride for the depthwise convolution.
        padding: Padding for the depthwise convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
    ) -> None:
        super().__init__()
        # Depthwise: each input channel gets its own filter
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,   # ← one filter per channel
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(in_channels)

        # Pointwise: 1×1 conv to project to out_channels
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape ``(B, C_in, L)``

        Returns:
            shape ``(B, C_out, L')`` (``L' = L`` when stride = 1)
        """
        out = F.relu(self.bn1(self.depthwise(x)))   # shape: (B, C_in, L')
        out = F.relu(self.bn2(self.pointwise(out)))  # shape: (B, C_out, L')
        return out


# ════════════════════════════════════════════════════════════════════════════
#  Squeeze-and-Excitation Block 1D
# ════════════════════════════════════════════════════════════════════════════

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps.

    Learns a per-channel attention vector via:
        GAP → FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid

    During KD, the SE attention weights are extracted and aligned with the
    Teacher's cross-attention weights.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio ``r``.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape ``(B, C, L)``

        Returns:
            scaled: shape ``(B, C, L)`` — input re-weighted by SE weights
            se_weights: shape ``(B, C)`` — the channel attention vector
        """
        b, c, _ = x.size()
        se = self.squeeze(x).view(b, c)            # shape: (B, C)
        se = self.excitation(se)                    # shape: (B, C)  ∈ [0, 1]
        scaled = x * se.unsqueeze(-1)               # shape: (B, C, L)
        return scaled, se


# ════════════════════════════════════════════════════════════════════════════
#  Student Block = DepthwiseSeparable + SE + MaxPool
# ════════════════════════════════════════════════════════════════════════════

class StudentBlock(nn.Module):
    """One stage of the Student: DSConv → SE → MaxPool.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        se_reduction: SE reduction ratio.
        pool_size: Max-pool kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_reduction: int = 4,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv1d(in_channels, out_channels)
        self.se = SEBlock1D(out_channels, reduction=se_reduction)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape ``(B, C_in, L)``

        Returns:
            out: shape ``(B, C_out, L // pool_size)``
            se_weights: shape ``(B, C_out)``
        """
        out = self.ds_conv(x)           # shape: (B, C_out, L)
        out, se_w = self.se(out)        # out: (B, C_out, L), se_w: (B, C_out)
        out = self.pool(out)            # shape: (B, C_out, L // pool_size)
        return out, se_w


# ════════════════════════════════════════════════════════════════════════════
#  Full Student Model
# ════════════════════════════════════════════════════════════════════════════

class StudentModel(nn.Module):
    """Lightweight Depthwise-Separable CNN + SE student for edge deployment.

    The model accepts ``(B, 2, seq_len)`` and outputs:
        - ``logits``: ``(B, num_classes)``
        - ``se_weights``: list of ``(B, C_i)`` tensors from each SE block

    Designed to be robust to missing modalities (one channel zeroed out).
    """

    def __init__(self, cfg: CFG | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = CFG()
        self.cfg = cfg

        channels = cfg.student_channels  # e.g. [32, 64, 128]

        # Build stages
        self.blocks = nn.ModuleList()
        in_ch = 2  # ECG + EDA
        for ch in channels:
            self.blocks.append(
                StudentBlock(in_ch, ch, se_reduction=cfg.se_reduction, pool_size=4)
            )
            in_ch = ch

        # Classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(channels[-1] // 2, cfg.num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: shape ``(B, 2, seq_len)``

        Returns:
            logits: ``(B, num_classes)``
            se_weights: list of 3 tensors, shapes ``(B, 32)``, ``(B, 64)``,
                ``(B, 128)`` — channel-attention vectors from each SE block.
        """
        se_weights: List[torch.Tensor] = []
        out = x  # shape: (B, 2, seq_len)

        for block in self.blocks:
            out, se_w = block(out)
            se_weights.append(se_w)
            # After block i: out shape (B, C_i, L_i)

        # Global average pool → classify
        pooled = self.gap(out).squeeze(-1)   # shape: (B, C_last)
        logits = self.classifier(pooled)     # shape: (B, num_classes)

        return logits, se_weights


# ════════════════════════════════════════════════════════════════════════════
#  Quick sanity check
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = CFG()
    model = StudentModel(cfg)
    dummy = torch.randn(2, 2, cfg.seq_len)   # (B=2, 2 channels, 7680)
    logits, se_list = model(dummy)
    print(f"logits  : {logits.shape}")        # (2, 3)
    for i, se in enumerate(se_list):
        print(f"SE[{i}]   : {se.shape}")      # (2, 32), (2, 64), (2, 128)
    total = sum(p.numel() for p in model.parameters())
    print(f"Student params: {total:,}")

    # Test missing modality — zero out EDA
    dummy_missing = dummy.clone()
    dummy_missing[:, 1, :] = 0.0
    logits_m, _ = model(dummy_missing)
    print(f"Missing-mod logits: {logits_m.shape}")  # Should still be (2, 3)
