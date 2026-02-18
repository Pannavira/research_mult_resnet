"""
config.py — Centralized hyperparameters for the Missing-Modality KD framework.

All tunable constants live here so that every other module imports from one place.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class CFG:
    """Master configuration object."""

    # ── Paths ───────────────────────────────────────────────────────────────
    data_dir: str = os.path.join(".", "data", "WESAD")
    checkpoint_dir: str = os.path.join(".", "checkpoints")

    # ── Subject IDs available in the dataset ────────────────────────────────
    all_subjects: List[str] = field(
        default_factory=lambda: [
            "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17",
        ]
    )

    # ── Signal parameters ───────────────────────────────────────────────────
    original_sr: int = 700          # RespiBAN chest sensor sampling rate (Hz)
    target_sr: int = 128            # Downsample target (Hz)
    window_sec: float = 60.0        # Sliding window duration (seconds)
    overlap: float = 0.5            # Overlap fraction (50 %)
    # Derived: seq_len = int(target_sr * window_sec) = 7680
    seq_len: int = 7680

    # ── Label mapping ──────────────────────────────────────────────────────
    #   Original WESAD labels: 0=not defined, 1=baseline, 2=stress,
    #   3=amusement, 4=meditation, 5/6/7=ignore.
    #   We keep 1, 2, 3 and remap to 0-indexed: {1→0, 2→1, 3→2}
    label_map: dict = field(
        default_factory=lambda: {1: 0, 2: 1, 3: 2}
    )
    num_classes: int = 3
    class_names: List[str] = field(
        default_factory=lambda: ["Baseline", "Stress", "Amusement"]
    )

    # ── Filter design ──────────────────────────────────────────────────────
    ecg_bandpass: tuple = (0.5, 40.0)   # Hz  (Butterworth band-pass)
    eda_lowpass: float = 5.0            # Hz  (Butterworth low-pass)
    filter_order: int = 4

    # ── Training — general ─────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 0 if os.name == "nt" else 2  # 0 on Windows to avoid hangs
    epochs_teacher: int = 80
    epochs_student: int = 80
    lr: float = 5e-4
    weight_decay: float = 1e-1
    early_stopping_patience: int = 10  # Stop if val loss doesn't improve for N epochs
    seed: int = 42

    # ── Knowledge Distillation ─────────────────────────────────────────────
    temperature: float = 4.0        # Softmax temperature T
    alpha: float = 0.7              # Weight for task CE loss
    beta: float = 0.3               # Weight for response-based KL loss
    gamma: float = 0.2              # Weight for feature-based MSE loss

    # ── Missing-modality simulation ────────────────────────────────────────
    missing_prob: float = 0.5       # Probability of dropping a modality
    drop_modality: str = "eda"      # Which modality to drop ("eda", "ecg", or "random")

    # ── Teacher architecture ───────────────────────────────────────────────
    resnet_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    resnet_blocks_per_stage: int = 2
    attn_heads: int = 4
    attn_dim: int = 256             # Must match last resnet channel

    # ── Student architecture ───────────────────────────────────────────────
    student_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128]
    )
    se_reduction: int = 4

    # ── Device ─────────────────────────────────────────────────────────────
    @property
    def device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
