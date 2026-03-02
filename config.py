"""
config.py — Centralized hyperparameters for the Cross-Modal KD framework.

All tunable constants live here so that every other module imports from one place.

Teacher uses: chest_ecg, chest_eda, chest_resp, wrist_bvp, wrist_eda, wrist_temp (6 modal)
Student uses: wrist_bvp, wrist_eda, wrist_temp (3 modal — on-device smartwatch)
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

    # ── Modality definitions ─────────────────────────────────────────────────
    # Teacher: 6 privileged sensors (chest + wrist).
    # Each string is "<location>_<signal>" used by data_loader.
    teacher_modalities: List[str] = field(
        default_factory=lambda: [
            "chest_ecg",   # Chest ECG
            "chest_eda",   # Chest EDA
            "chest_resp",  # Chest RESP
            "wrist_bvp",   # Wrist BVP
            "wrist_eda",   # Wrist EDA
            "wrist_temp",  # Wrist TEMP
        ]
    )

    # Student: 3 wrist-only sensors for on-device smartwatch.
    student_modalities: List[str] = field(
        default_factory=lambda: [
            "wrist_bvp",   # Wrist BVP
            "wrist_eda",   # Wrist EDA
            "wrist_temp",  # Wrist TEMP
        ]
    )

    # ── Signal parameters ───────────────────────────────────────────────────
    # Chest sensors: 700 Hz; Wrist sensors: 64 Hz (E4 wristband).
    chest_sr: int = 700             # Chest (RespiBAN) sampling rate (Hz)
    wrist_sr: int = 64              # Wrist (E4) sampling rate (Hz)
    target_sr: int = 128            # Downsample target (Hz) — NOTE: wrist is upsampled from 64
    window_sec: float = 60.0        # Sliding window duration (seconds)
    overlap: float = 0.5            # Overlap fraction (50%)
    # Derived: seq_len = int(target_sr * window_sec) = 7680
    seq_len: int = 7680

    # ── Label mapping ──────────────────────────────────────────────────────
    #   Original WESAD labels: 0=not defined, 1=baseline, 2=stress,
    #   3=amusement, 4=meditation, 5/6/7=ignore.
    #   BINARY: Baseline (1) and Amusement (3) → Non-Stress (0).
    #           Stress (2) → Stress (1).
    label_map: dict = field(
        default_factory=lambda: {1: 0, 2: 1, 3: 0}
    )
    num_classes: int = 2
    class_names: List[str] = field(
        default_factory=lambda: ["Non-Stress", "Stress"]
    )

    # ── Filter design ──────────────────────────────────────────────────────
    ecg_bandpass: tuple = (0.5, 40.0)    # Hz — chest ECG
    eda_bandpass: tuple = (0.05, 5.0)    # Hz — EDA (chest & wrist)
    resp_bandpass: tuple = (0.1, 0.5)    # Hz — chest RESP (respiration rate band)
    bvp_bandpass: tuple = (0.5, 8.0)     # Hz — wrist BVP (PPG signal)
    temp_lowpass: float = 1.0            # Hz — wrist TEMP (skin temperature, slow signal)
    filter_order: int = 4

    # ── Training — general ─────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4 # 0 on Windows to avoid hangs
    epochs_teacher: int = 100
    epochs_student: int = 80
    lr: float = 1e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 25  # Stop if val loss doesn't improve for N epochs
    seed: int = 42
    noise_std: float = 0.1          # Standard deviation for Gaussian noise augmentation

    # ── Personalization (subject calibration) ──────────────────────────
    personalize: bool = False       # Whether to fine-tune using test subject baseline
    personalize_baseline_minutes: float = 2.0  # Minutes of baseline data to mix in
    personalize_finetune_epochs: int = 5       # Quick fine-tune epochs on mixed data

    # ── Knowledge Distillation ─────────────────────────────────────────────
    temperature: float = 4.0        # Softmax temperature T
    alpha: float = 0.5              # Weight for task CE loss
    beta: float = 0.5               # Weight for response-based KL loss (trust teacher)
    gamma: float = 1.0              # Weight for feature-based MSE loss (force SE→Attn match)

    # ── Missing-modality simulation ────────────────────────────────────────
    missing_prob: float = 0.5       # Probability of dropping a modality (applied to student)
    drop_modality: str = "random"   # Which student modality to drop ("random" or index 0,1,2)

    # ── Teacher architecture ───────────────────────────────────────────────
    resnet_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128]
    )
    resnet_blocks_per_stage: int = 2
    # TransformerEncoder for sensor-token fusion
    attn_heads: int = 4
    attn_dim: int = 128             # Must match last resnet channel
    transformer_layers: int = 2     # Number of TransformerEncoder layers
    transformer_dropout: float = 0.1

    # ── Student architecture ───────────────────────────────────────────────
    student_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256]  # 2x capacity vs original
    )
    se_reduction: int = 4

    # ── Device ─────────────────────────────────────────────────────────────
    @property
    def device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
