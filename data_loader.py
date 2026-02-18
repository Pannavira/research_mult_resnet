"""
data_loader.py — WESAD data loading, preprocessing, and PyTorch Dataset/DataLoader.

Pipeline
--------
1. Load .pkl files (one per subject).
2. Extract chest ECG (1-ch) and EDA (1-ch) + integer label vector.
3. Apply Butterworth filtering (band-pass for ECG, low-pass for EDA).
4. Downsample from 700 Hz → 128 Hz.
5. Sliding-window segmentation (60 s, 50 % overlap).
6. Majority-vote label per window; discard windows whose majority is not
   in {Baseline=1, Stress=2, Amusement=3}.
7. Pack into a PyTorch Dataset and DataLoader.
8. Provide a MissingModalityWrapper that zeros out one modality with a
   configurable probability (used during Student training).
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import butter, decimate, sosfiltfilt
from torch.utils.data import DataLoader, Dataset, Subset

from config import CFG


# ════════════════════════════════════════════════════════════════════════════
#  Filtering utilities
# ════════════════════════════════════════════════════════════════════════════

def _butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    """Design a Butterworth band-pass filter (second-order sections)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band", output="sos")


def _butter_lowpass(cutoff: float, fs: int, order: int = 4):
    """Design a Butterworth low-pass filter (second-order sections)."""
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype="low", output="sos")


def filter_ecg(signal: np.ndarray, fs: int, cfg: CFG) -> np.ndarray:
    """Apply band-pass filter to ECG signal.

    Args:
        signal: Raw ECG, shape ``(N,)``
        fs: Sampling frequency (Hz)
        cfg: Configuration object

    Returns:
        Filtered ECG, shape ``(N,)``
    """
    sos = _butter_bandpass(cfg.ecg_bandpass[0], cfg.ecg_bandpass[1], fs, cfg.filter_order)
    return sosfiltfilt(sos, signal).astype(np.float32)


def filter_eda(signal: np.ndarray, fs: int, cfg: CFG) -> np.ndarray:
    """Apply band-pass filter to extract phasic EDA component (removing drift).

    Args:
        signal: Raw EDA, shape ``(N,)``
        fs: Sampling frequency (Hz)
        cfg: Configuration object

    Returns:
        Filtered EDA, shape ``(N,)``
    """
    sos = _butter_bandpass(cfg.eda_bandpass[0], cfg.eda_bandpass[1], fs, cfg.filter_order)
    return sosfiltfilt(sos, signal).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Loading & preprocessing one subject
# ════════════════════════════════════════════════════════════════════════════

def _load_subject(subject_id: str, cfg: CFG) -> Dict[str, np.ndarray]:
    """Load a single WESAD subject .pkl file and return preprocessed arrays.

    Returns:
        dict with keys ``'ecg'``, ``'eda'``, ``'label'`` — all at
        *original* sampling rate (700 Hz), filtered.
    """
    pkl_path = os.path.join(cfg.data_dir, subject_id, f"{subject_id}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Chest signals: each is (N, 1) → squeeze to (N,)
    ecg_raw = data["signal"]["chest"]["ECG"].squeeze()   # shape: (N,)
    eda_raw = data["signal"]["chest"]["EDA"].squeeze()   # shape: (N,)
    labels  = data["label"].squeeze()                    # shape: (N,)

    # Apply Butterworth filters at original sampling rate
    ecg_filt = filter_ecg(ecg_raw, cfg.original_sr, cfg)  # shape: (N,)
    eda_filt = filter_eda(eda_raw, cfg.original_sr, cfg)   # shape: (N,)

    return {"ecg": ecg_filt, "eda": eda_filt, "label": labels}


def _downsample(signal: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Downsample a 1-D signal using scipy.signal.decimate.

    Args:
        signal: shape ``(N,)``
        original_sr: Original sampling rate
        target_sr: Target sampling rate

    Returns:
        Downsampled signal, shape roughly ``(N * target_sr / original_sr,)``
    """
    factor = original_sr // target_sr  # 700 // 128 ≈ 5
    if factor <= 1:
        return signal
    return decimate(signal, factor, zero_phase=True).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Sliding-window segmentation
# ════════════════════════════════════════════════════════════════════════════

def _segment_windows(
    ecg: np.ndarray,
    eda: np.ndarray,
    labels: np.ndarray,
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment continuous signals into fixed-length windows.

    Args:
        ecg: Downsampled ECG, shape ``(T,)``
        eda: Downsampled EDA, shape ``(T,)``
        labels: Downsampled labels, shape ``(T,)``
        cfg: Configuration object

    Returns:
        ecg_wins: ``(num_windows, seq_len)``
        eda_wins: ``(num_windows, seq_len)``
        label_wins: ``(num_windows,)``  — majority-vote label per window
    """
    win_len = cfg.seq_len                              # 7680 samples
    step = int(win_len * (1.0 - cfg.overlap))          # 3840 samples

    total = min(len(ecg), len(eda), len(labels))

    ecg_list, eda_list, lbl_list = [], [], []
    for start in range(0, total - win_len + 1, step):
        end = start + win_len
        lbl_window = labels[start:end]

        # Majority-vote label
        counts = np.bincount(lbl_window.astype(int), minlength=8)
        majority_label = int(np.argmax(counts))

        # Keep only windows whose majority label is in {1, 2, 3}
        if majority_label not in cfg.label_map:
            continue

        ecg_list.append(ecg[start:end])
        eda_list.append(eda[start:end])
        lbl_list.append(cfg.label_map[majority_label])  # remap to 0-indexed

    ecg_wins = np.stack(ecg_list, axis=0)   # shape: (num_windows, seq_len)
    eda_wins = np.stack(eda_list, axis=0)   # shape: (num_windows, seq_len)
    label_wins = np.array(lbl_list, dtype=np.int64)  # shape: (num_windows,)

    return ecg_wins, eda_wins, label_wins


# ════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ════════════════════════════════════════════════════════════════════════════

class WESADDataset(Dataset):
    """PyTorch dataset for windowed WESAD data.

    Each sample returns
    ``(x, label)`` where ``x`` has shape ``(2, seq_len)`` — channel 0 is ECG,
    channel 1 is EDA — and ``label`` is an int in {0, 1, 2}.
    """

    def __init__(self, subjects: List[str], cfg: CFG) -> None:
        super().__init__()
        self.cfg = cfg

        all_ecg, all_eda, all_labels = [], [], []
        for sid in subjects:
            print(f"  [data_loader] Loading {sid} ...")
            raw = _load_subject(sid, cfg)

            # Downsample all signals (including labels via nearest)
            ds_factor = cfg.original_sr // cfg.target_sr
            ecg_ds = _downsample(raw["ecg"], cfg.original_sr, cfg.target_sr)
            eda_ds = _downsample(raw["eda"], cfg.original_sr, cfg.target_sr)
            lbl_ds = raw["label"][::ds_factor]  # nearest-neighbour for labels

            # Align lengths
            min_len = min(len(ecg_ds), len(eda_ds), len(lbl_ds))
            ecg_ds, eda_ds, lbl_ds = ecg_ds[:min_len], eda_ds[:min_len], lbl_ds[:min_len]

            # Segment into windows
            ecg_w, eda_w, lbl_w = _segment_windows(ecg_ds, eda_ds, lbl_ds, cfg)
            all_ecg.append(ecg_w)
            all_eda.append(eda_w)
            all_labels.append(lbl_w)

        self.ecg = np.concatenate(all_ecg, axis=0)       # (N_total, seq_len)
        self.eda = np.concatenate(all_eda, axis=0)       # (N_total, seq_len)
        self.labels = np.concatenate(all_labels, axis=0) # (N_total,)

        # Z-score normalisation (per-channel, across dataset)
        self.ecg = (self.ecg - self.ecg.mean()) / (self.ecg.std() + 1e-8)
        self.eda = (self.eda - self.eda.mean()) / (self.eda.std() + 1e-8)

        print(f"  [data_loader] Dataset ready — {len(self)} windows, "
              f"class distribution: {np.bincount(self.labels).tolist()}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        ecg = torch.from_numpy(self.ecg[idx]).float()   # shape: (seq_len,)
        eda = torch.from_numpy(self.eda[idx]).float()   # shape: (seq_len,)
        x = torch.stack([ecg, eda], dim=0)              # shape: (2, seq_len)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, label


# ════════════════════════════════════════════════════════════════════════════
#  Missing-modality simulation (used during Student training)
# ════════════════════════════════════════════════════════════════════════════

class MissingModalityWrapper(Dataset):
    """Wraps a WESADDataset and randomly zeros out a modality.

    Args:
        base_dataset: Underlying WESADDataset.
        missing_prob: Probability of dropping a modality per sample.
        drop_modality: ``"eda"`` (drop channel 1), ``"ecg"`` (drop channel 0),
            or ``"random"`` (randomly choose which to drop each time).
    """

    def __init__(
        self,
        base_dataset: WESADDataset,
        missing_prob: float = 0.5,
        drop_modality: str = "eda",
    ) -> None:
        self.base = base_dataset
        self.missing_prob = missing_prob
        self.drop_modality = drop_modality

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, label = self.base[idx]  # x shape: (2, seq_len)

        if torch.rand(1).item() < self.missing_prob:
            if self.drop_modality == "eda":
                x[1] = 0.0   # Zero out EDA channel
            elif self.drop_modality == "ecg":
                x[0] = 0.0   # Zero out ECG channel
            elif self.drop_modality == "random":
                ch = torch.randint(0, 2, (1,)).item()
                x[ch] = 0.0
            else:
                raise ValueError(f"Unknown drop_modality: {self.drop_modality}")

        return x, label


# ════════════════════════════════════════════════════════════════════════════
#  DataLoader builders
# ════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    cfg: CFG,
    train_subjects: Optional[List[str]] = None,
    val_subjects: Optional[List[str]] = None,
    wrap_missing: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Uses a simple train/val subject split.  For Leave-One-Subject-Out (LOSO),
    call this function inside a loop over subjects.

    Args:
        cfg: Configuration object.
        train_subjects: List of subject IDs for training.
        val_subjects: List of subject IDs for validation.
        wrap_missing: If True, wrap the *training* set with
            ``MissingModalityWrapper`` (used for Student training).

    Returns:
        ``(train_loader, val_loader)``
    """
    if train_subjects is None:
        # Default: first 12 subjects train, last 3 val
        train_subjects = cfg.all_subjects[:12]
        val_subjects = cfg.all_subjects[12:]
    if val_subjects is None:
        val_subjects = cfg.all_subjects[12:]

    print("[data_loader] Building training set ...")
    train_ds = WESADDataset(train_subjects, cfg)

    print("[data_loader] Building validation set ...")
    val_ds = WESADDataset(val_subjects, cfg)

    if wrap_missing:
        train_ds = MissingModalityWrapper(
            train_ds,
            missing_prob=cfg.missing_prob,
            drop_modality=cfg.drop_modality,
        )

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
    )

    return train_loader, val_loader


def build_loso_splits(cfg: CFG):
    """Generator for Leave-One-Subject-Out cross-validation.

    Yields:
        ``(test_subject, train_loader, val_loader)`` for each fold.
    """
    for test_subj in cfg.all_subjects:
        remaining = [s for s in cfg.all_subjects if s != test_subj]
        train_subjects = remaining[:-1]
        val_subjects = [remaining[-1]]

        train_loader, val_loader = build_dataloaders(
            cfg,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
        )
        yield test_subj, train_loader, val_loader
