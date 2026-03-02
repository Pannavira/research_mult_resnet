"""
data_loader.py — WESAD data loading, preprocessing, and PyTorch Dataset/DataLoader.

Pipeline
--------
1. Load .pkl files (one per subject).
2. Extract sensors from data['signal']['chest'] and data['signal']['wrist']
   according to cfg.teacher_modalities and cfg.student_modalities lists.
3. Apply per-sensor bandpass/lowpass filters and downsample to target_sr.
4. Sliding-window segmentation (60 s, 50 % overlap).
5. Majority-vote label per window; discard windows whose majority is not
   in {Baseline=1, Stress=2, Amusement=3}.
6. __getitem__ returns (x_teacher, x_student, label) where:
     x_teacher shape: (6, seq_len) — all 6 teacher modalities
     x_student shape: (3, seq_len) — 3 wrist-only student modalities
7. MissingModalityWrapper zeros out modalities only on x_student.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import butter, decimate, resample_poly, sosfiltfilt
from torch.utils.data import DataLoader, Dataset

from config import CFG
from augmentation import augment_signal


# ════════════════════════════════════════════════════════════════════════════
#  Per-sensor filter helpers
# ════════════════════════════════════════════════════════════════════════════

def _butter_bandpass_sos(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")


def _butter_lowpass_sos(cutoff: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype="low", output="sos")


def _filter_signal(signal: np.ndarray, modality: str, fs: int, cfg: CFG) -> np.ndarray:
    """Apply the appropriate Butterworth filter for a given modality name.

    Args:
        signal:   Raw 1-D signal array.
        modality: One of the modality name strings, e.g. 'chest_ecg'.
        fs:       Sampling rate of the signal (Hz).
        cfg:      Configuration object.

    Returns:
        Filtered 1-D signal (float32).
    """
    signal = signal.astype(np.float64)  # sosfiltfilt needs float64

    sig_type = modality.split("_", 1)[1]  # 'ecg', 'eda', 'resp', 'bvp', 'temp'

    if sig_type == "ecg":
        sos = _butter_bandpass_sos(cfg.ecg_bandpass[0], cfg.ecg_bandpass[1], fs, cfg.filter_order)
    elif sig_type == "eda":
        sos = _butter_bandpass_sos(cfg.eda_bandpass[0], cfg.eda_bandpass[1], fs, cfg.filter_order)
    elif sig_type == "resp":
        sos = _butter_bandpass_sos(cfg.resp_bandpass[0], cfg.resp_bandpass[1], fs, cfg.filter_order)
    elif sig_type == "bvp":
        sos = _butter_bandpass_sos(cfg.bvp_bandpass[0], cfg.bvp_bandpass[1], fs, cfg.filter_order)
    elif sig_type == "temp":
        sos = _butter_lowpass_sos(cfg.temp_lowpass, fs, cfg.filter_order)
    else:
        raise ValueError(f"Unknown signal type in modality name: '{modality}'")

    return sosfiltfilt(sos, signal).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Resampling helpers
# ════════════════════════════════════════════════════════════════════════════

def _resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal from orig_sr → target_sr using polyphase method.

    Works for both downsampling (700→128) and upsampling (64→128).

    Args:
        signal:    shape ``(N,)``
        orig_sr:   Original sampling rate (Hz).
        target_sr: Target sampling rate (Hz).

    Returns:
        Resampled signal (float32).
    """
    if orig_sr == target_sr:
        return signal.astype(np.float32)

    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(signal, up, down).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  WESAD sensor extraction per modality name
# ════════════════════════════════════════════════════════════════════════════

# Map modality name → (location key, signal key, native_sr)
# Chest sensors sample at 700 Hz; Wrist (E4) samples vary by signal:
#   BVP: 64 Hz, EDA: 4 Hz, TEMP: 4 Hz (we still resample all to target_sr)
_MODALITY_META: Dict[str, Tuple[str, str, int]] = {
    "chest_ecg":  ("chest", "ECG",  700),
    "chest_eda":  ("chest", "EDA",  700),
    "chest_resp": ("chest", "Resp", 700),
    "wrist_bvp":  ("wrist", "BVP",   64),
    "wrist_eda":  ("wrist", "EDA",    4),
    "wrist_temp": ("wrist", "TEMP",   4),
}


def _extract_modality(data: dict, modality: str, cfg: CFG) -> np.ndarray:
    """Extract, resample, then filter a single modality to target_sr.

    Order: resample FIRST → filter SECOND.
    Reason: low-rate wrist sensors (EDA/TEMP at 4 Hz, BVP at 64 Hz) have a
    Nyquist frequency that is lower than some filter cutoffs (e.g. EDA highcut
    5 Hz > Nyquist 2 Hz at 4 Hz).  By resampling to target_sr=128 Hz first,
    the Nyquist becomes 64 Hz, which is safely above all filter cutoffs.

    Args:
        data:     Raw pickle dict for one subject.
        modality: Modality string, e.g. 'chest_ecg'.
        cfg:      Configuration object.

    Returns:
        Processed signal, shape ``(M,)`` at ``cfg.target_sr`` Hz.
    """
    location, signal_key, native_sr = _MODALITY_META[modality]
    raw = data["signal"][location][signal_key].squeeze().astype(np.float64)

    # Step 1: resample to target_sr (handles both up- and down-sampling)
    resampled = _resample_signal(raw, native_sr, cfg.target_sr)

    # Step 2: filter at target_sr (Nyquist = target_sr/2 = 64 Hz — safe for all cutoffs)
    filtered = _filter_signal(resampled, modality, cfg.target_sr, cfg)
    return filtered


# ════════════════════════════════════════════════════════════════════════════
#  Load and preprocess one subject
# ════════════════════════════════════════════════════════════════════════════

def _load_subject(subject_id: str, cfg: CFG) -> Dict[str, np.ndarray]:
    """Load one subject .pkl and return arrays for all teacher/student modalities.

    Returns:
        dict with:
            - one key per modality in cfg.teacher_modalities  (already at target_sr)
            - 'label': integer label array at target_sr
    """
    pkl_path = os.path.join(cfg.data_dir, subject_id, f"{subject_id}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    processed: Dict[str, np.ndarray] = {}

    # Extract all unique modalities needed (teacher_modalities is a superset)
    all_modalities = list(dict.fromkeys(cfg.teacher_modalities))
    for mod in all_modalities:
        processed[mod] = _extract_modality(data, mod, cfg)

    # Labels come from chest at 700 Hz — downsample by nearest-neighbour
    labels_raw = data["label"].squeeze().astype(np.int32)
    # Chest → target_sr: factor = 700 // 128 ≈ 5 (we use gcd-based resampling for labels)
    from math import gcd
    g = gcd(cfg.chest_sr, cfg.target_sr)
    ds_factor = cfg.chest_sr // g
    up_factor  = cfg.target_sr // g
    # For labels, simple stride-based downsampling is fine (majority vote later)
    # Use the same factor as chest signal resampling (700 → 128 ≈ stride 5.46)
    # Approximate: take every round(native_sr/target_sr)-th sample
    stride = max(1, round(cfg.chest_sr / cfg.target_sr))
    processed["label"] = labels_raw[::stride].astype(np.int32)

    return processed


# ════════════════════════════════════════════════════════════════════════════
#  Sliding-window segmentation
# ════════════════════════════════════════════════════════════════════════════

def _segment_windows(
    modality_arrays: Dict[str, np.ndarray],
    labels: np.ndarray,
    cfg: CFG,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Segment all modalities into sliding windows simultaneously.

    Args:
        modality_arrays: Dict[modality_name → (T,)] at target_sr.
        labels:          (T,) label array at target_sr.
        cfg:             Configuration.

    Returns:
        wins:         Dict[modality_name → (W, seq_len)] windowed arrays.
        label_wins:   (W,) remapped integer labels.
    """
    win_len = cfg.seq_len
    step = int(win_len * (1.0 - cfg.overlap))

    # Align all lengths to the shortest available signal
    min_len = min(len(labels), *[len(v) for v in modality_arrays.values()])

    lists: Dict[str, list] = {k: [] for k in modality_arrays}
    lbl_list = []

    for start in range(0, min_len - win_len + 1, step):
        end = start + win_len
        lbl_window = labels[start:end]

        counts = np.bincount(lbl_window.astype(int), minlength=8)
        majority_label = int(np.argmax(counts))

        if majority_label not in cfg.label_map:
            continue

        for mod, arr in modality_arrays.items():
            lists[mod].append(arr[start:end])
        lbl_list.append(cfg.label_map[majority_label])

    wins = {mod: np.stack(lst, axis=0) for mod, lst in lists.items()}
    label_wins = np.array(lbl_list, dtype=np.int64)
    return wins, label_wins


# ════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ════════════════════════════════════════════════════════════════════════════

class WESADDataset(Dataset):
    """PyTorch dataset returning (x_teacher, x_student, label) tuples.

    ``x_teacher`` shape: ``(N_teacher, seq_len)`` = ``(6, seq_len)``
    ``x_student``  shape: ``(N_student, seq_len)`` = ``(3, seq_len)``
    ``label``      dtype: ``torch.long``

    Teacher modalities index order: chest_ecg, chest_eda, chest_resp,
                                    wrist_bvp, wrist_eda, wrist_temp
    Student modalities index order: wrist_bvp, wrist_eda, wrist_temp
    """

    def __init__(self, subjects: List[str], cfg: CFG, augment: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.augment = augment

        teacher_mods = cfg.teacher_modalities   # 6 modalities
        student_mods = cfg.student_modalities   # 3 modalities (subset of teacher)

        # Accumulators for each modality + labels
        all_wins: Dict[str, list] = {mod: [] for mod in teacher_mods}
        all_labels = []

        for sid in subjects:
            print(f"  [data_loader] Loading {sid} ...")
            raw = _load_subject(sid, cfg)

            # Build {modality → array} at target_sr (already filtered + resampled)
            mod_arrays = {mod: raw[mod] for mod in teacher_mods}

            # Z-score normalize each modality independently
            for mod in teacher_mods:
                arr = mod_arrays[mod]
                mod_arrays[mod] = (arr - arr.mean()) / (arr.std() + 1e-8)

            # Segment all modalities in lock-step
            wins, lbl_w = _segment_windows(mod_arrays, raw["label"], cfg)

            for mod in teacher_mods:
                all_wins[mod].append(wins[mod])
            all_labels.append(lbl_w)

        # Concatenate across subjects
        self._data: Dict[str, np.ndarray] = {
            mod: np.concatenate(all_wins[mod], axis=0) for mod in teacher_mods
        }  # each: (N_total, seq_len)

        self.labels = np.concatenate(all_labels, axis=0)  # (N_total,)
        self._teacher_mods = teacher_mods
        self._student_mods = student_mods

        print(
            f"  [data_loader] Dataset ready — {len(self)} windows, "
            f"class distribution: {np.bincount(self.labels).tolist()}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Teacher tensor: (6, seq_len)
        teacher_channels = [
            torch.from_numpy(self._data[mod][idx]).float()
            for mod in self._teacher_mods
        ]
        x_teacher = torch.stack(teacher_channels, dim=0)

        # Student tensor: (3, seq_len) — subset of teacher
        student_channels = [
            torch.from_numpy(self._data[mod][idx]).float()
            for mod in self._student_mods
        ]
        x_student = torch.stack(student_channels, dim=0)

        if self.augment:
            # Augment teacher and student independently (each is a multi-channel tensor)
            x_teacher = augment_signal(x_teacher, noise_std=self.cfg.noise_std, p_permutation=0.0)
            x_student = augment_signal(x_student, noise_std=self.cfg.noise_std, p_permutation=0.0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_teacher, x_student, label


# ════════════════════════════════════════════════════════════════════════════
#  Missing-modality simulation (applied only to x_student during training)
# ════════════════════════════════════════════════════════════════════════════

class MissingModalityWrapper(Dataset):
    """Wraps WESADDataset and randomly zeros out a student modality.

    Teacher tensor ``x_teacher`` is always returned intact.
    Zeros are applied only to ``x_student``.

    Args:
        base_dataset:   Underlying WESADDataset.
        missing_prob:   Probability of dropping a modality per sample.
        drop_modality:  ``"random"`` (randomly choose which student channel to
                        drop each sample) or an int index into student channels.
    """

    def __init__(
        self,
        base_dataset: WESADDataset,
        missing_prob: float = 0.5,
        drop_modality: str = "random",
    ) -> None:
        self.base = base_dataset
        self.missing_prob = missing_prob
        self.drop_modality = drop_modality
        self._n_student = len(base_dataset.cfg.student_modalities)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x_teacher, x_student, label = self.base[idx]
        # x_teacher: (6, seq_len)  — never corrupted
        # x_student:  (3, seq_len)  — may have 1 channel zeroed

        if torch.rand(1).item() < self.missing_prob:
            if self.drop_modality == "random":
                ch = torch.randint(0, self._n_student, (1,)).item()
            elif isinstance(self.drop_modality, int):
                ch = self.drop_modality
            else:
                # Attempt to parse as int string, e.g. "0", "1", "2"
                try:
                    ch = int(self.drop_modality)
                except ValueError:
                    raise ValueError(
                        f"drop_modality must be 'random' or an int-like string. "
                        f"Got: '{self.drop_modality}'"
                    )
            x_student = x_student.clone()
            x_student[ch] = 0.0

        return x_teacher, x_student, label


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

    Args:
        cfg:             Configuration object.
        train_subjects:  Subject IDs for training.
        val_subjects:    Subject IDs for validation.
        wrap_missing:    If True, wrap train set with MissingModalityWrapper.

    Returns:
        ``(train_loader, val_loader)``
    """
    if train_subjects is None:
        train_subjects = cfg.all_subjects[:12]
        val_subjects = cfg.all_subjects[12:]
    if val_subjects is None:
        val_subjects = cfg.all_subjects[12:]

    print("[data_loader] Building training set ...")
    train_ds = WESADDataset(train_subjects, cfg, augment=True)

    print("[data_loader] Building validation set ...")
    val_ds = WESADDataset(val_subjects, cfg, augment=False)

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


def extract_baseline_windows(
    subject_id: str,
    cfg: CFG,
    minutes: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the first ``minutes`` of baseline data from a subject.

    Returns teacher_wins (W, 6, seq_len), student_wins (W, 3, seq_len),
    and labels (W,) — all numpy arrays ready to be used in personalization.
    """
    raw = _load_subject(subject_id, cfg)
    teacher_mods = cfg.teacher_modalities
    student_mods = cfg.student_modalities

    mod_arrays = {mod: raw[mod] for mod in teacher_mods}
    for mod in teacher_mods:
        arr = mod_arrays[mod]
        mod_arrays[mod] = (arr - arr.mean()) / (arr.std() + 1e-8)

    labels_ds = raw["label"]
    baseline_mask = (labels_ds == 1)
    max_samples = int(minutes * 60 * cfg.target_sr)
    baseline_indices = np.where(baseline_mask)[0]

    if len(baseline_indices) == 0:
        empty_t = np.zeros((0, len(teacher_mods), cfg.seq_len), dtype=np.float32)
        empty_s = np.zeros((0, len(student_mods), cfg.seq_len), dtype=np.float32)
        return empty_t, empty_s, np.zeros((0,), dtype=np.int64)

    baseline_indices = baseline_indices[:max_samples]
    bl_arrays = {mod: mod_arrays[mod][baseline_indices] for mod in teacher_mods}
    bl_labels = labels_ds[baseline_indices]

    wins, lbl_w = _segment_windows(bl_arrays, bl_labels, cfg)

    teacher_wins = np.stack([wins[mod] for mod in teacher_mods], axis=1)  # (W,6,L)
    student_wins = np.stack([wins[mod] for mod in student_mods], axis=1)  # (W,3,L)

    print(f"  [personalize] Extracted {len(lbl_w)} baseline windows "
          f"({minutes:.1f} min) from {subject_id}")
    return teacher_wins, student_wins, lbl_w
