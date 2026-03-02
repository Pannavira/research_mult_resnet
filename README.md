# Missing-Modality Knowledge Distillation for Stress Detections

A deep learning framework for **wearable stress detection** using the [WESAD dataset](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/). The system trains a large **Teacher model** (Deep 1D-ResNet + Multimodal Transformer) and distills its knowledge into a lightweight **Student model** (Depthwise Separable CNN + Squeeze-and-Excitation blocks) that is robust to **missing sensor modalities**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEACHER MODEL                                │
│  ┌───────────────┐        ┌──────────────────┐                  │
│  │ Deep 1D-ResNet│──ECG──▶│                  │                  │
│  │  (ECG branch) │        │  MulT-style      │──▶ Logits        │
│  │  [16,32,64]   │        │  Cross-Attention  │──▶ Attn Maps    │
│  ├───────────────┤        │  (InstanceNorm)   │                  │
│  │ Deep 1D-ResNet│──EDA──▶│                  │                  │
│  │  (EDA branch) │        └──────────────────┘                  │
│  │  [16,32,64]   │                                              │
│  └───────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
         │ Knowledge Distillation (CE + KL + MSE)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STUDENT MODEL                                │
│  ┌────────────────────┐   ┌────────────────┐                    │
│  │ Depthwise Separable│──▶│ SE Blocks      │──▶ Logits          │
│  │ 1D-CNN (2ch input) │   │ (channel attn) │──▶ SE Weights      │
│  │ [64,128,256]       │   │ reduction=4    │                    │
│  └────────────────────┘   └────────────────┘                    │
│  🔇 Robust to missing ECG or EDA channels                      │
└─────────────────────────────────────────────────────────────────┘
```

## Knowledge Distillation Loss

The Student is trained with a combined loss function:

```
L = α · CE(student, labels)               # Task loss
  + β · KL(student_soft, teacher_soft)·T²  # Response-based KD
  + γ · MSE(projected_SE, teacher_attn)    # Feature-based KD
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| α (alpha) | 0.5 | Weight for cross-entropy task loss |
| β (beta)  | 0.5 | Weight for KL divergence (soft labels) |
| γ (gamma) | 1.0 | Weight for feature alignment (MSE) |
| T         | 4.0 | Softmax temperature |

---

## Key Features

### Macro-F1 Early Stopping

Both Teacher and Student training use **Macro-F1 score** (not loss or accuracy) for early stopping and model checkpointing. This is critical for WESAD's imbalanced class distribution where accuracy can be misleading.

- **Teacher**: Saves `teacher_best.pt` when `val_f1` improves; patience = 25 epochs
- **Student**: Saves `student_best.pt` based on best `val_f1`

### Signal Augmentation Pipeline

Three augmentation techniques are applied during training to combat inter-subject variability (`augmentation.py`):

| Augmentation | Probability | Description |
|-------------|-------------|-------------|
| Gaussian Noise | Always (std=0.1) | Adds random noise for noise robustness |
| Random Scaling | 50% | Multiplies signal by random factor (0.8–1.2) per channel |
| Time Permutation | 30% | Splits signal into 5 segments and shuffles order |

### Personalization (Subject Calibration)

For LOSO evaluation, an optional **personalization** step mixes a small amount (default: 2 minutes) of the test subject's *baseline* data into the training set. This adapts the model to individual signal characteristics without leaking stress labels.

```bash
python evaluate.py --personalize
```

### Regularization Strategy

The Teacher uses aggressive regularization to prevent memorizing the small training pool:

- **InstanceNorm** (instead of BatchNorm) — robust to batch-size variation and subject-specific statistics
- **Dropout = 0.5** on ResNet blocks, cross-attention layers, positional encoding, and classifier
- **Small channel sizes** `[16, 32, 64]` — reduced model capacity to prevent overfitting on 13 training subjects

---

## Project Structure

```
research_mult_resnet/
├── config.py           # All hyperparameters & settings (single source of truth)
├── data_loader.py      # WESAD data pipeline (filtering, windowing, DataLoaders)
├── augmentation.py     # Signal augmentation (Gaussian Noise, Random Scaling, Time Permutation)
├── teacher.py          # Teacher: Deep 1D-ResNet + MulT Cross-Attention (InstanceNorm)
├── student.py          # Student: DS-CNN + SE blocks (lightweight, edge-ready)
├── train_kd.py         # Two-phase training loop (Teacher → Student KD)
├── evaluate.py         # LOSO cross-validation evaluation & metrics
├── data/
│   └── WESAD/          # WESAD dataset (place .pkl files here)
│       ├── S2/S2.pkl
│       ├── S3/S3.pkl
│       └── ...
├── checkpoints/        # Saved model weights (auto-created)
└── results/            # Evaluation CSV reports (auto-created)
```

---

## Requirements

- **Python** 3.10+
- **PyTorch** 2.0+ (with CUDA support recommended)
- **NumPy**, **SciPy**

### Installation

```bash
# 1. Install PyTorch with CUDA (adjust cu128 for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

# 2. Install other dependencies
pip install numpy scipy
```

### Dataset Setup

1. Download the [WESAD dataset](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
2. Place subject folders in `./data/WESAD/`:
   ```
   data/WESAD/S2/S2.pkl
   data/WESAD/S3/S3.pkl
   ...
   data/WESAD/S17/S17.pkl
   ```

---

## Usage

### Quick Smoke Test

Verify everything works with a small run (2 subjects, 2 epochs):

```bash
python train_kd.py --epochs 2 --batch_size 4 --subjects S2 S3
```

### Full Training

Train both Teacher (Phase 1) and Student via KD (Phase 2) with all 15 subjects:

```bash
# Recommended: set memory optimization on Windows
# PowerShell:
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Train both phases
python train_kd.py --phase both
```

### Train Phases Separately

```bash
# Phase 1: Train Teacher only
python train_kd.py --phase teacher

# Phase 2: Train Student with KD (using saved teacher checkpoint)
python train_kd.py --phase student --teacher_ckpt ./checkpoints/teacher_best.pt
```

### Evaluation (LOSO Cross-Validation)

Run Leave-One-Subject-Out evaluation for rigorous subject-independent results:

```bash
# Quick LOSO test (subset of subjects)
python evaluate.py --subjects S2 S3 S4 S5 --epochs 10

# Full LOSO (all 15 subjects)
python evaluate.py

# Single fold
python evaluate.py --test_subject S15 --epochs 20

# With personalization (mix test subject baseline into training)
python evaluate.py --personalize

# Teacher-only evaluation
python evaluate.py --phase teacher
```

Results are saved to `./results/`:
- `loso_summary.csv` — Aggregated metrics per scenario
- `loso_per_fold.csv` — Per-fold accuracy & F1
- `confusion_matrix_teacher.csv` / `confusion_matrix_student.csv`

---

## CLI Arguments

### `train_kd.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase` | `both` | Training phase: `teacher`, `student`, or `both` |
| `--epochs` | — | Override epoch count for both phases (default: teacher=100, student=80) |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--subjects` | all 15 | Subset of subjects, e.g., `S2 S3 S4` |
| `--teacher_ckpt` | — | Path to pre-trained teacher checkpoint |

### `evaluate.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase` | `both` | `both` / `teacher` / `student` / `eval` |
| `--subjects` | all 15 | Subjects for LOSO |
| `--test_subject` | — | Run single fold for this subject |
| `--epochs` | — | Override epochs per fold |
| `--batch_size` | 32 | Batch size |
| `--teacher_ckpt` | — | Path to pre-trained teacher |
| `--personalize` | `false` | Mix test subject baseline data into training |

---

## Data Pipeline

The preprocessing pipeline (handled automatically by `data_loader.py`):

1. **Load** — Read `.pkl` files from WESAD (chest-worn RespiBAN sensor)
2. **Extract** — ECG (channel 0) and EDA (channel 1) signals
3. **Filter** — Butterworth band-pass: ECG (0.5–40 Hz), EDA (0.05–5.0 Hz), order 4
4. **Downsample** — 700 Hz → 128 Hz (using `scipy.signal.decimate`)
5. **Normalize** — Per-subject global Z-score normalization
6. **Segment** — 60-second sliding windows with 50% overlap
7. **Augment** — Gaussian Noise + Random Scaling + Time Permutation (training only)
8. **Missing-Modality Simulation** — Randomly zero out ECG or EDA channel (50% prob) during training

### Label Mapping (Binary Classification)

| WESAD Label | Class | Index |
|-------------|-------|-------|
| 1 (Baseline) | Non-Stress | 0 |
| 3 (Amusement) | Non-Stress | 0 |
| 2 (Stress) | Stress | 1 |

Labels 0, 4, 5, 6, 7 are excluded (not defined / meditation / ignored).

---

## Model Details

### Teacher — Deep 1D-ResNet + MulT (with InstanceNorm)

- Two parallel Deep 1D-ResNet branches (ECG and EDA)
- Each branch: 3 stages of residual blocks `[16, 32, 64]` channels (2 blocks per stage)
- InstanceNorm1d (instead of BatchNorm) for subject-invariant normalization
- Dropout 0.5 on all residual blocks, cross-attention, and classifier
- MulT-style cross-attention fusion (4 heads, d_model=64)
- Sinusoidal positional encoding
- Outputs: class logits + cross-attention weight matrices

### Student — DS-CNN + SE

- Depthwise Separable 1D-CNN (2-channel input: ECG + EDA)
- 3 stages with SE blocks `[64, 128, 256]` channels
- SE reduction ratio: 4
- InstanceNorm1d in all convolution blocks
- MaxPool1d(4) between stages
- Classifier: FC → ReLU → Dropout(0.2) → FC
- Robust to zeroed-out modality channels

---

## Configuration

All hyperparameters are centralized in `config.py`. Key settings:

```python
# Signal processing
target_sr = 128          # Hz (downsampled from 700 Hz)
window_sec = 60.0        # seconds
overlap = 0.5            # 50%

# Training
batch_size = 32
epochs_teacher = 100
epochs_student = 80
lr = 1e-4
weight_decay = 1e-2
early_stopping_patience = 25   # Based on Macro-F1

# Data Augmentation
noise_std = 0.1          # Gaussian noise std
# Random Scaling: 0.8–1.2 (50% prob)
# Time Permutation: 5 segments (30% prob)

# Personalization
personalize = False      # Enable via --personalize
personalize_baseline_minutes = 2.0
personalize_finetune_epochs = 5

# Knowledge Distillation
temperature = 4.0
alpha = 0.5              # CE weight
beta = 0.5               # KL weight
gamma = 1.0              # MSE weight

# Missing-modality simulation
missing_prob = 0.5       # 50% chance of dropping
drop_modality = "random" # Randomly drop ECG or EDA

# Teacher architecture
resnet_channels = [16, 32, 64]  # Compact model
attn_heads = 4
attn_dim = 64

# Student architecture
student_channels = [64, 128, 256]
se_reduction = 4
```

---

## Troubleshooting

### CUDA Out of Memory (RTX 3050 / 4GB VRAM)

```powershell
# Set memory optimization
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Use smaller batch size
python train_kd.py --batch_size 8
```

### PyTorch shows CPU even with NVIDIA GPU

Your PyTorch may be a CPU-only build. Reinstall with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### DataLoader hangs on Windows

Set `num_workers=0` in `config.py` (currently set to 4; reduce to 0 if DataLoader hangs):

```python
num_workers: int = 0  # 0 on Windows to avoid hangs
```

---

## Citation

This project uses the **WESAD** (Wearable Stress and Affect Detection) dataset:

```bibtex
@inproceedings{schmidt2018introducing,
  title={Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marber{\-{g}}er, Claus and Van Laerhoven, Kristof},
  booktitle={Proceedings of the 20th ACM International Conference on Multimodal Interaction},
  pages={400--408},
  year={2018}
}
```

---

## License

This project is for academic research purposes (thesis/skripsi).
