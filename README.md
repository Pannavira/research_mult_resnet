# Missing-Modality Knowledge Distillation for Stress Detection

A deep learning framework for **wearable stress detection** using the [WESAD dataset](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/). The system trains a large **Teacher model** (Deep 1D-ResNet + Multimodal Transformer) and distills its knowledge into a lightweight **Student model** (Depthwise Separable CNN + Squeeze-and-Excitation blocks) that is robust to **missing sensor modalities**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEACHER MODEL (~6M params)                   │
│  ┌───────────────┐        ┌──────────────────┐                  │
│  │ Deep 1D-ResNet│──ECG──▶│                │                      │
│  │  (ECG branch) │        │  MulT-style      │──▶ Logits          │
│  ├───────────────┤        │  Cross-Attention  │──▶ Attention Maps │
│  │ Deep 1D-ResNet│──EDA──▶│                  │                    │    
│  │  (EDA branch) │        └──────────────────┘                  │
│  └───────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
         │ Knowledge Distillation (KL + MSE)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STUDENT MODEL (~31K params)                  │
│  ┌────────────────────┐   ┌────────────────┐                     │
│  │ Depthwise Separable│──▶│ SE Blocks      │──▶ Logits          │
│  │ 1D-CNN (2ch input) │   │ (channel attn) │──▶ SE Weights      │
│  └────────────────────┘   └────────────────┘                     │
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
| α (alpha) | 0.4 | Weight for cross-entropy task loss |
| β (beta)  | 0.4 | Weight for KL divergence (soft labels) |
| γ (gamma) | 0.2 | Weight for feature alignment (MSE) |
| T         | 4.0 | Softmax temperature |

---

## Project Structure

```
research_mult_resnet/
├── config.py           # All hyperparameters & settings (single source of truth)
├── data_loader.py      # WESAD data pipeline (filtering, windowing, DataLoaders)
├── teacher.py          # Teacher: Deep 1D-ResNet + MulT Cross-Attention
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

# Full LOSO (all 15 subjects, default 80 epochs per fold)
python evaluate.py

# Single fold
python evaluate.py --test_subject S15 --epochs 20
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
| `--epochs` | 80 | Number of training epochs (overrides both phases) |
| `--batch_size` | 32 | Batch size |
| `--subjects` | all 15 | Subset of subjects, e.g., `S2 S3 S4` |
| `--teacher_ckpt` | — | Path to pre-trained teacher checkpoint |

### `evaluate.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase` | `both` | `both` / `student` / `eval` |
| `--subjects` | all 15 | Subjects for LOSO |
| `--test_subject` | — | Run single fold for this subject |
| `--epochs` | 80 | Epochs per fold |
| `--batch_size` | 32 | Batch size |
| `--teacher_ckpt` | — | Path to pre-trained teacher |

---

## Data Pipeline

The preprocessing pipeline (handled automatically by `data_loader.py`):

1. **Load** — Read `.pkl` files from WESAD (chest-worn RespiBAN sensor)
2. **Extract** — ECG (channel 0) and EDA (channel 1) signals
3. **Filter** — Butterworth band-pass (0.5–40 Hz) for ECG, low-pass (5 Hz) for EDA
4. **Downsample** — 700 Hz → 128 Hz
5. **Segment** — 60-second sliding windows with 50% overlap
6. **Normalize** — Z-score normalization per window
7. **Missing-Modality Simulation** — Randomly zero out EDA channel during student training

### Label Mapping

| WESAD Label | Class | Index |
|-------------|-------|-------|
| 1 | Baseline | 0 |
| 2 | Stress | 1 |
| 3 | Amusement | 2 |

Labels 0, 4, 5, 6, 7 are excluded (not defined / meditation / ignored).

---

## Model Details

### Teacher — Deep 1D-ResNet + MulT (~6M parameters)

- Two parallel Deep 1D-ResNet branches (ECG and EDA)
- Each branch: 3 stages of residual blocks `[64, 128, 256]` channels
- MulT-style cross-attention fusion between branches
- Outputs: class logits + cross-attention weight matrices

### Student — DS-CNN + SE (~31K parameters)

- Depthwise Separable 1D-CNN (2-channel input)
- 3 stages with SE blocks `[32, 64, 128]` channels
- SE reduction ratio: 4
- ~**196× smaller** than Teacher
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
epochs_teacher = 80
epochs_student = 80
lr = 1e-3
weight_decay = 1e-2

# Knowledge Distillation
temperature = 4.0
alpha = 0.4              # CE weight
beta = 0.4               # KL weight
gamma = 0.2              # MSE weight

# Missing-modality simulation
missing_prob = 0.5       # 50% chance of dropping
drop_modality = "eda"    # Drop EDA by default
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

This is handled automatically (`num_workers=0` on Windows). If issues persist, set manually in `config.py`.

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
