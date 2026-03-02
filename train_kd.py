"""
train_kd.py — Two-phase Cross-Modal Knowledge Distillation training loop.

Phase 1  Train the Teacher (6-modal ResNet+Transformer) on full privileged data.
Phase 2  Freeze the Teacher; train Student (3-modal wrist) with combined KD loss.

In Phase 2 every batch yields (x_teacher, x_student, y):
    - Teacher receives x_teacher  (B, 6, seq_len)
    - Student receives x_student  (B, 3, seq_len)

KDLoss = α·CE(student, y) + β·KL(student_soft, teacher_soft)·T²
         + γ·MSE_feat  (SE weights → projected → aligned to Teacher attention summary)

Feature-alignment mechanism (MSE_feat):
    Teacher returns attn_map: (B, heads, 6, 6).
    We compute a compact summary: mean over heads, then mean over key-dim → (B, 6).
    Student SE weights at each block: (B, C_i).
    Tiny projectors map (B, C_i) → (B, 6) then MSE vs the teacher summary.
    This enforces the student's channel importance to mimic inter-sensor attention.

Usage
-----
    python train_kd.py                        # full both-phase training
    python train_kd.py --phase teacher        # only train teacher
    python train_kd.py --phase student --teacher_ckpt checkpoints/teacher_best.pt
    python train_kd.py --epochs 1 --batch_size 4 --subjects S2 S3  # smoke test
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ── AMP compatibility shim (CPU-safe + PyTorch ≥ 2.4) ────────────────────
_USE_CUDA = torch.cuda.is_available()

try:
    from torch.amp import GradScaler, autocast as _autocast
    def amp_autocast():
        return _autocast("cuda", enabled=_USE_CUDA)
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast as _autocast
        def amp_autocast():
            return _autocast(enabled=_USE_CUDA)
    except ImportError:
        from contextlib import nullcontext
        class GradScaler:
            def __init__(self, **kwargs): pass
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
        def amp_autocast():
            return nullcontext()

from config import CFG
from data_loader import WESADDataset, MissingModalityWrapper, build_dataloaders
from teacher import TeacherModel
from student import StudentModel


# ════════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════════════
#  Knowledge-Distillation Loss
# ════════════════════════════════════════════════════════════════════════════

class KDLoss(nn.Module):
    """Combined Knowledge Distillation loss for Cross-Modal KD.

    L = α · CE(student, y)
      + β · KL(student_soft, teacher_soft) · T²
      + γ · MSE_feat

    Feature-alignment (MSE_feat) strategy
    ──────────────────────────────────────
    Teacher attention_map:  (B, heads, 6, 6)
        → mean over heads  → (B, 6, 6)
        → mean over key-dim → (B, 6)       ← "teacher inter-sensor summary"

    Student SE weight (block i): (B, C_i)
        → tiny Linear(C_i → 6)             ← projects to same 6-dim space
        → MSE vs teacher inter-sensor summary

    Why 6-dim alignment?  The teacher attends over 6 sensor tokens.
    Forcing SE weights to predict the teacher's inter-sensor attention profile
    transfers which sensors matter most — even though the student never sees
    3 of those sensors.

    Args:
        cfg:            Configuration object (α, β, γ, T, num_classes).
        student_dims:   List of SE output dims, e.g. [64, 128, 256].
        n_teacher_mods: Number of teacher modalities (6) — projection target dim.
        class_weights:  Optional class weight tensor for CE loss.
    """

    # Fixed alignment target dim = number of teacher sensors
    N_TEACHER = 6

    def __init__(
        self,
        cfg: CFG,
        student_dims: List[int],
        n_teacher_mods: int = 6,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = cfg.alpha
        self.beta  = cfg.beta
        self.gamma = cfg.gamma
        self.T     = cfg.temperature

        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        # Tiny linear projectors: SE_dim_i → N_TEACHER (6)
        # e.g. Linear(64→6), Linear(128→6), Linear(256→6) ← very cheap (~1.2K params)
        self.projectors = nn.ModuleList([
            nn.Linear(s_dim, n_teacher_mods)
            for s_dim in student_dims
        ])

    def forward(
        self,
        student_logits: torch.Tensor,           # (B, num_classes)
        teacher_logits: torch.Tensor,           # (B, num_classes)  — detached
        student_se_weights: List[torch.Tensor], # [(B, C_i)]
        teacher_attn_map: torch.Tensor,         # (B, heads, 6, 6)
        targets: torch.Tensor,                  # (B,)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            components: dict with 'ce', 'kl', 'mse', 'total' for logging
        """
        # ── 1. Task loss ─────────────────────────────────────────────────
        loss_ce = self.ce(student_logits, targets)

        # ── 2. Response-based KD (KL Divergence) ─────────────────────────
        s_soft = F.log_softmax(student_logits / self.T, dim=1)
        t_soft = F.softmax(teacher_logits / self.T, dim=1)
        loss_kl = F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.T ** 2)

        # ── 3. Feature-based KD (MSE) ─────────────────────────────────────
        # Compute teacher inter-sensor attention summary: (B, 6)
        #   attn_map: (B, heads, 6, 6)
        #   → mean over heads → (B, 6, 6)
        #   → mean over key-dim → (B, 6)
        with torch.no_grad():
            teacher_summary = teacher_attn_map.mean(dim=1)  # (B, 6, 6)
            teacher_summary = teacher_summary.mean(dim=-1)   # (B, 6)

        loss_mse = torch.tensor(0.0, device=student_logits.device)
        for proj, se_w in zip(self.projectors, student_se_weights):
            projected = proj(se_w)           # (B, 6)
            loss_mse = loss_mse + F.mse_loss(projected, teacher_summary)
        loss_mse = loss_mse / len(self.projectors)

        # ── 4. Total ──────────────────────────────────────────────────────
        total = self.alpha * loss_ce + self.beta * loss_kl + self.gamma * loss_mse

        components = {
            "ce":    loss_ce.item(),
            "kl":    loss_kl.item(),
            "mse":   loss_mse.item(),
            "total": total.item(),
        }
        return total, components


# ════════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def compute_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    preds = logits.argmax(dim=1)
    f1_per_class = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1.item())
    return sum(f1_per_class) / num_classes


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation helper (works for both Teacher and Student)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: CFG,
    use_teacher_input: bool = False,
) -> Tuple[float, float, float]:
    """Evaluate a model on the given loader.

    Args:
        use_teacher_input: If True, pass x_teacher to the model (eval teacher).
                           If False, pass x_student (eval student).

    Returns:
        ``(avg_loss, accuracy_%, macro_f1)``
    """
    device = cfg.device
    model.eval()
    total_loss, total = 0.0, 0
    all_logits, all_targets = [], []

    for x_teacher, x_student, y in loader:
        x_in = x_teacher if use_teacher_input else x_student
        x_in = x_in.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        out = model(x_in)
        logits = out[0] if isinstance(out, tuple) else out

        loss = criterion(logits, y)
        total_loss += loss.item() * x_in.size(0)
        total += x_in.size(0)

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    all_logits  = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / total
    acc      = compute_accuracy(all_logits, all_targets)
    f1       = compute_f1(all_logits, all_targets, cfg.num_classes)

    return avg_loss, acc, f1


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Train Teacher (6-modal input)
# ════════════════════════════════════════════════════════════════════════════

def train_teacher(
    teacher: TeacherModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
) -> TeacherModel:
    """Train the Teacher on all 6 privileged modalities.

    Each batch from the loader yields (x_teacher, x_student, y).
    Only x_teacher (B, 6, seq_len) and y are used here.

    Returns the best Teacher (by val Macro-F1).
    """
    device = cfg.device
    teacher = teacher.to(device)

    # Inverse-frequency class weights
    train_ds = train_loader.dataset
    if hasattr(train_ds, "base"):
        train_ds = train_ds.base
    labels = train_ds.labels
    class_counts = np.bincount(labels, minlength=cfg.num_classes).astype(np.float32)
    class_weights = (1.0 / class_counts) * class_counts.sum() / cfg.num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[Teacher] Class counts : {class_counts.astype(int).tolist()}")
    print(f"[Teacher] Class weights: {class_weights.tolist()}")

    optimizer = AdamW(teacher.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs_teacher)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = GradScaler()

    best_val_f1 = 0.0
    patience_counter = 0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs_teacher + 1):
        teacher.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for batch_idx, (x_teacher_batch, _x_student, y) in enumerate(train_loader):
            # Only use x_teacher (6-modal) and y in Phase 1
            x = x_teacher_batch.to(device, non_blocking=True)  # (B, 6, seq_len)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                logits, _attn = teacher(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss    += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total   += x.size(0)

        scheduler.step()
        train_loss = running_loss    / running_total
        train_acc  = running_correct / running_total * 100

        val_loss, val_acc, val_f1 = evaluate(
            teacher, val_loader, criterion, cfg, use_teacher_input=True
        )

        print(
            f"[Teacher] Epoch {epoch:3d}/{cfg.epochs_teacher} │ "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:.1f}% │ "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:.1f}%  F1 {val_f1:.3f} │ "
            f"LR {scheduler.get_last_lr()[0]:.2e}  "
            f"ES {patience_counter}/{cfg.early_stopping_patience}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                teacher.state_dict(),
                os.path.join(cfg.checkpoint_dir, "teacher_best.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            print(f"[Teacher] Early stopping at epoch {epoch}.")
            break

    teacher.load_state_dict(
        torch.load(
            os.path.join(cfg.checkpoint_dir, "teacher_best.pt"),
            map_location=device, weights_only=True,
        )
    )
    print(f"[Teacher] Best val Macro-F1: {best_val_f1:.3f}")
    return teacher


# ════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Train Student with Cross-Modal KD
# ════════════════════════════════════════════════════════════════════════════

def train_student_kd(
    teacher: TeacherModel,
    student: StudentModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
) -> StudentModel:
    """Train the wrist Student via knowledge distillation from the frozen Teacher.

    Each batch yields (x_teacher, x_student, y):
        - Teacher receives x_teacher (B, 6, seq_len)  → produces logits + attn_map
        - Student receives x_student (B, 3, seq_len)  → produces logits + SE weights

    KD loss aligns:
        • Response: soft labels from teacher → student (KL)
        • Feature:  teacher inter-sensor attention (B,6) → student SE weights (B,C)
                    via tiny Linear projectors (barely any extra params)

    Returns the best Student model (by val Macro-F1 using student wrist inputs).
    """
    device  = cfg.device
    teacher = teacher.to(device).eval()
    student = student.to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    # Class weights from training labels
    base_ds = train_loader.dataset
    if hasattr(base_ds, "base"):
        base_ds = base_ds.base
    labels = base_ds.labels
    class_counts  = np.bincount(labels, minlength=cfg.num_classes).astype(np.float32)
    class_weights = (1.0 / class_counts) * class_counts.sum() / cfg.num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[KD] Class counts : {class_counts.astype(int).tolist()}")
    print(f"[KD] Class weights: {class_weights.tolist()}")

    # KD loss: projects each SE output to (B, 6) to align with teacher summary
    n_teacher_mods = len(cfg.teacher_modalities)  # 6
    student_dims   = cfg.student_channels          # e.g. [64, 128, 256]
    kd_loss_fn = KDLoss(
        cfg,
        student_dims=student_dims,
        n_teacher_mods=n_teacher_mods,
        class_weights=class_weights,
    ).to(device)

    # Optimizer covers student params + projector params in KDLoss
    all_params = list(student.parameters()) + list(kd_loss_fn.projectors.parameters())
    optimizer  = AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg.epochs_student)
    scaler     = GradScaler()

    best_val_f1   = 0.0
    criterion_val = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs_student + 1):
        student.train()
        kd_loss_fn.train()
        epoch_losses = {"ce": 0.0, "kl": 0.0, "mse": 0.0, "total": 0.0}
        running_correct, running_total = 0, 0

        num_batches = len(train_loader)
        for batch_idx, (x_teacher_batch, x_student_batch, y) in enumerate(train_loader):
            x_t = x_teacher_batch.to(device, non_blocking=True)  # (B, 6, L)
            x_s = x_student_batch.to(device, non_blocking=True)  # (B, 3, L)
            y   = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                # Teacher forward — frozen, no grad
                with torch.no_grad():
                    t_logits, t_attn_map = teacher(x_t)  # (B,C), (B,heads,6,6)

                # Student forward
                s_logits, s_se_weights = student(x_s)    # (B,C), [(B,C_i)]

                # KD loss
                loss, comps = kd_loss_fn(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    student_se_weights=s_se_weights,
                    teacher_attn_map=t_attn_map,
                    targets=y,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k in epoch_losses:
                epoch_losses[k] += comps[k] * x_s.size(0)
            running_correct += (s_logits.argmax(1) == y).sum().item()
            running_total   += x_s.size(0)

            # Free intermediate tensors proactively to reduce VRAM pressure
            del t_logits, t_attn_map, s_logits, s_se_weights, loss, x_t, x_s, y

            print(
                f"\r  batch {batch_idx+1}/{num_batches} "
                f"loss={comps['total']:.4f}", end="", flush=True
            )
        print()  # newline after progress bar

        if _USE_CUDA and epoch % 10 == 0:
            torch.cuda.empty_cache()

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= running_total
        train_acc = running_correct / running_total * 100

        # Validate student on wrist-only inputs (no missing mod at val time)
        val_loss, val_acc, val_f1 = evaluate(
            student, val_loader, criterion_val, cfg, use_teacher_input=False
        )

        print(
            f"[Student] Epoch {epoch:3d}/{cfg.epochs_student} │ "
            f"CE {epoch_losses['ce']:.4f}  KL {epoch_losses['kl']:.4f}  "
            f"MSE {epoch_losses['mse']:.4f}  Total {epoch_losses['total']:.4f} │ "
            f"Acc {train_acc:.1f}% │ "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:.1f}%  F1 {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                student.state_dict(),
                os.path.join(cfg.checkpoint_dir, "student_best.pt"),
            )

    student.load_state_dict(
        torch.load(
            os.path.join(cfg.checkpoint_dir, "student_best.pt"),
            map_location=device, weights_only=True,
        )
    )
    print(f"[Student] Best val Macro-F1: {best_val_f1:.3f}")
    return student


# ════════════════════════════════════════════════════════════════════════════
#  CLI & Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Modal KD — Teacher (6-modal) / Student (3-modal wrist) Training"
    )
    parser.add_argument(
        "--phase", type=str, default="both",
        choices=["teacher", "student", "both"],
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count for both phases.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--subjects", type=str, nargs="+", default=None,
        help="Subset of subjects (e.g. S2 S3). Default: all.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Pre-trained teacher checkpoint for Phase 2 only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = CFG()

    if args.epochs is not None:
        cfg.epochs_teacher = args.epochs
        cfg.epochs_student = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.subjects is not None:
        cfg.all_subjects = args.subjects

    seed_everything(cfg.seed)
    device = cfg.device
    print(f"[main] Device  : {device}")
    print(f"[main] Subjects: {cfg.all_subjects}")
    print(f"[main] Batch   : {cfg.batch_size}")
    print(f"[main] Teacher modalities ({len(cfg.teacher_modalities)}): {cfg.teacher_modalities}")
    print(f"[main] Student modalities ({len(cfg.student_modalities)}): {cfg.student_modalities}")

    n = len(cfg.all_subjects)
    split = max(1, int(n * 0.8))
    train_subjects = cfg.all_subjects[:split]
    val_subjects   = cfg.all_subjects[split:] if split < n else cfg.all_subjects[-1:]

    print(f"[main] Train subjects: {train_subjects}")
    print(f"[main] Val   subjects: {val_subjects}")

    # ── Phase 1: Teacher ───────────────────────────────────────────────────
    teacher = TeacherModel(cfg)
    print(f"[main] Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    if args.phase in ("teacher", "both"):
        print("\n" + "=" * 72)
        print("  PHASE 1 — Training Teacher (6 modalities)")
        print("=" * 72)
        # Teacher is trained WITHOUT missing-modality augmentation (it has privileged access)
        train_loader, val_loader = build_dataloaders(
            cfg,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wrap_missing=False,
        )
        teacher = train_teacher(teacher, train_loader, val_loader, cfg)

    elif args.teacher_ckpt is not None:
        teacher.load_state_dict(
            torch.load(args.teacher_ckpt, map_location=device, weights_only=True)
        )
        teacher = teacher.to(device).eval()
        print(f"[main] Loaded teacher from {args.teacher_ckpt}")
    else:
        default_ckpt = os.path.join(cfg.checkpoint_dir, "teacher_best.pt")
        if os.path.exists(default_ckpt):
            teacher.load_state_dict(
                torch.load(default_ckpt, map_location=device, weights_only=True)
            )
            teacher = teacher.to(device).eval()
            print(f"[main] Loaded teacher from {default_ckpt}")
        else:
            raise FileNotFoundError(
                "No teacher checkpoint found. Run Phase 1 first "
                "(--phase teacher) or provide --teacher_ckpt."
            )

    # ── Phase 2: Student with KD ──────────────────────────────────────────
    if args.phase in ("student", "both"):
        gc.collect()
        if _USE_CUDA:
            torch.cuda.empty_cache()
            print(f"[main] CUDA free: {torch.cuda.mem_get_info()[0] / 1024**2:.0f} MiB")

        print("\n" + "=" * 72)
        print("  PHASE 2 — Training Student (3 wrist modalities, KD from 6-modal Teacher)")
        print("=" * 72)

        student = StudentModel(cfg)
        print(f"[main] Student params: {sum(p.numel() for p in student.parameters()):,}")

        # Student train loader uses MissingModalityWrapper (drops x_student channels)
        train_loader, val_loader = build_dataloaders(
            cfg,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wrap_missing=True,
        )
        student = train_student_kd(teacher, student, train_loader, val_loader, cfg)

    print("\n[main] Done! Checkpoints saved to:", cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
