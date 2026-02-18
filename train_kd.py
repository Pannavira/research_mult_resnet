"""
train_kd.py — Two-phase training loop for the Missing-Modality KD framework.

Phase 1  Train the Teacher (MulT + Deep1DResNet) on full multimodal data.
Phase 2  Freeze the Teacher, train the Student with a combined KD loss
         (CE + KL + MSE) while simulating missing modalities.

Usage
-----
    # Train with default config
    python train_kd.py

    # Quick smoke test (1 epoch, 2 subjects, small batch)
    python train_kd.py --epochs 1 --batch_size 4 --subjects S2 S3
"""

from __future__ import annotations

import argparse
import os
import gc
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ── AMP compatibility shim (works on CPU-only and older PyTorch builds) ────
_USE_CUDA = torch.cuda.is_available()

try:
    # PyTorch ≥ 2.4 new-style API
    from torch.amp import GradScaler, autocast as _autocast
    def amp_autocast():
        return _autocast("cuda", enabled=_USE_CUDA)
except ImportError:
    try:
        # PyTorch < 2.4 legacy API
        from torch.cuda.amp import GradScaler, autocast as _autocast
        def amp_autocast():
            return _autocast(enabled=_USE_CUDA)
    except ImportError:
        # CPU-only build — no AMP at all
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
    """Set seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════════════
#  Knowledge-Distillation loss
# ════════════════════════════════════════════════════════════════════════════

class KDLoss(nn.Module):
    """Combined Knowledge Distillation loss.

    ``L = α · CE(student, y) + β · KL(student_soft, teacher_soft) · T²
         + γ · MSE(student_feats, teacher_feats)``

    Feature-based MSE works as follows (memory-efficient):
        1. Teacher attention ``(B, T', T')`` → mean over key-dim → ``(B, T')``
        2. Average the two cross-attention maps → ``(B, T')``
        3. Adaptive-pool ``T'`` down to a compact ``align_dim`` (default 64)
        4. Project each student SE weight ``(B, C_i)`` → ``(B, align_dim)``
        5. MSE between the projected SE weights and the compact attention summary

    This preserves the feature-distillation signal while keeping projector
    params tiny (~14K vs ~51M for the full-flattened approach).
    """

    # Compact alignment dimension (tune-free; 64 works well)
    ALIGN_DIM = 64

    def __init__(
        self,
        cfg: CFG,
        student_dims: List[int],
        teacher_attn_dim: int,          # kept for API compat (unused now)
    ) -> None:
        super().__init__()
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.gamma = cfg.gamma
        self.T = cfg.temperature

        self.ce = nn.CrossEntropyLoss()

        # Adaptive pool compresses teacher attention summary (T' → ALIGN_DIM)
        self.attn_pool = nn.AdaptiveAvgPool1d(self.ALIGN_DIM)

        # Tiny projectors: student SE weights → compact align space
        # e.g. Linear(32→64), Linear(64→64), Linear(128→64)  — ~14K params
        self.projectors = nn.ModuleList([
            nn.Linear(s_dim, self.ALIGN_DIM)
            for s_dim in student_dims
        ])

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_se_weights: List[torch.Tensor],
        teacher_attn_weights: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            student_logits: ``(B, C)``
            teacher_logits: ``(B, C)``  — detached (teacher is frozen)
            student_se_weights: list of ``(B, C_i)`` tensors
            teacher_attn_weights: list of 2 attention matrices ``(B, T', T')``
            targets: ``(B,)``  ground-truth labels

        Returns:
            total_loss: scalar
            components: dict with ``'ce'``, ``'kl'``, ``'mse'`` for logging
        """
        # ── Task loss (Cross-Entropy) ──────────────────────────────────────
        loss_ce = self.ce(student_logits, targets)

        # ── Response-based KD loss (KL Divergence) ─────────────────────────
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        loss_kl = F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (self.T ** 2)

        # ── Feature-based KD loss (MSE) ────────────────────────────────────
        # Step 1: Compress each attention map (B, T', T') → mean over keys
        #         → (B, T') = per-query attention summary
        t_summaries = []
        for attn in teacher_attn_weights:
            t_summaries.append(attn.mean(dim=-1))   # shape: (B, T')

        # Step 2: Average the two cross-attention summaries
        teacher_summary = torch.stack(t_summaries, dim=0).mean(dim=0)
        # teacher_summary shape: (B, T')

        # Step 3: Adaptive pool T' → ALIGN_DIM (64)
        teacher_compact = self.attn_pool(
            teacher_summary.unsqueeze(1)             # shape: (B, 1, T')
        ).squeeze(1)                                 # shape: (B, 64)

        # Step 4: Align each SE weight with the compact attention target
        loss_mse = torch.tensor(0.0, device=student_logits.device)
        for proj, se_w in zip(self.projectors, student_se_weights):
            projected = proj(se_w)                   # shape: (B, 64)
            loss_mse = loss_mse + F.mse_loss(projected, teacher_compact.detach())
        loss_mse = loss_mse / len(self.projectors)

        # ── Total ──────────────────────────────────────────────────────────
        total = self.alpha * loss_ce + self.beta * loss_kl + self.gamma * loss_mse

        components = {
            "ce": loss_ce.item(),
            "kl": loss_kl.item(),
            "mse": loss_mse.item(),
            "total": total.item(),
        }
        return total, components


# ════════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in percent."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def compute_f1(logits: torch.Tensor, targets: torch.Tensor,
               num_classes: int = 3) -> float:
    """Macro-averaged F1 score."""
    preds = logits.argmax(dim=1)
    f1_per_class = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1.item())
    return sum(f1_per_class) / num_classes


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Train Teacher
# ════════════════════════════════════════════════════════════════════════════

def train_teacher(
    teacher: TeacherModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
) -> TeacherModel:
    """Train the Teacher model on full multimodal data.

    Args:
        teacher: Uninitialised Teacher model.
        train_loader: Training data (full modality).
        val_loader: Validation data.
        cfg: Configuration.

    Returns:
        Trained Teacher model (best validation checkpoint).
    """
    device = cfg.device
    teacher = teacher.to(device)

    # ── Compute inverse-frequency class weights from training data ──────
    train_ds = train_loader.dataset
    labels = train_ds.labels  # numpy array from WESADDataset
    class_counts = np.bincount(labels, minlength=cfg.num_classes).astype(np.float32)
    class_weights = (1.0 / class_counts) * class_counts.sum() / cfg.num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[Teacher] Class counts : {class_counts.astype(int).tolist()}")
    print(f"[Teacher] Class weights: {class_weights.tolist()}")

    optimizer = AdamW(teacher.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs_teacher)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler() if _USE_CUDA else GradScaler()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs_teacher + 1):
        # ── Train ──────────────────────────────────────────────────────────
        teacher.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)  # shape: (B, 2, seq_len)
            y = y.to(device, non_blocking=True)   # shape: (B,)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                logits, _, _ = teacher(x)          # shape: (B, num_classes)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += x.size(0)

        scheduler.step()
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total * 100

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_acc, val_f1 = evaluate(teacher, val_loader, criterion, cfg)

        print(
            f"[Teacher] Epoch {epoch:3d}/{cfg.epochs_teacher} │ "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:.1f}% │ "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:.1f}%  F1 {val_f1:.3f} │ "
            f"LR {scheduler.get_last_lr()[0]:.2e}  "
            f"ES {patience_counter}/{cfg.early_stopping_patience}"
        )

        # ── Early stopping on val loss ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            print(f"[Teacher] Early stopping triggered at epoch {epoch} "
                  f"(val loss did not improve for {cfg.early_stopping_patience} epochs)")
            break

        # Save best (based on val accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                teacher.state_dict(),
                os.path.join(cfg.checkpoint_dir, "teacher_best.pt"),
            )

    # Reload best
    teacher.load_state_dict(
        torch.load(os.path.join(cfg.checkpoint_dir, "teacher_best.pt"),
                    map_location=device, weights_only=True)
    )
    print(f"[Teacher] Best val accuracy: {best_val_acc:.1f}%")
    return teacher


# ════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Train Student with KD
# ════════════════════════════════════════════════════════════════════════════

def train_student_kd(
    teacher: TeacherModel,
    student: StudentModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
) -> StudentModel:
    """Train the Student model via knowledge distillation from a frozen Teacher.

    The training set should be wrapped with ``MissingModalityWrapper`` so
    that the student learns to cope with dropped modalities.

    Args:
        teacher: Pre-trained (frozen) Teacher model.
        student: Uninitialised Student model.
        train_loader: Training data (with missing-modality simulation).
        val_loader: Validation data (full modality, for fair comparison).
        cfg: Configuration.

    Returns:
        Trained Student model (best validation checkpoint).
    """
    device = cfg.device
    teacher = teacher.to(device).eval()
    student = student.to(device)

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    # Probe teacher to get attention map size
    print("[KD] Probing teacher for attention map size ...")
    with torch.no_grad():
        probe_x = torch.randn(1, 2, cfg.seq_len, device=device)
        _, attn1, attn2 = teacher(probe_x)
        teacher_attn_dim = attn1.view(1, -1).size(1)  # T' * T'
        del probe_x, attn1, attn2  # free probe tensors immediately
    if _USE_CUDA:
        torch.cuda.empty_cache()
    print(f"[KD] Teacher attention flattened dim: {teacher_attn_dim}")
    print(f"[KD] Using compact alignment dim: {KDLoss.ALIGN_DIM}")

    # Build KD loss
    student_dims = cfg.student_channels  # [32, 64, 128]
    kd_loss_fn = KDLoss(cfg, student_dims, teacher_attn_dim).to(device)

    # Optimizer covers both student parameters AND the projector parameters
    all_params = list(student.parameters()) + list(kd_loss_fn.projectors.parameters())
    optimizer = AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs_student)
    scaler = GradScaler() if _USE_CUDA else GradScaler()

    best_val_acc = 0.0
    criterion_val = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs_student + 1):
        # ── Train ──────────────────────────────────────────────────────────
        student.train()
        kd_loss_fn.train()
        epoch_losses = {"ce": 0.0, "kl": 0.0, "mse": 0.0, "total": 0.0}
        running_correct, running_total = 0, 0

        num_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)  # shape: (B, 2, seq_len)
            y = y.to(device, non_blocking=True)   # shape: (B,)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                # Teacher forward (no grad)
                with torch.no_grad():
                    t_logits, t_attn1, t_attn2 = teacher(x)

                # Student forward
                s_logits, s_se_weights = student(x)

                # KD loss
                loss, comps = kd_loss_fn(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    student_se_weights=s_se_weights,
                    teacher_attn_weights=[t_attn1, t_attn2],
                    targets=y,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Record metrics BEFORE freeing tensors
            for k in epoch_losses:
                epoch_losses[k] += comps[k] * x.size(0)
            running_correct += (s_logits.argmax(1) == y).sum().item()
            running_total += x.size(0)

            # Free ALL intermediate tensors after use to save VRAM
            del t_logits, t_attn1, t_attn2, s_logits, s_se_weights, loss, x, y

            # Per-batch progress
            print(f"\r  batch {batch_idx+1}/{num_batches} "
                  f"loss={comps['total']:.4f}", end="", flush=True)
        print()  # newline after progress

        # Clear CUDA cache every 10 epochs to reduce fragmentation
        if _USE_CUDA and epoch % 10 == 0:
            torch.cuda.empty_cache()

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= running_total
        train_acc = running_correct / running_total * 100

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_acc, val_f1 = evaluate(student, val_loader, criterion_val, cfg)

        print(
            f"[Student] Epoch {epoch:3d}/{cfg.epochs_student} │ "
            f"CE {epoch_losses['ce']:.4f}  KL {epoch_losses['kl']:.4f}  "
            f"MSE {epoch_losses['mse']:.4f}  Total {epoch_losses['total']:.4f} │ "
            f"Acc {train_acc:.1f}% │ "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:.1f}%  F1 {val_f1:.3f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                student.state_dict(),
                os.path.join(cfg.checkpoint_dir, "student_best.pt"),
            )

    # Reload best
    student.load_state_dict(
        torch.load(os.path.join(cfg.checkpoint_dir, "student_best.pt"),
                    map_location=device, weights_only=True)
    )
    print(f"[Student] Best val accuracy: {best_val_acc:.1f}%")
    return student


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation helper
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: CFG,
) -> Tuple[float, float, float]:
    """Evaluate a model on the given loader.

    Returns:
        ``(avg_loss, accuracy_%, macro_f1)``
    """
    device = cfg.device
    model.eval()
    total_loss, total = 0.0, 0
    all_logits, all_targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Handle both teacher (returns 3 vals) and student (returns 2 vals)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / total
    acc = compute_accuracy(all_logits, all_targets)
    f1 = compute_f1(all_logits, all_targets, cfg.num_classes)

    return avg_loss, acc, f1


# ════════════════════════════════════════════════════════════════════════════
#  CLI & Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Missing-Modality KD — Teacher/Student Training"
    )
    parser.add_argument(
        "--phase", type=str, default="both",
        choices=["teacher", "student", "both"],
        help="Which phase to run: 'teacher', 'student', or 'both'.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count for both phases.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--subjects", type=str, nargs="+", default=None,
        help="Subset of subjects to use (e.g. S2 S3). Default: all.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to pre-trained teacher checkpoint for "
                             "Phase 2 only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFG()

    # Apply CLI overrides
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
    print(f"[main] Device : {device}")
    print(f"[main] Subjects: {cfg.all_subjects}")
    print(f"[main] Batch   : {cfg.batch_size}")

    # ── Split subjects: first ~80 % train, rest val ────────────────────────
    n = len(cfg.all_subjects)
    split = max(1, int(n * 0.8))
    train_subjects = cfg.all_subjects[:split]
    val_subjects = cfg.all_subjects[split:] if split < n else cfg.all_subjects[-1:]

    print(f"[main] Train subjects: {train_subjects}")
    print(f"[main] Val   subjects: {val_subjects}")

    # ── Phase 1: Teacher ───────────────────────────────────────────────────
    teacher = TeacherModel(cfg)
    print(f"[main] Teacher params: "
          f"{sum(p.numel() for p in teacher.parameters()):,}")

    if args.phase in ("teacher", "both"):
        print("\n" + "=" * 72)
        print("  PHASE 1 — Training Teacher")
        print("=" * 72)

        train_loader, val_loader = build_dataloaders(
            cfg,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wrap_missing=False,
        )
        teacher = train_teacher(teacher, train_loader, val_loader, cfg)

    elif args.teacher_ckpt is not None:
        # Load pre-trained teacher
        teacher.load_state_dict(
            torch.load(args.teacher_ckpt, map_location=device, weights_only=True)
        )
        teacher = teacher.to(device).eval()
        print(f"[main] Loaded teacher from {args.teacher_ckpt}")
    else:
        # Try default checkpoint
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
        # Free Phase 1 memory before starting Phase 2
        gc.collect()
        if _USE_CUDA:
            torch.cuda.empty_cache()
            print(f"[main] CUDA memory free: "
                  f"{torch.cuda.mem_get_info()[0] / 1024**2:.0f} MiB")

        print("\n" + "=" * 72)
        print("  PHASE 2 — Training Student (Knowledge Distillation)")
        print("=" * 72)

        student = StudentModel(cfg)
        print(f"[main] Student params: "
              f"{sum(p.numel() for p in student.parameters()):,}")

        # Rebuild dataloaders WITH missing-modality simulation
        train_loader, val_loader = build_dataloaders(
            cfg,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wrap_missing=True,   # ← wraps train set with MissingModalityWrapper
        )
        student = train_student_kd(
            teacher, student, train_loader, val_loader, cfg,
        )

    print("\n[main] Done! Checkpoints saved to:", cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
