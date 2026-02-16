"""
evaluate.py — Comprehensive evaluation with Leave-One-Subject-Out (LOSO) CV.

This script evaluates the Teacher and Student models using rigorous LOSO
cross-validation, which is the gold standard for subject-independent
evaluation on WESAD (small subject pool).

Features
--------
- LOSO cross-validation (train on N-1 subjects, test on 1)
- Per-fold and aggregated metrics (Accuracy, Precision, Recall, F1)
- Confusion matrices
- Missing-modality robustness comparison:
    Student (full) vs Student (EDA dropped) vs Student (ECG dropped)
- Results saved to ``./results/`` as CSV and printed summary

Usage
-----
    # Full LOSO evaluation (trains from scratch per fold)
    python evaluate.py

    # Quick test with subset of subjects
    python evaluate.py --subjects S2 S3 S4 S5

    # Evaluate only the Student (skip teacher training per fold)
    python evaluate.py --phase student

    # Evaluate pre-trained models on a single fold
    python evaluate.py --test_subject S15 --teacher_ckpt ./checkpoints/teacher_best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from data_loader import WESADDataset, MissingModalityWrapper, build_dataloaders
from teacher import TeacherModel
from student import StudentModel
from train_kd import (
    seed_everything,
    train_teacher,
    train_student_kd,
    evaluate as eval_fn,
    compute_accuracy,
    compute_f1,
)


# ════════════════════════════════════════════════════════════════════════════
#  Detailed Metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_detailed_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
) -> Dict[str, float]:
    """Compute per-class and macro-averaged metrics.

    Args:
        logits: ``(N, C)`` raw model outputs.
        targets: ``(N,)`` ground-truth labels.
        class_names: List of human-readable class names.

    Returns:
        Dictionary with per-class P/R/F1, macro averages, and accuracy.
    """
    preds = logits.argmax(dim=1)
    num_classes = len(class_names)
    metrics: Dict[str, float] = {}

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()

        p = (tp / (tp + fp + 1e-8)).item()
        r = (tp / (tp + fn + 1e-8)).item()
        f = (2 * p * r / (p + r + 1e-8)) if (p + r) > 0 else 0.0

        name = class_names[c]
        metrics[f"precision_{name}"] = p
        metrics[f"recall_{name}"] = r
        metrics[f"f1_{name}"] = f

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    metrics["macro_precision"] = sum(precisions) / num_classes
    metrics["macro_recall"] = sum(recalls) / num_classes
    metrics["macro_f1"] = sum(f1s) / num_classes
    metrics["accuracy"] = (preds == targets).float().mean().item() * 100.0

    return metrics


def compute_confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> np.ndarray:
    """Compute confusion matrix (rows = true, cols = predicted).

    Returns:
        ``(num_classes, num_classes)`` integer array.
    """
    preds = logits.argmax(dim=1).numpy()
    targets = targets.numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def format_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> str:
    """Format confusion matrix as a readable string."""
    header = "         " + "  ".join(f"{n:>10s}" for n in class_names)
    lines = [header]
    for i, name in enumerate(class_names):
        row = f"{name:>8s} " + "  ".join(f"{cm[i, j]:>10d}" for j in range(len(class_names)))
        lines.append(row)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Model forward pass (collect all predictions)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model on entire loader, return all logits and targets on CPU.

    Returns:
        ``(all_logits, all_targets)`` both on CPU.
    """
    model.eval()
    all_logits, all_targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.cpu())
        all_targets.append(y)

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


@torch.no_grad()
def collect_predictions_missing(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    drop_channel: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as collect_predictions but zeros out a specific channel.

    Args:
        drop_channel: 0 = drop ECG, 1 = drop EDA.
    """
    model.eval()
    all_logits, all_targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        x[:, drop_channel, :] = 0.0  # Zero out the channel
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.cpu())
        all_targets.append(y)

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


# ════════════════════════════════════════════════════════════════════════════
#  Single LOSO fold
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """Stores results for a single LOSO fold."""
    test_subject: str
    teacher_metrics: Dict[str, float] = field(default_factory=dict)
    student_full_metrics: Dict[str, float] = field(default_factory=dict)
    student_no_eda_metrics: Dict[str, float] = field(default_factory=dict)
    student_no_ecg_metrics: Dict[str, float] = field(default_factory=dict)
    teacher_cm: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=int))
    student_cm: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=int))


def run_fold(
    test_subject: str,
    all_subjects: List[str],
    cfg: CFG,
    phase: str = "both",
    teacher_ckpt: Optional[str] = None,
) -> FoldResult:
    """Run a single LOSO fold: train on all subjects except test_subject,
    then evaluate Teacher + Student under various modality conditions.

    Args:
        test_subject: Subject ID to hold out for testing.
        all_subjects: Full list of subject IDs.
        cfg: Configuration.
        phase: ``"both"`` (train both), ``"student"`` (train student only),
            ``"eval"`` (evaluate pre-trained only).
        teacher_ckpt: Path to pre-trained teacher (for ``phase="student"``
            or ``"eval"``).

    Returns:
        FoldResult with all metrics.
    """
    device = cfg.device
    result = FoldResult(test_subject=test_subject)

    train_subjects = [s for s in all_subjects if s != test_subject]
    print(f"\n{'─' * 60}")
    print(f"  LOSO Fold: test={test_subject}, train={train_subjects}")
    print(f"{'─' * 60}")

    # ── Build test loader (always full modality for fair evaluation) ────
    print(f"  Loading test data ({test_subject}) ...")
    test_ds = WESADDataset([test_subject], cfg)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ── Build train + val loaders ──────────────────────────────────────
    # Use last train subject as validation
    val_subject = train_subjects[-1]
    actual_train = train_subjects[:-1]

    print(f"  Loading train data ({actual_train}) ...")
    train_ds = WESADDataset(actual_train, cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    print(f"  Loading val data ({val_subject}) ...")
    val_ds = WESADDataset([val_subject], cfg)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ── Teacher ────────────────────────────────────────────────────────
    teacher = TeacherModel(cfg)

    if phase in ("both",):
        print(f"  Training Teacher ...")
        teacher = train_teacher(teacher, train_loader, val_loader, cfg)
    elif teacher_ckpt and os.path.exists(teacher_ckpt):
        teacher.load_state_dict(
            torch.load(teacher_ckpt, map_location=device, weights_only=True)
        )
        teacher = teacher.to(device).eval()
    else:
        fold_ckpt = os.path.join(cfg.checkpoint_dir, "teacher_best.pt")
        if os.path.exists(fold_ckpt):
            teacher.load_state_dict(
                torch.load(fold_ckpt, map_location=device, weights_only=True)
            )
            teacher = teacher.to(device).eval()

    # Evaluate Teacher on test set
    print(f"  Evaluating Teacher on {test_subject} ...")
    t_logits, t_targets = collect_predictions(teacher, test_loader, device)
    result.teacher_metrics = compute_detailed_metrics(
        t_logits, t_targets, cfg.class_names
    )
    result.teacher_cm = compute_confusion_matrix(t_logits, t_targets, cfg.num_classes)

    # ── Student (KD) ──────────────────────────────────────────────────
    if phase in ("both", "student"):
        # Wrap train set with missing modality for KD
        train_ds_missing = MissingModalityWrapper(
            train_ds,
            missing_prob=cfg.missing_prob,
            drop_modality=cfg.drop_modality,
        )
        train_loader_kd = DataLoader(
            train_ds_missing, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        student = StudentModel(cfg)
        print(f"  Training Student (KD) ...")
        student = train_student_kd(
            teacher, student, train_loader_kd, val_loader, cfg,
        )
    else:
        student = StudentModel(cfg)
        stu_ckpt = os.path.join(cfg.checkpoint_dir, "student_best.pt")
        if os.path.exists(stu_ckpt):
            student.load_state_dict(
                torch.load(stu_ckpt, map_location=device, weights_only=True)
            )
        student = student.to(device).eval()

    # Evaluate Student — 3 scenarios
    print(f"  Evaluating Student on {test_subject} ...")

    # 1. Full modality
    s_logits, s_targets = collect_predictions(student, test_loader, device)
    result.student_full_metrics = compute_detailed_metrics(
        s_logits, s_targets, cfg.class_names
    )
    result.student_cm = compute_confusion_matrix(s_logits, s_targets, cfg.num_classes)

    # 2. Missing EDA (drop channel 1)
    s_logits_no_eda, _ = collect_predictions_missing(
        student, test_loader, device, drop_channel=1
    )
    result.student_no_eda_metrics = compute_detailed_metrics(
        s_logits_no_eda, s_targets, cfg.class_names
    )

    # 3. Missing ECG (drop channel 0)
    s_logits_no_ecg, _ = collect_predictions_missing(
        student, test_loader, device, drop_channel=0
    )
    result.student_no_ecg_metrics = compute_detailed_metrics(
        s_logits_no_ecg, s_targets, cfg.class_names
    )

    # ── Print fold summary ────────────────────────────────────────────
    print(f"\n  ╔═══ Fold {test_subject} Results ═══")
    print(f"  ║ Teacher         : Acc={result.teacher_metrics['accuracy']:.1f}%  "
          f"F1={result.teacher_metrics['macro_f1']:.3f}")
    print(f"  ║ Student (full)  : Acc={result.student_full_metrics['accuracy']:.1f}%  "
          f"F1={result.student_full_metrics['macro_f1']:.3f}")
    print(f"  ║ Student (no EDA): Acc={result.student_no_eda_metrics['accuracy']:.1f}%  "
          f"F1={result.student_no_eda_metrics['macro_f1']:.3f}")
    print(f"  ║ Student (no ECG): Acc={result.student_no_ecg_metrics['accuracy']:.1f}%  "
          f"F1={result.student_no_ecg_metrics['macro_f1']:.3f}")
    print(f"  ╚{'═' * 35}")

    # Clean up GPU memory between folds
    del teacher, student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ════════════════════════════════════════════════════════════════════════════
#  Aggregate results across all LOSO folds
# ════════════════════════════════════════════════════════════════════════════

def aggregate_results(
    fold_results: List[FoldResult],
    cfg: CFG,
) -> None:
    """Print and save aggregated LOSO results across all folds."""

    results_dir = os.path.join(".", "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Collect per-fold numbers ──────────────────────────────────────
    scenarios = {
        "Teacher (full)": "teacher_metrics",
        "Student (full)": "student_full_metrics",
        "Student (no EDA)": "student_no_eda_metrics",
        "Student (no ECG)": "student_no_ecg_metrics",
    }

    print("\n" + "=" * 80)
    print("  LOSO Cross-Validation — Aggregated Results")
    print("=" * 80)

    # Summary table header
    print(f"\n{'Scenario':<22s} {'Accuracy':>10s} {'Macro-P':>10s} "
          f"{'Macro-R':>10s} {'Macro-F1':>10s}")
    print("─" * 64)

    summary_rows = []
    for scenario_name, attr_name in scenarios.items():
        accs, precs, recs, f1s = [], [], [], []
        for r in fold_results:
            m = getattr(r, attr_name)
            accs.append(m["accuracy"])
            precs.append(m["macro_precision"])
            recs.append(m["macro_recall"])
            f1s.append(m["macro_f1"])

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_p = np.mean(precs)
        mean_r = np.mean(recs)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)

        print(f"{scenario_name:<22s} {mean_acc:>7.1f}±{std_acc:<4.1f} "
              f"{mean_p:>9.3f}  {mean_r:>9.3f}  {mean_f1:>7.3f}±{std_f1:<5.3f}")

        summary_rows.append({
            "scenario": scenario_name,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "macro_precision": mean_p,
            "macro_recall": mean_r,
            "macro_f1_mean": mean_f1,
            "macro_f1_std": std_f1,
        })

    # ── Aggregated confusion matrices ─────────────────────────────────
    print("\n  Aggregated Confusion Matrix — Teacher")
    teacher_cm = sum(r.teacher_cm for r in fold_results)
    print(format_confusion_matrix(teacher_cm, cfg.class_names))

    print("\n  Aggregated Confusion Matrix — Student (full modality)")
    student_cm = sum(r.student_cm for r in fold_results)
    print(format_confusion_matrix(student_cm, cfg.class_names))

    # ── Per-fold detail table ─────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"{'Fold':<8s} {'Teacher Acc':>12s} {'Stu(full)':>12s} "
          f"{'Stu(noEDA)':>12s} {'Stu(noECG)':>12s}")
    print(f"{'─' * 80}")
    for r in fold_results:
        print(f"{r.test_subject:<8s} "
              f"{r.teacher_metrics['accuracy']:>11.1f}% "
              f"{r.student_full_metrics['accuracy']:>11.1f}% "
              f"{r.student_no_eda_metrics['accuracy']:>11.1f}% "
              f"{r.student_no_ecg_metrics['accuracy']:>11.1f}%")

    # ── Model size comparison ─────────────────────────────────────────
    teacher = TeacherModel(cfg)
    student = StudentModel(cfg)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())

    print(f"\n{'─' * 50}")
    print(f"  Model Size Comparison")
    print(f"{'─' * 50}")
    print(f"  Teacher : {t_params:>12,} parameters")
    print(f"  Student : {s_params:>12,} parameters")
    print(f"  Ratio   : {t_params / s_params:>12.1f}× compression")

    # ── Save CSV ──────────────────────────────────────────────────────
    # Summary CSV
    summary_csv = os.path.join(results_dir, "loso_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  Summary saved to: {summary_csv}")

    # Per-fold CSV
    fold_csv = os.path.join(results_dir, "loso_per_fold.csv")
    with open(fold_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_subject",
            "teacher_acc", "teacher_f1",
            "student_full_acc", "student_full_f1",
            "student_no_eda_acc", "student_no_eda_f1",
            "student_no_ecg_acc", "student_no_ecg_f1",
        ])
        for r in fold_results:
            writer.writerow([
                r.test_subject,
                f"{r.teacher_metrics['accuracy']:.2f}",
                f"{r.teacher_metrics['macro_f1']:.4f}",
                f"{r.student_full_metrics['accuracy']:.2f}",
                f"{r.student_full_metrics['macro_f1']:.4f}",
                f"{r.student_no_eda_metrics['accuracy']:.2f}",
                f"{r.student_no_eda_metrics['macro_f1']:.4f}",
                f"{r.student_no_ecg_metrics['accuracy']:.2f}",
                f"{r.student_no_ecg_metrics['macro_f1']:.4f}",
            ])
    print(f"  Per-fold saved to: {fold_csv}")

    # Confusion matrix CSV
    for name, cm in [("teacher", teacher_cm), ("student", student_cm)]:
        cm_csv = os.path.join(results_dir, f"confusion_matrix_{name}.csv")
        with open(cm_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + cfg.class_names)
            for i, cls_name in enumerate(cfg.class_names):
                writer.writerow([cls_name] + cm[i].tolist())
        print(f"  Confusion matrix saved to: {cm_csv}")


# ════════════════════════════════════════════════════════════════════════════
#  CLI & Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO Cross-Validation Evaluation"
    )
    parser.add_argument(
        "--subjects", type=str, nargs="+", default=None,
        help="Subset of subjects for LOSO (default: all 15).",
    )
    parser.add_argument(
        "--test_subject", type=str, default=None,
        help="Run a single fold for this test subject only.",
    )
    parser.add_argument(
        "--phase", type=str, default="both",
        choices=["both", "student", "eval"],
        help="'both' = train teacher+student per fold, "
             "'student' = use pre-trained teacher, "
             "'eval' = evaluate pre-trained checkpoints only.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for per-fold training.")
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFG()

    if args.epochs is not None:
        cfg.epochs_teacher = args.epochs
        cfg.epochs_student = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    subjects = args.subjects if args.subjects else cfg.all_subjects

    seed_everything(cfg.seed)
    print(f"[evaluate] Device  : {cfg.device}")
    print(f"[evaluate] Subjects: {subjects}")
    print(f"[evaluate] Phase   : {args.phase}")

    start_time = time.time()

    # ── Run LOSO folds ────────────────────────────────────────────────
    if args.test_subject:
        # Single fold
        folds = [args.test_subject]
    else:
        folds = subjects

    fold_results: List[FoldResult] = []
    for i, test_subj in enumerate(folds):
        print(f"\n{'█' * 60}")
        print(f"  LOSO FOLD {i + 1}/{len(folds)} — Testing on {test_subj}")
        print(f"{'█' * 60}")

        result = run_fold(
            test_subject=test_subj,
            all_subjects=subjects,
            cfg=cfg,
            phase=args.phase,
            teacher_ckpt=args.teacher_ckpt,
        )
        fold_results.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────
    aggregate_results(fold_results, cfg)

    elapsed = time.time() - start_time
    print(f"\n[evaluate] Total time: {elapsed / 60:.1f} minutes")
    print("[evaluate] Done!")


if __name__ == "__main__":
    main()
