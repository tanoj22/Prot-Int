"""
Metrics helpers for multilabel protein localization.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss as sk_hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()  # torch tensor path
    return np.asarray(x)


def compute_metrics(
    y_true: Any,
    y_pred_proba: Any,
    y_pred_binary: Any,
    label_names: Sequence[str],
) -> Dict[str, Any]:
    y_true_np = _to_numpy(y_true).astype(np.float32)
    y_proba_np = _to_numpy(y_pred_proba).astype(np.float32)
    y_bin_np = _to_numpy(y_pred_binary).astype(np.int64)

    if y_true_np.ndim != 2:
        raise ValueError(f"y_true must be 2D, got shape {y_true_np.shape}")
    if y_true_np.shape != y_proba_np.shape or y_true_np.shape != y_bin_np.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true_np.shape}, y_pred_proba={y_proba_np.shape}, "
            f"y_pred_binary={y_bin_np.shape}"
        )
    if len(label_names) != y_true_np.shape[1]:
        raise ValueError(
            f"len(label_names)={len(label_names)} does not match num_classes={y_true_np.shape[1]}"
        )

    per_cls: Dict[str, Dict[str, Any]] = {}
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true_np,
        y_bin_np,
        average=None,
        zero_division=0,
    )

    for i, label in enumerate(label_names):
        yt = y_true_np[:, i]
        yp = y_proba_np[:, i]
        has_pos = bool(np.any(yt == 1))
        has_neg = bool(np.any(yt == 0))

        if has_pos and has_neg:
            auroc = float(roc_auc_score(yt, yp))
            auroc_note = ""
        else:
            auroc = float("nan")
            auroc_note = "undefined: no positive or no negative samples in y_true"

        if has_pos:
            ap = float(average_precision_score(yt, yp))
        else:
            ap = float("nan")

        per_cls[label] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "auroc": auroc,
            "ap": ap,
            "auroc_note": auroc_note,
        }

    metrics: Dict[str, Any] = {
        "per_class": per_cls,
        "macro_f1": float(f1_score(y_true_np, y_bin_np, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true_np, y_bin_np, average="micro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true_np == y_bin_np, axis=1))),
        "hamming_loss": float(sk_hamming_loss(y_true_np, y_bin_np)),
    }
    return metrics


def find_optimal_thresholds(
    y_true: Any,
    y_pred_proba: Any,
    label_names: Sequence[str],
) -> Dict[str, float]:
    y_true_np = _to_numpy(y_true).astype(np.int64)
    y_proba_np = _to_numpy(y_pred_proba).astype(np.float32)
    if y_true_np.shape != y_proba_np.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_np.shape}, y_pred_proba={y_proba_np.shape}")
    if len(label_names) != y_true_np.shape[1]:
        raise ValueError(
            f"len(label_names)={len(label_names)} does not match num_classes={y_true_np.shape[1]}"
        )

    thresholds = np.arange(0.1, 0.9001, 0.05)
    best: Dict[str, float] = {}

    for i, name in enumerate(label_names):
        yt = y_true_np[:, i]
        yp = y_proba_np[:, i]
        best_thr = 0.5
        best_f1 = -1.0
        for thr in thresholds:
            yb = (yp >= thr).astype(np.int64)
            f1 = f1_score(yt, yb, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_thr = float(thr)
        best[name] = best_thr
    return best


def format_metrics_table(metrics_dict: Dict[str, Any], label_names: Sequence[str]) -> str:
    per_class = metrics_dict.get("per_class", {})
    header = f"{'Label':<28} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUROC':>9} {'AP':>9}"
    sep = "-" * len(header)
    lines = [header, sep]

    prec_vals = []
    rec_vals = []
    f1_vals = []
    auroc_vals = []
    ap_vals = []

    for name in label_names:
        row = per_class.get(name, {})
        p = float(row.get("precision", np.nan))
        r = float(row.get("recall", np.nan))
        f1 = float(row.get("f1", np.nan))
        au = float(row.get("auroc", np.nan))
        ap = float(row.get("ap", np.nan))

        prec_vals.append(p)
        rec_vals.append(r)
        f1_vals.append(f1)
        auroc_vals.append(au)
        ap_vals.append(ap)

        lines.append(f"{name:<28} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {au:>9.3f} {ap:>9.3f}")

    macro_prec = float(np.nanmean(np.asarray(prec_vals, dtype=float)))
    macro_rec = float(np.nanmean(np.asarray(rec_vals, dtype=float)))
    macro_f1 = float(metrics_dict.get("macro_f1", np.nan))
    macro_auroc = float(np.nanmean(np.asarray(auroc_vals, dtype=float)))
    macro_ap = float(np.nanmean(np.asarray(ap_vals, dtype=float)))

    lines.append(sep)
    lines.append(
        f"{'Macro avg':<28} {macro_prec:>7.3f} {macro_rec:>7.3f} {macro_f1:>7.3f} "
        f"{macro_auroc:>9.3f} {macro_ap:>9.3f}"
    )
    lines.append(
        f"{'Summary':<28} micro_f1={float(metrics_dict.get('micro_f1', np.nan)):.3f}  "
        f"subset_acc={float(metrics_dict.get('subset_accuracy', np.nan)):.3f}  "
        f"hamming_loss={float(metrics_dict.get('hamming_loss', np.nan)):.3f}"
    )
    return "\n".join(lines)
