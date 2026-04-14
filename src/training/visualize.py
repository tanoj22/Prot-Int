"""
Visualization utilities for multilabel protein localization training and evaluation.

All plot functions save PNG files to ``output_dir`` (150 DPI) and return matplotlib Figure objects.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, f1_score

# Clean default style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _subplot_grid(n: int, max_cols: int = 5) -> Tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be positive")
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_dir: str | Path,
    best_epoch: Optional[int] = None,
    filename: str = "training_curves.png",
) -> Figure:
    """Plot train vs validation loss per epoch; vertical line at best epoch (1-based, min val loss if omitted)."""
    out = _ensure_output_dir(output_dir)
    train_losses = list(train_losses)
    val_losses = list(val_losses)
    if len(train_losses) != len(val_losses):
        raise ValueError("train_losses and val_losses must have the same length")
    if best_epoch is None and len(val_losses) > 0:
        best_epoch = int(np.argmin(np.asarray(val_losses, dtype=np.float64))) + 1
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train loss", color="#1f77b4", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation loss", color="#ff7f0e", linewidth=2)
    if best_epoch is not None and 1 <= best_epoch <= len(train_losses):
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.5, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_f1_progression(
    epoch_metrics_list: Sequence[Mapping[str, float]],
    label_names: Sequence[str],
    output_dir: str | Path,
    filename: str = "per_class_f1_progression.png",
) -> Figure:
    """
    ``epoch_metrics_list``: one dict per epoch, each mapping ``label_name -> f1`` for validation.
    """
    out = _ensure_output_dir(output_dir)
    label_names = list(label_names)
    n_epochs = len(epoch_metrics_list)
    if n_epochs == 0:
        raise ValueError("epoch_metrics_list is empty")
    epochs = np.arange(1, n_epochs + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("husl", n_colors=max(len(label_names), 1))
    for i, name in enumerate(label_names):
        series = [float(epoch_metrics_list[e].get(name, np.nan)) for e in range(n_epochs)]
        ax.plot(epochs, series, label=name, color=palette[i % len(palette)], linewidth=1.8)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("Per-Class F1 Score Across Training", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def _auroc_color(val: float) -> str:
    if np.isnan(val):
        return "#cccccc"
    if val < 0.7:
        return "#d62728"
    if val <= 0.85:
        return "#ffbf00"
    return "#2ca02c"


def plot_auroc_bars(
    test_metrics: Mapping[str, Any],
    label_names: Sequence[str],
    output_dir: str | Path,
    filename: str = "test_auroc_bars.png",
) -> Figure:
    """Horizontal bar chart of per-class AUROC (from ``test_metrics['per_class'][name]['auroc']``)."""
    out = _ensure_output_dir(output_dir)
    per_class = test_metrics.get("per_class", {})
    rows: List[Tuple[str, float]] = []
    for name in label_names:
        d = per_class.get(name, {})
        a = float(d.get("auroc", np.nan))
        rows.append((name, a))
    rows.sort(
        key=lambda x: (0 if not np.isnan(x[1]) else 1, -(x[1] if not np.isnan(x[1]) else 0.0), x[0]),
    )
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(labels))))
    colors = [_auroc_color(v) for v in values]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title("Test Set AUROC by Subcellular Location", fontsize=13, fontweight="bold")
    for bar, v in zip(bars, values):
        if np.isnan(v):
            ax.text(0.02, bar.get_y() + bar.get_height() / 2, "nan", va="center", fontsize=9)
        else:
            ax.text(min(v + 0.02, 1.0), bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(
    y_true: Any,
    y_pred: Any,
    label_names: Sequence[str],
    output_dir: str | Path,
    filename: str = "confusion_matrices.png",
) -> Figure:
    """One 2×2 confusion matrix per class (binary multilabel)."""
    out = _ensure_output_dir(output_dir)
    yt = _to_numpy(y_true).astype(np.int64)
    yp = _to_numpy(y_pred).astype(np.int64)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays of the same shape")
    n = len(label_names)
    nrows, ncols = _subplot_grid(n, max_cols=5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, name in enumerate(label_names):
        ax = axes_flat[i]
        cm = confusion_matrix(yt[:, i], yp[:, i], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(name, fontsize=10, fontweight="bold")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def _cooccurrence_matrix(y_bin: np.ndarray) -> np.ndarray:
    """Symmetric co-occurrence counts: (y.T @ y) for binary matrix y (n_samples, n_labels)."""
    return (y_bin.T @ y_bin).astype(np.float64)


def plot_label_cooccurrence(
    y_true: Any,
    y_pred: Any,
    label_names: Sequence[str],
    output_dir: str | Path,
    filename: str = "label_cooccurrence.png",
) -> Figure:
    """Side-by-side heatmaps: ground-truth vs predicted label co-occurrence."""
    out = _ensure_output_dir(output_dir)
    yt = _to_numpy(y_true).astype(np.int64)
    yp = _to_numpy(y_pred).astype(np.int64)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    c_true = _cooccurrence_matrix(yt)
    c_pred = _cooccurrence_matrix(yp)
    vmax = max(c_true.max(), c_pred.max(), 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(
        c_true,
        xticklabels=label_names,
        yticklabels=label_names,
        annot=False,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax1,
        vmin=0,
        vmax=vmax,
        square=True,
    )
    ax1.set_title("Ground truth co-occurrence", fontsize=12, fontweight="bold")
    sns.heatmap(
        c_pred,
        xticklabels=label_names,
        yticklabels=label_names,
        annot=False,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax2,
        vmin=0,
        vmax=vmax,
        square=True,
    )
    ax2.set_title("Predicted co-occurrence", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_probability_distributions(
    y_true: Any,
    y_pred_proba: Any,
    label_names: Sequence[str],
    output_dir: str | Path,
    filename: str = "probability_distributions.png",
) -> Figure:
    """Per-class histograms: predicted probability for negatives (red) vs positives (blue)."""
    out = _ensure_output_dir(output_dir)
    yt = _to_numpy(y_true).astype(np.float64)
    yp = _to_numpy(y_pred_proba).astype(np.float64)
    n = len(label_names)
    nrows, ncols = _subplot_grid(n, max_cols=5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, name in enumerate(label_names):
        ax = axes_flat[i]
        pos_mask = yt[:, i] > 0.5
        neg_mask = ~pos_mask
        pos_p = yp[pos_mask, i]
        neg_p = yp[neg_mask, i]
        ax.hist(neg_p, bins=20, alpha=0.65, color="#d62728", label="Negative", density=True)
        ax.hist(pos_p, bins=20, alpha=0.65, color="#1f77b4", label="Positive", density=True)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_threshold_analysis(
    y_true: Any,
    y_pred_proba: Any,
    label_names: Sequence[str],
    output_dir: str | Path,
    filename: str = "threshold_analysis.png",
    thresholds: np.ndarray | None = None,
) -> Figure:
    """Per-class F1 vs threshold; marks optimal threshold (max F1) with a red dot."""
    out = _ensure_output_dir(output_dir)
    yt = _to_numpy(y_true).astype(np.int64)
    yp = _to_numpy(y_pred_proba).astype(np.float64)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(label_names)
    nrows, ncols = _subplot_grid(n, max_cols=5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, name in enumerate(label_names):
        ax = axes_flat[i]
        f1s = []
        for thr in thresholds:
            yb = (yp[:, i] >= thr).astype(np.int64)
            f1s.append(f1_score(yt[:, i], yb, zero_division=0))
        f1s = np.asarray(f1s)
        best_idx = int(np.argmax(f1s))
        best_thr = float(thresholds[best_idx])
        best_f1 = float(f1s[best_idx])
        ax.plot(thresholds, f1s, color="#1f77b4", linewidth=1.5)
        ax.scatter([best_thr], [best_f1], color="red", s=40, zorder=5)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=8)
        ax.set_ylabel("F1", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.35)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def generate_all_plots(
    *,
    output_dir: str | Path,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    epoch_metrics_list: Sequence[Mapping[str, float]],
    label_names: Sequence[str],
    test_metrics: Mapping[str, Any],
    y_true: Any,
    y_pred: Any,
    y_pred_proba: Any,
    best_epoch: Optional[int] = None,
    mlflow_log: bool = True,
) -> Dict[str, Figure]:
    """
    Generate all analysis plots and optionally log each PNG to the active MLflow run.

    ``epoch_metrics_list``: one dict per epoch, each mapping ``label_name -> validation F1``.
    """
    out = _ensure_output_dir(output_dir)
    figures: Dict[str, Figure] = {}

    figures["training_curves"] = plot_training_curves(
        train_losses, val_losses, out, best_epoch=best_epoch
    )
    figures["per_class_f1"] = plot_per_class_f1_progression(epoch_metrics_list, label_names, out)
    figures["auroc_bars"] = plot_auroc_bars(test_metrics, label_names, out)
    figures["confusion"] = plot_confusion_matrices(y_true, y_pred, label_names, out)
    figures["cooccurrence"] = plot_label_cooccurrence(y_true, y_pred, label_names, out)
    figures["proba_hist"] = plot_probability_distributions(y_true, y_pred_proba, label_names, out)
    figures["threshold"] = plot_threshold_analysis(y_true, y_pred_proba, label_names, out)

    if mlflow_log and mlflow is not None:
        try:
            if mlflow.active_run() is not None:
                artifact_files = [
                    "training_curves.png",
                    "per_class_f1_progression.png",
                    "test_auroc_bars.png",
                    "confusion_matrices.png",
                    "label_cooccurrence.png",
                    "probability_distributions.png",
                    "threshold_analysis.png",
                ]
                for fname in artifact_files:
                    p = out / fname
                    if p.is_file():
                        mlflow.log_artifact(str(p), artifact_path="plots")
        except Exception:
            pass

    return figures


def plot_training_curves_comparison(
    train_losses_a: Sequence[float],
    val_losses_a: Sequence[float],
    train_losses_b: Sequence[float],
    val_losses_b: Sequence[float],
    label_a: str,
    label_b: str,
    output_dir: str | Path,
    best_epoch_a: Optional[int] = None,
    best_epoch_b: Optional[int] = None,
    filename: str = "compare_training_val_loss.png",
) -> Figure:
    """Overlay train/val loss curves for two runs."""
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(11, 5))
    ea = np.arange(1, len(train_losses_a) + 1)
    eb = np.arange(1, len(train_losses_b) + 1)
    ax.plot(ea, train_losses_a, label=f"Train ({label_a})", color="#1f77b4", linewidth=2, linestyle="-")
    ax.plot(ea, val_losses_a, label=f"Val ({label_a})", color="#aec7e8", linewidth=2, linestyle="-")
    ax.plot(eb, train_losses_b, label=f"Train ({label_b})", color="#d62728", linewidth=2, linestyle="--")
    ax.plot(eb, val_losses_b, label=f"Val ({label_b})", color="#ff9896", linewidth=2, linestyle="--")
    if best_epoch_a is not None and 1 <= best_epoch_a <= len(val_losses_a):
        ax.axvline(best_epoch_a, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8)
    if best_epoch_b is not None and 1 <= best_epoch_b <= len(val_losses_b):
        ax.axvline(best_epoch_b, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training & Validation Loss — comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_val_macro_f1_comparison(
    macro_f1_a: Sequence[float],
    macro_f1_b: Sequence[float],
    label_a: str,
    label_b: str,
    output_dir: str | Path,
    filename: str = "compare_val_macro_f1.png",
) -> Figure:
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1, len(macro_f1_a) + 1), macro_f1_a, label=label_a, color="#1f77b4", linewidth=2)
    ax.plot(np.arange(1, len(macro_f1_b) + 1), macro_f1_b, label=label_b, color="#d62728", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation macro F1", fontsize=11)
    ax.set_title("Validation macro F1 — comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_f1_panels(
    epoch_metrics_a: Sequence[Mapping[str, float]],
    epoch_metrics_b: Sequence[Mapping[str, float]],
    label_names: Sequence[str],
    title_a: str,
    title_b: str,
    output_dir: str | Path,
    filename: str = "compare_per_class_f1_panels.png",
) -> Figure:
    """Side-by-side panels: per-class F1 vs epoch for each run."""
    out = _ensure_output_dir(output_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    palette = sns.color_palette("husl", n_colors=max(len(label_names), 1))
    n_a, n_b = len(epoch_metrics_a), len(epoch_metrics_b)
    for i, name in enumerate(label_names):
        sa = [float(epoch_metrics_a[e].get(name, np.nan)) for e in range(n_a)]
        sb = [float(epoch_metrics_b[e].get(name, np.nan)) for e in range(n_b)]
        ax1.plot(np.arange(1, n_a + 1), sa, label=name, color=palette[i % len(palette)], linewidth=1.5)
        ax2.plot(np.arange(1, n_b + 1), sb, label=name, color=palette[i % len(palette)], linewidth=1.5)
    ax1.set_title(title_a, fontsize=12, fontweight="bold")
    ax2.set_title(title_b, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax2.set_xlabel("Epoch")
    ax1.set_ylabel("F1")
    ax1.grid(True, alpha=0.35)
    ax2.grid(True, alpha=0.35)
    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, title="Class")
    fig.suptitle("Per-class validation F1 — comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_auroc_comparison_bars(
    test_metrics_a: Mapping[str, Any],
    test_metrics_b: Mapping[str, Any],
    label_names: Sequence[str],
    label_a: str,
    label_b: str,
    output_dir: str | Path,
    filename: str = "compare_test_auroc.png",
) -> Figure:
    """Grouped horizontal bars: AUROC per class for two runs."""
    out = _ensure_output_dir(output_dir)
    names = list(label_names)
    y = np.arange(len(names))
    h = 0.35
    va = [float(test_metrics_a.get("per_class", {}).get(n, {}).get("auroc", np.nan)) for n in names]
    vb = [float(test_metrics_b.get("per_class", {}).get(n, {}).get("auroc", np.nan)) for n in names]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(names))))
    ax.barh(y - h / 2, va, h, label=label_a, color="#1f77b4", alpha=0.85)
    ax.barh(y + h / 2, vb, h, label=label_b, color="#d62728", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title("Test AUROC by class — comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices_two_runs(
    y_true_a: Any,
    y_pred_a: Any,
    y_true_b: Any,
    y_pred_b: Any,
    label_names: Sequence[str],
    row_title_a: str,
    row_title_b: str,
    output_dir: str | Path,
    filename: str = "compare_confusion_matrices.png",
) -> Figure:
    """Two rows of per-class confusion matrices (binary multilabel)."""
    out = _ensure_output_dir(output_dir)
    yt_a = _to_numpy(y_true_a).astype(np.int64)
    yp_a = _to_numpy(y_pred_a).astype(np.int64)
    yt_b = _to_numpy(y_true_b).astype(np.int64)
    yp_b = _to_numpy(y_pred_b).astype(np.int64)
    n = len(label_names)
    nrows, ncols = _subplot_grid(n, max_cols=5)
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(3.2 * ncols, 3.0 * 2 * nrows))
    axes_arr = np.atleast_2d(axes)
    for i, name in enumerate(label_names):
        r0, c0 = divmod(i, ncols)
        for ver, (yt, yp, rtitle) in enumerate(
            [(yt_a, yp_a, row_title_a), (yt_b, yp_b, row_title_b)]
        ):
            ax = axes_arr[ver * nrows + r0, c0]
            cm = confusion_matrix(yt[:, i], yp[:, i], labels=[0, 1])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["P0", "P1"],
                yticklabels=["T0", "T1"],
            )
            ax.set_title(f"{name}\n{rtitle}", fontsize=8, fontweight="bold")
    used: Set[Tuple[int, int]] = set()
    for i in range(n):
        r0, c0 = divmod(i, ncols)
        for ver in (0, 1):
            used.add((ver * nrows + r0, c0))
    for r in range(2 * nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes_arr[r, c].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_label_cooccurrence_four(
    y_true_a: Any,
    y_pred_a: Any,
    y_true_b: Any,
    y_pred_b: Any,
    label_names: Sequence[str],
    titles: Sequence[str],
    output_dir: str | Path,
    filename: str = "compare_label_cooccurrence.png",
) -> Figure:
    """Four heatmaps: GT/pred co-occurrence for each run."""
    out = _ensure_output_dir(output_dir)
    mats = []
    for yt, yp in [(y_true_a, y_pred_a), (y_true_b, y_pred_b)]:
        yt = _to_numpy(yt).astype(np.int64)
        yp = _to_numpy(yp).astype(np.int64)
        mats.append((_cooccurrence_matrix(yt), _cooccurrence_matrix(yp)))
    vmax = 1.0
    for gt_mat, pred_mat in mats:
        vmax = max(vmax, float(gt_mat.max()), float(pred_mat.max()))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    flat_titles = [titles[0], titles[1], titles[2], titles[3]]
    idx = 0
    for row, pair in enumerate(mats):
        for col, cmat in enumerate(pair):
            ax = axes[row, col]
            sns.heatmap(
                cmat,
                xticklabels=label_names,
                yticklabels=label_names,
                cmap="YlOrRd",
                ax=ax,
                vmin=0,
                vmax=vmax,
                square=True,
            )
            ax.set_title(flat_titles[idx], fontsize=11, fontweight="bold")
            idx += 1
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_probability_distributions_two_runs(
    y_true_a: Any,
    y_proba_a: Any,
    y_true_b: Any,
    y_proba_b: Any,
    label_names: Sequence[str],
    title_a: str,
    title_b: str,
    output_dir: str | Path,
    filename: str = "compare_probability_distributions.png",
) -> Figure:
    """Two rows of per-class probability histograms (neg vs pos)."""
    out = _ensure_output_dir(output_dir)
    n = len(label_names)
    nrows, ncols = _subplot_grid(n, max_cols=5)
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(3.2 * ncols, 2.6 * 2 * nrows))
    axes_arr = np.atleast_2d(axes)

    def _row(yt: Any, yp: Any, ver: int) -> None:
        yt = _to_numpy(yt).astype(np.float64)
        yp = _to_numpy(yp).astype(np.float64)
        for i, name in enumerate(label_names):
            r0, c0 = divmod(i, ncols)
            ax = axes_arr[ver * nrows + r0, c0]
            pos_mask = yt[:, i] > 0.5
            neg_mask = ~pos_mask
            ax.hist(yp[neg_mask, i], bins=20, alpha=0.6, color="#d62728", density=True, label="Neg")
            ax.hist(yp[pos_mask, i], bins=20, alpha=0.6, color="#1f77b4", density=True, label="Pos")
            ax.set_title(f"{name}", fontsize=8, fontweight="bold")
            ax.set_xlim(0, 1)

    _row(y_true_a, y_proba_a, 0)
    _row(y_true_b, y_proba_b, 1)
    for j in range(n, nrows * ncols):
        r0, c0 = divmod(j, ncols)
        for ver in range(2):
            axes_arr[ver * nrows + r0, c0].set_visible(False)
    fig.suptitle(f"Predicted probability distributions — {title_a} (top) vs {title_b} (bottom)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_threshold_analysis_two_runs(
    y_true_a: Any,
    y_proba_a: Any,
    y_true_b: Any,
    y_proba_b: Any,
    label_names: Sequence[str],
    title_a: str,
    title_b: str,
    output_dir: str | Path,
    filename: str = "compare_threshold_analysis.png",
) -> Figure:
    """Two rows: F1 vs threshold per class."""
    out = _ensure_output_dir(output_dir)
    thresholds = np.linspace(0.01, 0.99, 99)
    n = len(label_names)
    nrows, ncols = _subplot_grid(n, max_cols=5)
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(3.2 * ncols, 2.4 * 2 * nrows))
    axes_arr = np.atleast_2d(axes)

    def _fill(yt: Any, yp: Any, ver: int) -> None:
        yt = _to_numpy(yt).astype(np.int64)
        yp = _to_numpy(yp).astype(np.float64)
        for i, name in enumerate(label_names):
            r0, c0 = divmod(i, ncols)
            ax = axes_arr[ver * nrows + r0, c0]
            f1s = [f1_score(yt[:, i], (yp[:, i] >= thr).astype(np.int64), zero_division=0) for thr in thresholds]
            ax.plot(thresholds, f1s, color="#1f77b4", linewidth=1.2)
            bi = int(np.argmax(f1s))
            ax.scatter([thresholds[bi]], [f1s[bi]], color="red", s=25, zorder=5)
            ax.set_title(f"{name}", fontsize=7, fontweight="bold")
            ax.set_xlabel("Thr", fontsize=7)
            ax.set_ylabel("F1", fontsize=7)

    _fill(y_true_a, y_proba_a, 0)
    _fill(y_true_b, y_proba_b, 1)
    for j in range(n, nrows * ncols):
        r0, c0 = divmod(j, ncols)
        for ver in range(2):
            axes_arr[ver * nrows + r0, c0].set_visible(False)
    fig.suptitle(f"F1 vs threshold — {title_a} (top) vs {title_b} (bottom)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    return fig


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_comparison_plots(
    *,
    v1_artifact_dir: Union[str, Path],
    v2_artifact_dir: Union[str, Path],
    label_names: Sequence[str],
    output_dir: Union[str, Path],
    label_v1: str = "baseline_v1",
    label_v2: str = "regularized_v2",
    mlflow_log: bool = True,
) -> Dict[str, Figure]:
    """
    Load two run artifact folders (``train_history.json``, ``final_test_metrics.json``,
    ``test_predictions.npz``) and write comparison figures.
    """
    d1 = Path(v1_artifact_dir).expanduser().resolve()
    d2 = Path(v2_artifact_dir).expanduser().resolve()
    out = _ensure_output_dir(output_dir)

    h1 = _load_json(d1 / "train_history.json")
    h2 = _load_json(d2 / "train_history.json")
    m1 = _load_json(d1 / "final_test_metrics.json")
    m2 = _load_json(d2 / "final_test_metrics.json")
    z1 = np.load(d1 / "test_predictions.npz")
    z2 = np.load(d2 / "test_predictions.npz")

    figures: Dict[str, Figure] = {}
    be1 = h1.get("best_epoch")
    be2 = h2.get("best_epoch")
    be1i = int(be1) if be1 is not None and int(be1) >= 1 else None
    be2i = int(be2) if be2 is not None and int(be2) >= 1 else None

    figures["compare_loss"] = plot_training_curves_comparison(
        h1.get("train_loss", []),
        h1.get("val_loss", []),
        h2.get("train_loss", []),
        h2.get("val_loss", []),
        label_v1,
        label_v2,
        out,
        best_epoch_a=be1i,
        best_epoch_b=be2i,
    )
    figures["compare_macro_f1"] = plot_val_macro_f1_comparison(
        h1.get("val_macro_f1", []),
        h2.get("val_macro_f1", []),
        label_v1,
        label_v2,
        out,
    )
    figures["compare_per_class_f1"] = plot_per_class_f1_panels(
        h1.get("val_per_class_f1", []),
        h2.get("val_per_class_f1", []),
        label_names,
        label_v1,
        label_v2,
        out,
    )
    figures["compare_auroc"] = plot_auroc_comparison_bars(m1, m2, label_names, label_v1, label_v2, out)
    figures["compare_confusion"] = plot_confusion_matrices_two_runs(
        z1["y_true"],
        z1["y_pred_binary"],
        z2["y_true"],
        z2["y_pred_binary"],
        label_names,
        label_v1,
        label_v2,
        out,
    )
    figures["compare_cooc"] = plot_label_cooccurrence_four(
        z1["y_true"],
        z1["y_pred_binary"],
        z2["y_true"],
        z2["y_pred_binary"],
        label_names,
        (f"{label_v1} GT", f"{label_v1} pred", f"{label_v2} GT", f"{label_v2} pred"),
        out,
    )
    figures["compare_proba"] = plot_probability_distributions_two_runs(
        z1["y_true"],
        z1["y_pred_proba"],
        z2["y_true"],
        z2["y_pred_proba"],
        label_names,
        label_v1,
        label_v2,
        out,
    )
    figures["compare_thr"] = plot_threshold_analysis_two_runs(
        z1["y_true"],
        z1["y_pred_proba"],
        z2["y_true"],
        z2["y_pred_proba"],
        label_names,
        label_v1,
        label_v2,
        out,
    )

    if mlflow_log and mlflow is not None:
        try:
            if mlflow.active_run() is not None:
                for fname in (
                    "compare_training_val_loss.png",
                    "compare_val_macro_f1.png",
                    "compare_per_class_f1_panels.png",
                    "compare_test_auroc.png",
                    "compare_confusion_matrices.png",
                    "compare_label_cooccurrence.png",
                    "compare_probability_distributions.png",
                    "compare_threshold_analysis.png",
                ):
                    p = out / fname
                    if p.is_file():
                        mlflow.log_artifact(str(p), artifact_path="comparison_plots")
        except Exception:
            pass

    return figures
