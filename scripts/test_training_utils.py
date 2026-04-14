"""
Smoke test for src/training/losses.py and src/training/metrics.py.

Run from project root:
    .\\venv\\Scripts\\python.exe scripts\\test_training_utils.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.losses import FocalLoss, WeightedBCEWithLogitsLoss, get_loss_function  # noqa: E402
from src.training.metrics import (  # noqa: E402
    compute_metrics,
    find_optimal_thresholds,
    format_metrics_table,
)


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    n_samples = 64
    n_labels = 11
    label_names = [
        "Membrane",
        "Cytoplasm",
        "Nucleus",
        "Extracellular",
        "Cell membrane",
        "Mitochondrion",
        "Plastid",
        "Endoplasmic reticulum",
        "Lysosome/Vacuole",
        "Golgi apparatus",
        "Peroxisome",
    ]

    # Synthetic logits / multilabel targets
    logits = torch.randn(n_samples, n_labels, dtype=torch.float32)
    y_true = (torch.rand(n_samples, n_labels) > 0.7).float()
    pos_weights = torch.linspace(1.0, 3.0, n_labels)

    print("=== Losses smoke test ===")
    bce_loss_fn = WeightedBCEWithLogitsLoss(pos_weights)
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    factory_bce = get_loss_function("bce", pos_weights=pos_weights, device="cpu")
    factory_focal = get_loss_function("focal", pos_weights=pos_weights, device="cpu")

    bce_loss = bce_loss_fn(logits, y_true)
    focal_loss = focal_loss_fn(logits, y_true)
    bce_loss2 = factory_bce(logits, y_true)
    focal_loss2 = factory_focal(logits, y_true)

    print(f"Weighted BCE loss: {float(bce_loss):.6f}")
    print(f"Focal loss:        {float(focal_loss):.6f}")
    print(f"Factory BCE loss:  {float(bce_loss2):.6f}")
    print(f"Factory focal:     {float(focal_loss2):.6f}")

    # Metrics inputs
    y_proba = torch.sigmoid(logits).numpy()
    y_true_np = y_true.numpy().astype(np.int64)
    y_bin = (y_proba >= 0.5).astype(np.int64)

    print("\n=== Metrics smoke test ===")
    metrics = compute_metrics(
        y_true=y_true_np,
        y_pred_proba=y_proba,
        y_pred_binary=y_bin,
        label_names=label_names,
    )

    print(
        f"macro_f1={metrics['macro_f1']:.4f}, micro_f1={metrics['micro_f1']:.4f}, "
        f"subset_accuracy={metrics['subset_accuracy']:.4f}, hamming_loss={metrics['hamming_loss']:.4f}"
    )

    thresholds = find_optimal_thresholds(
        y_true=y_true_np,
        y_pred_proba=y_proba,
        label_names=label_names,
    )
    print(f"Found thresholds for {len(thresholds)} labels. Example: Membrane={thresholds['Membrane']:.2f}")

    table = format_metrics_table(metrics, label_names)
    print("\n" + table)

    print("\nOK - losses/metrics smoke test passed.")


if __name__ == "__main__":
    main()
