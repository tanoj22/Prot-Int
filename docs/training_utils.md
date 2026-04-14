# `src/training/losses.py` and `src/training/metrics.py` — documentation

This document explains the training utilities used for multilabel protein localization.

---

## `src/training/losses.py`

### Overview

Contains two loss classes and one factory:

- `WeightedBCEWithLogitsLoss`
- `FocalLoss`
- `get_loss_function(...)`

All losses expect **raw logits** (no external sigmoid).

---

### `WeightedBCEWithLogitsLoss`

Wrapper around `torch.nn.BCEWithLogitsLoss` with class-wise `pos_weight`.

- Constructor:
  - `pos_weight: Tensor` of shape `(num_labels,)`
- Forward:
  - `forward(logits, targets) -> Tensor`

Use when classes are imbalanced and you want to up-weight positive examples per class.

---

### `FocalLoss`

Multi-label focal loss variant operating on logits.

- Constructor:
  - `gamma` (default `2.0`)
  - `alpha` (default `0.25`)
  - `reduction` in `{"mean", "sum", "none"}`
- Forward:
  - internally computes `binary_cross_entropy_with_logits`
  - applies sigmoid to logits
  - applies focal weighting:
    - `pt = y*p + (1-y)*(1-p)`
    - `alpha_t = alpha*y + (1-alpha)*(1-y)`
    - `loss = alpha_t * (1-pt)^gamma * BCE`

Useful when you want to focus more on hard examples.

---

### `get_loss_function(name, pos_weights, device)`

Factory that returns a configured loss module.

- `name="bce"` -> `WeightedBCEWithLogitsLoss(pos_weights.to(device))`
- `name="focal"` -> `FocalLoss()`
- Unknown names raise `ValueError`.

For `"bce"`, `pos_weights` is required.

---

## `src/training/metrics.py`

### Overview

Provides three functions:

- `compute_metrics(...)`
- `find_optimal_thresholds(...)`
- `format_metrics_table(...)`

Inputs can be NumPy arrays or torch tensors; tensors are converted internally.

---

### `compute_metrics(y_true, y_pred_proba, y_pred_binary, label_names)`

Computes per-class and global multilabel metrics.

#### Inputs

- `y_true`: shape `(N, C)` ground-truth binary targets
- `y_pred_proba`: shape `(N, C)` probabilities
- `y_pred_binary`: shape `(N, C)` thresholded predictions
- `label_names`: length `C`

#### Returns dict

- `per_class[label_name]`:
  - `precision`, `recall`, `f1`, `auroc`, `ap`, `auroc_note`
- global:
  - `macro_f1`
  - `micro_f1`
  - `subset_accuracy` (exact-match ratio)
  - `hamming_loss`

#### Edge-case handling

- If a class has only one class present in `y_true` (no positives or no negatives), AUROC is undefined:
  - sets `auroc = NaN`
  - sets `auroc_note` to explain why.
- If a class has no positives, AP is set to `NaN`.

---

### `find_optimal_thresholds(y_true, y_pred_proba, label_names)`

Per-label threshold search:

- sweeps thresholds from `0.10` to `0.90` in steps of `0.05`
- computes F1 for each threshold
- picks threshold with max F1 per class

Returns:

- `dict[label_name -> best_threshold]`

This is useful for replacing global `0.5` thresholds with label-specific operating points.

---

### `format_metrics_table(metrics_dict, label_names)`

Builds a printable text table with:

- one row per class (`precision`, `recall`, `f1`, `auroc`, `ap`)
- bottom `Macro avg` row
- bottom `Summary` row with:
  - `micro_f1`
  - `subset_accuracy`
  - `hamming_loss`

Returns a single multi-line string for logs/console output.

---

## Typical usage pattern

1. Choose loss:
   - `get_loss_function("bce", pos_weights, device)` or
   - `get_loss_function("focal", ..., device)`
2. Train model on logits + targets.
3. At evaluation:
   - get probabilities via sigmoid
   - threshold at `0.5` (or optimized thresholds)
   - call `compute_metrics(...)`
   - optionally call `format_metrics_table(...)` for readable logging
4. Tune thresholds with `find_optimal_thresholds(...)` using validation data.
