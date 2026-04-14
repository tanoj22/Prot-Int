# `src/models/classifier.py` — documentation

This module defines a feed-forward classifier head for multilabel protein localization on top of precomputed embeddings.

---

## Purpose

- Consume fixed-length embedding vectors (for example from ESM-2).
- Produce raw logits per localization label.
- Keep inference helpers close to the model (`predict_proba`, `predict`).
- Keep label-name mapping inside the model object so outputs are human-readable.

---

## Label handling

`ProteinLocalizationClassifier` supports dynamic label configuration:

1. If `label_names` is passed explicitly, those names are used.
2. Else, it tries to load names from `label_columns.json` (default path points to the `esm2_t33_650M` embeddings folder).
3. If that file is missing/invalid, it falls back to built-in DeepLoc label names (including `Peroxisome`).

`num_labels` is optional:

- If `num_labels=None`, it is inferred as `len(label_names)`.
- If provided, it must match `len(label_names)`, otherwise `ValueError` is raised.

---

## Architecture

The model is an MLP with batch norm and dropout:

1. `Linear(embedding_dim -> 512)`  
2. `BatchNorm1d(512)`  
3. `ReLU`  
4. `Dropout(0.3)`  
5. `Linear(512 -> 256)`  
6. `BatchNorm1d(256)`  
7. `ReLU`  
8. `Dropout(0.3)`  
9. `Linear(256 -> 128)`  
10. `BatchNorm1d(128)`  
11. `ReLU`  
12. `Dropout(0.2)`  
13. `Linear(128 -> num_labels)`

### Initialization

All linear layers use He/Kaiming normal initialization (`kaiming_normal_` with ReLU nonlinearity), and biases are set to zero.

---

## Forward and inference methods

### `forward(x)`

- Input: tensor of shape `(B, embedding_dim)`.
- Output: raw logits `(B, num_labels)`.
- No sigmoid is applied; this is intended for `BCEWithLogitsLoss`.

### `predict_proba(embedding)`

- Accepts either a single vector `(embedding_dim,)` or a batch `(B, embedding_dim)`.
- Temporarily switches model to `eval()`, runs forward pass, applies sigmoid.
- Returns:
  - single input -> `dict[label_name -> probability]`
  - batch input -> `list[dict[label_name -> probability]]`

### `predict(embedding, thresholds=None)`

- Also accepts single or batch embeddings.
- Uses sigmoid probabilities and thresholds to produce binary outputs.
- `thresholds` options:
  - `None` -> all thresholds = `0.5`
  - `dict[label_name -> threshold]` -> missing labels default to `0.5`
  - tensor of length `num_labels`
- Returns:
  - single input -> `dict[label_name -> 0/1]`
  - batch input -> `list[dict[label_name -> 0/1]]`

---

## Utility functions

### `count_parameters(model)`

Prints:

- total parameter count
- trainable parameter count

### `load_model(path, embedding_dim, num_labels, device)`

Loads model weights from checkpoint and returns an eval-mode model on target device.

Supported checkpoint formats:

- `{"state_dict": ...}` (preferred)
- plain state-dict mapping

If `num_labels is None`, it infers label count from:

1. `label_names` in checkpoint (if present), else
2. final layer weight shape (`net.12.weight`).

---

## Notes for training/inference

- Because the model uses `BatchNorm1d`, batch size 1 in training mode can raise errors; use `eval()` for single-sample smoke/inference checks.
- Save `label_names` in checkpoints to keep label mapping portable and deterministic across runs.
