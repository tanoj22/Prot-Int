# `src/data/dataset.py` — documentation

This module wires **precomputed protein embeddings** and **multilabel localization targets** into PyTorch so you can train heads (e.g. linear + `BCEWithLogitsLoss`) without recomputing ESM embeddings inside the training loop.

---

## Data flow (high level)

1. **`embeddings.py`** writes a model-specific folder (e.g. `data/processed/embeddings/esm2_t12_35M/`) containing NumPy arrays and JSON metadata.
2. **`ProteinLocalizationDataset`** loads those files once, keeps tensors in memory, and exposes standard `Dataset` indexing.
3. **`create_splits`** partitions indices into train / val / test while trying to preserve label balance per split.
4. **`create_dataloaders`** wraps splits in `DataLoader` instances with a custom collate function for batched tensors and string accessions.
5. **`compute_class_weights`** derives per-label `pos_weight` vectors for imbalanced multilabel classification.

All filesystem paths use **`pathlib.Path`** so the same code runs on Windows, macOS, and Linux.

---

## Expected files per embeddings directory

| File | Role |
|------|------|
| `embeddings.npy` | 2D float array, shape `(N, embedding_dim)` — one row per protein |
| `multilabel_targets.npy` | 2D float array, shape `(N, num_labels)` — binary (or soft) multilabel targets |
| `accessions.npy` | 1D object array of accession strings, length `N`, same row order as above |
| `label_columns.json` | JSON with key `"label_columns"`: list of label names, length `num_labels`, column order matching `multilabel_targets.npy` |

On init, the dataset checks that row counts align and that the target matrix width matches `label_columns.json`.

---

## `ProteinLocalizationDataset`

### Purpose

Implements `torch.utils.data.Dataset`. It does **not** read raw sequences or run transformers; it only serves tensors already produced offline.

### `__init__(embeddings_dir)`

- Resolves `embeddings_dir` with `expanduser().resolve()`.
- Loads the four files above; `accessions.npy` uses `allow_pickle=True` because accession strings are often stored as a NumPy object array.
- Converts embeddings and targets to **`float32`** PyTorch tensors (suitable for mixed-precision training downstream).
- Stores accessions as the raw NumPy array; `__getitem__` normalizes a scalar to a Python `str`.

### `__getitem__(idx)` → `(embedding, target, accession)`

Returns a single sample:

- **embedding**: `torch.Tensor` of shape `(embedding_dim,)`
- **target**: `torch.Tensor` of shape `(num_labels,)`
- **accession**: Python `str` (protein ID)

### Properties

| Property | Meaning |
|----------|--------|
| `label_names` | Copy of the label names list from `label_columns.json` |
| `embedding_dim` | Second dimension of the embedding matrix |
| `num_labels` | Second dimension of the target matrix (number of multilabel columns) |
| `__len__` | Number of samples `N` |

The number of labels is **data-driven** (whatever your preprocessing and embeddings pipeline wrote), e.g. DeepLoc-style tables often use **10 or 11** compartment columns.

---

## `_collate_localization_batch`

PyTorch’s default collate does not batch arbitrary Python strings cleanly. This helper:

- **Stacks** embeddings and targets into shape `(batch, …)`.
- Collects accessions into a **Python list of strings** (one per row in the batch).

So each `DataLoader` batch is `(emb_batch, tgt_batch, acc_list)`.

---

## Train / validation / test splitting

### `create_splits(dataset, train_ratio, val_ratio, test_ratio, random_seed)`

- Requires `train_ratio + val_ratio + test_ratio ≈ 1.0`.
- Reads the full multilabel matrix `y` from the dataset (all `N` rows).

**Preferred path — `iterstrat`:**

- Uses `importlib.import_module("iterstrat.ml_stratifiers")` and `MultilabelStratifiedShuffleSplit` (avoids static import resolution issues in some IDEs).
- **Two-stage split:**
  1. Split off **train** vs **temp** where `temp` has fraction `val_ratio + test_ratio` of the data, stratified on `y`.
  2. Split **temp** into **val** and **test** with a second stratified split; the inner `test_size` is `test_ratio / (val_ratio + test_ratio)` so val and test each get the intended fraction of the full dataset (up to rounding).

**Fallback — random:**

- If `iterstrat` is not installed (`ImportError`), emits a **`UserWarning`** and uses **`_split_indices_random`**: a single permutation of `0..N-1`, then contiguous slices for train / val / test lengths from the rounded ratios. This preserves approximate sizes but **does not** stratify labels.

After splitting, the function prints **per-split label statistics** (positive count and percentage of that split per label) via `_print_split_label_distribution`.

**Return value:** three `torch.utils.data.Subset` objects sharing the same underlying `ProteinLocalizationDataset`, indexed by the chosen row indices.

---

## `create_dataloaders(train, val, test, batch_size, num_workers)`

Builds a dictionary:

```text
{"train": DataLoader, "val": DataLoader, "test": DataLoader}
```

- **Train**: `shuffle=True`.
- **Val / test**: `shuffle=False`.
- All use the custom **`collate_fn`** described above.

`num_workers=0` is the default (simple debugging on Windows; increase for faster loading if needed).

---

## `compute_class_weights(train_dataset)`

Used for **`torch.nn.BCEWithLogitsLoss(..., pos_weight=...)`**:

- Accepts either a full **`ProteinLocalizationDataset`** or a **`Subset`** (e.g. training split only).
- For each label \(k\):  
  \(\text{pos\_weight}_k = \frac{\#\text{negatives}}{\#\text{positives}}\)  
  with a small floor on positives to avoid division by zero.
- Returns a **`torch.float32`** tensor of shape `(num_labels,)`.
- Prints each weight and marks **extreme** values (`> 50` or `< 0.1`) for inspection.

Label names for printing come from the base dataset’s `label_names` property.

---

## Dependencies

- **Required:** `numpy`, `torch`.
- **Optional (recommended):** `iterstrat` — multilabel stratified splits. Listed in `requirements.txt`; install in a Python version the package supports (some very new Python releases may not have wheels yet).

---

## Related scripts

- **`scripts/test_dataset.py`** — smoke test: load a folder, run splits, one train batch, class weights.

---

## Design notes

- **No I/O in `__getitem__`:** all arrays are loaded in `__init__`, so training loops are not disk-bound per epoch (at the cost of RAM for large `N`).
- **Subset-aware helpers:** splitting only changes which indices are used; embedding rows and targets stay aligned by construction.
