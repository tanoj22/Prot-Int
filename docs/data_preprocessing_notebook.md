# `notebooks/data_preprocessing.ipynb` — documentation

This note describes what the notebook does, cell by cell.

---

## Cell 0 — Load DeepLoc and inspect it

| Code | What it does |
|------|----------------|
| `import pandas as pd` | Uses pandas for tabular data. |
| `df = pd.read_csv("../data/raw/deeploc.csv")` | Reads the raw CSV from the project’s `data/raw` folder (path is relative to the notebook under `notebooks/`). |
| `print(df.shape)` | Prints row and column counts (example run: **28,303** rows × **16** columns). |
| `print(df.columns.tolist())` | Lists all column names: identifiers, metadata, label columns, and the sequence column. |
| `df.head()` | Shows the first five rows as a preview. |

**Takeaway:** The file includes columns such as `ACC`, `Kingdom`, `Partition`, eleven subcellular-location columns (numeric), `Sequence`, and an index-like `Unnamed: 0` column.

---

## Cell 1 — Shared constants for cleaning and modeling

| Code | What it does |
|------|----------------|
| `from pathlib import Path` | Used later for path handling when saving files (`save_processed_data`). |
| `VALID_AMINO_ACIDS` | Allowed one-letter codes: standard 20 amino acids plus common extensions (**X** unknown, **B** Asx, **Z** Glx, **U** selenocysteine, **O** pyrrolysine). Rows are kept only if every character in the sequence is in this set (after uppercasing). |
| `LABEL_COLUMNS` | The **11** DeepLoc subcellular compartments treated as **binary multilabel** targets. |
| `SEQUENCE_COLUMN = "Sequence"` | Name of the protein sequence column. |
| `ID_COLUMN = "ACC"` | Accession / protein ID used as the row identifier in the final table. |

**Purpose:** Single place to define valid amino acid alphabets and which columns are labels.

---

## Cell 2 — Cleaning pipeline and saving processed data

This cell defines helper functions and runs a full pipeline inside `if __name__ == "__main__":`. In Jupyter, `__name__` is usually `"__main__"`, so that block **does** run when you execute the cell.

### `is_valid_sequence(seq)`

- Requires a **string**; strips whitespace and uppercases.
- Rejects empty sequences.
- Returns **True** only if **every** character is in `VALID_AMINO_ACIDS`.

### `load_dataset(file_path)`

- Reads the CSV with `pd.read_csv`.
- Prints confirmation, shape, and column list; returns the dataframe.

### `clean_sequences(df)`

- Works on a **copy** of the dataframe.
- `dropna(subset=[SEQUENCE_COLUMN])` — removes rows with missing sequences.
- Converts `Sequence` to string, **strip + upper**.
- Keeps rows where `is_valid_sequence` is true.
- `drop_duplicates(subset=[SEQUENCE_COLUMN])` — drops duplicate sequences (first row kept).
- Prints row counts before and after (example: **28,303** unchanged if nothing was dropped).

### `clean_labels(df)`

- For each label column: `to_numeric(..., errors="coerce")`, `fillna(0)`, `astype(int)`, then **`clip(0, 1)`** so labels are binary **0/1**.
- Adds **`label_count`**: sum of the eleven labels per row (number of “positive” compartments).
- Prints the distribution of `label_count`.

### `build_final_dataframe(df)`

- Keeps **`ACC`**, **`Sequence`**, and the **11** label columns only (drops `Unnamed: 0`, `Kingdom`, `Partition`, etc.).
- Example shape: **28,303 × 13** (1 ID + 1 sequence + 11 labels).
- Prints shape and `head()` of the result.

### `save_processed_data(df, output_path)`

- Uses `Path(output_path)`; creates parent directories if missing (`mkdir(parents=True, exist_ok=True)`).
- Writes CSV with `to_csv(..., index=False)`.
- Prints the output path.

### `if __name__ == "__main__":` block

- Sets **`input_file`** and **`output_file`** (raw `deeploc.csv` → processed `deeploc_multilabel.csv`). Uses **raw strings** `r"..."` so Windows backslashes are not interpreted as escape sequences.
- Pipeline order: **load → clean sequences → clean labels → build final dataframe → save**.

**Net effect:** A cleaned **multilabel** CSV with protein ID, normalized sequence, and eleven binary compartment columns, saved under `data/processed/` (exact path as configured in the notebook).

---

## One-line summary

The notebook **loads** `deeploc.csv`, **sanitizes** sequences and **binarizes** subcellular labels, **drops** non-essential columns, and **writes** a compact multilabel file **`deeploc_multilabel.csv`**, after an exploratory load in cell 0.
