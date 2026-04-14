"""
Compute per-sequence protein embeddings from the preprocessed DeepLoc CSV.

Run from project root:
    python src/embeddings.py --help
    python src/embeddings.py --model_name facebook/esm2_t12_35M_UR50D --debug

Optional environment variables:
    PROTEIN_EMBEDDING_BATCH  Overrides --batch_size if set (integer).

Outputs go to data/processed/embeddings/{model_short_name}/ by default so legacy
data/processed/embeddings/ (flat) is left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_INPUT_REL = Path("data/processed/deeploc_multilabel.csv")
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 1024
DEBUG_SAMPLE_SIZE = 100

ID_COLUMN = "ACC"
SEQUENCE_COLUMN = "Sequence"

# DeepLoc multilabel targets — 11 compartments (matches notebooks/data_preprocessing LABEL_COLUMNS).
# Canonical order for multilabel_targets.npy columns.
EXPECTED_LABEL_COLUMNS: Tuple[str, ...] = (
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
)

_EXPECTED_LABEL_SET = frozenset(EXPECTED_LABEL_COLUMNS)


def model_short_name_from_hf_id(model_id: str) -> str:
    """e.g. facebook/esm2_t33_650M_UR50D -> esm2_t33_650M"""
    name = model_id.rstrip("/").split("/")[-1]
    if name.endswith("_UR50D"):
        name = name[: -len("_UR50D")]
    safe = re.sub(r"[^\w.\-]+", "_", name)
    return safe or "model"


def resolve_input_csv(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def resolve_output_dir(model_name: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (ROOT / p).resolve()
    short = model_short_name_from_hf_id(model_name)
    return ROOT / "data" / "processed" / "embeddings" / short


def _is_trailing_csv_ghost_column(df: pd.DataFrame, col: str) -> bool:
    """True for Unnamed / blank-header columns that are all missing (typical trailing-comma artifacts)."""
    if str(col).strip() == "":
        return True
    if not str(col).startswith("Unnamed"):
        return False
    series = df[col]
    if series.isna().all():
        return True
    num = pd.to_numeric(series, errors="coerce")
    if num.isna().all():
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        s = series.astype(str).str.strip()
        blank = s.isna() | (s == "") | (s.str.lower() == "nan")
        if blank.all():
            return True
    return False


def strip_dataframe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every column name (in place on a copy)."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def resolve_label_columns(df: pd.DataFrame) -> List[str]:
    """
    Collect label columns (non-id, non-sequence), drop ghosts/blank names,
    then require that label columns match EXPECTED_LABEL_COLUMNS exactly (set and count).
    Returns columns in canonical order.
    """
    excluded: List[str] = []
    labels: List[str] = []
    for c in df.columns:
        if c in (ID_COLUMN, SEQUENCE_COLUMN):
            continue
        if c == "" or not str(c).strip():
            excluded.append(repr(c))
            continue
        if _is_trailing_csv_ghost_column(df, c):
            excluded.append(str(c))
            continue
        labels.append(c)
    if excluded:
        print(
            f"Warning: excluded {len(excluded)} non-label column(s) "
            f"(blank/Unnamed/empty artifacts): {excluded}"
        )

    found_set = set(labels)
    if len(labels) != len(found_set):
        raise ValueError(
            f"Duplicate label column names after stripping headers: {labels!r}"
        )

    if found_set != _EXPECTED_LABEL_SET or len(labels) != len(EXPECTED_LABEL_COLUMNS):
        extra = sorted(found_set - _EXPECTED_LABEL_SET)
        missing = sorted(_EXPECTED_LABEL_SET - found_set)
        raise ValueError(
            f"Expected exactly {len(EXPECTED_LABEL_COLUMNS)} label columns "
            f"{list(EXPECTED_LABEL_COLUMNS)!r}. "
            f"Found {len(labels)} column(s): {labels!r}. "
            f"Extra (not in expected set): {extra!r}. "
            f"Missing (not in CSV): {missing!r}. "
            f"All stripped columns: {list(df.columns)!r}"
        )

    return list(EXPECTED_LABEL_COLUMNS)


def label_matrix_from_df(df: pd.DataFrame, label_columns: List[str]) -> np.ndarray:
    sub = df[label_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return sub.to_numpy(dtype=np.float32)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {ID_COLUMN, SEQUENCE_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {missing}. Found: {list(df.columns)}")
    return df


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    max_length: int,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is None or tok_max > 1_000_000:
        effective_cap = max_length
    else:
        effective_cap = min(max_length, int(tok_max))

    return model, tokenizer, effective_cap


def mean_pool_last_hidden(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pool over non-padding positions (padding tokens masked out)."""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def _forward_batch_to_pooled(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    batch_seqs: List[str],
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        batch_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)
    return mean_pool_last_hidden(outputs.last_hidden_state, encoded["attention_mask"])


def _is_oom_error(err: BaseException) -> bool:
    if isinstance(err, MemoryError):
        return True
    msg = str(err).lower()
    if "out of memory" in msg:
        return True
    if "cuda" in msg and "memory" in msg:
        return True
    if "mps" in msg and "allocate" in msg:
        return True
    return False


@torch.inference_mode()
def embed_with_oom_handling(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    sequences: List[str],
    accessions: List[str],
    targets: np.ndarray,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns (embeddings, accessions, targets, num_skipped_oom). Rows aligned."""
    n = len(sequences)
    assert n == len(accessions) == len(targets)
    all_emb: List[np.ndarray] = []
    all_acc: List[str] = []
    all_tgt: List[np.ndarray] = []
    skipped = 0

    num_batches = (n + batch_size - 1) // batch_size
    print(f"Embedding {n:,} sequences in {num_batches:,} batches (batch_size={batch_size}).")

    starts = list(range(0, n, batch_size))

    for start in tqdm(starts, desc="Embedding batches", unit="batch", total=num_batches):
        end = min(start + batch_size, n)
        batch_seqs = sequences[start:end]
        batch_acc = accessions[start:end]
        batch_tgt = targets[start:end]

        try:
            pooled = _forward_batch_to_pooled(model, tokenizer, device, batch_seqs, max_length)
            all_emb.append(pooled.detach().float().cpu().numpy())
            all_acc.extend(batch_acc)
            all_tgt.append(batch_tgt)
        except (RuntimeError, MemoryError) as e:
            if not _is_oom_error(e):
                raise
            clear_device_cache(device)
            for j in range(len(batch_seqs)):
                one_seq = [batch_seqs[j]]
                one_acc = batch_acc[j]
                one_t = batch_tgt[j : j + 1]
                try:
                    pooled_one = _forward_batch_to_pooled(model, tokenizer, device, one_seq, max_length)
                    all_emb.append(pooled_one.detach().float().cpu().numpy())
                    all_acc.append(one_acc)
                    all_tgt.append(one_t)
                except (RuntimeError, MemoryError) as e2:
                    if _is_oom_error(e2):
                        skipped += 1
                        print(
                            f"Warning: OOM on single sequence (skipped): accession={one_acc!r} "
                            f"(len={len(batch_seqs[j])} aa)"
                        )
                    else:
                        raise
                finally:
                    clear_device_cache(device)
        finally:
            clear_device_cache(device)

    if not all_emb:
        raise RuntimeError("No embeddings produced (all sequences skipped or empty input).")

    embeddings = np.vstack(all_emb)
    targets_out = np.vstack(all_tgt)
    acc_arr = np.array(all_acc, dtype=object)
    return embeddings, targets_out, acc_arr, skipped


def save_outputs(
    output_dir: Path,
    embeddings: np.ndarray,
    accessions: np.ndarray,
    multilabel_targets: np.ndarray,
    label_columns: List[str],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / "embeddings.npy"
    acc_path = output_dir / "accessions.npy"
    targets_path = output_dir / "multilabel_targets.npy"
    meta_path = output_dir / "metadata.json"
    labels_path = output_dir / "label_columns.json"

    np.save(emb_path, embeddings)
    np.save(acc_path, accessions)
    np.save(targets_path, multilabel_targets)

    label_payload = {
        "label_columns": label_columns,
        "num_labels": len(label_columns),
    }
    labels_path.write_text(json.dumps(label_payload, indent=2), encoding="utf-8")

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved embeddings:          {emb_path}  shape={tuple(embeddings.shape)}")
    print(f"Saved accessions:          {acc_path}  shape={tuple(accessions.shape)}")
    print(f"Saved multilabel_targets:  {targets_path}  shape={tuple(multilabel_targets.shape)}")
    print(f"Saved label_columns:       {labels_path}  ({len(label_columns)} labels)")
    print(f"Saved metadata:            {meta_path}")


def verify_saved_arrays(output_dir: Path) -> None:
    """Load saved artifacts and print integrity checks."""
    emb_path = output_dir / "embeddings.npy"
    acc_path = output_dir / "accessions.npy"
    tgt_path = output_dir / "multilabel_targets.npy"

    emb = np.load(emb_path)
    tgt = np.load(tgt_path)
    acc = np.load(acc_path, allow_pickle=True)

    print("\n--- Verification ---")
    print(f"embeddings.npy:  shape={emb.shape}, dtype={emb.dtype}")
    print(f"multilabel_targets.npy: shape={tgt.shape}, dtype={tgt.dtype}")
    print(f"accessions.npy:  shape={acc.shape}, dtype={acc.dtype}")

    n = emb.shape[0]
    ok = n == acc.shape[0] == tgt.shape[0]
    print(
        f"Row alignment:   embeddings={n}, accessions={acc.shape[0]}, targets={tgt.shape[0]} "
        f"-> {'OK (row i matches across arrays)' if ok else 'MISMATCH'}"
    )
    for i in range(min(3, n)):
        print(f"  row {i}: accession={acc.flat[i]!r}")

    if np.issubdtype(emb.dtype, np.floating):
        print(f"embeddings NaN:  {np.isnan(emb).any()}  Inf: {np.isinf(emb).any()}")
        print(f"embeddings min:  {np.nanmin(emb):.6g}  max: {np.nanmax(emb):.6g}")
    if np.issubdtype(tgt.dtype, np.floating):
        print(f"targets NaN:     {np.isnan(tgt).any()}  Inf: {np.isinf(tgt).any()}")
        print(f"targets min:     {np.nanmin(tgt):.6g}  max: {np.nanmax(tgt):.6g}")

    if acc.shape[0] >= 2:
        print(f"Sample accessions [0], [1]: {acc.flat[0]!r}, {acc.flat[1]!r}")
    print("--- End verification ---\n")


def run(
    input_csv: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    debug: bool,
    verify: bool,
) -> None:
    t_run_start = time.perf_counter()

    print(f"Project root (inferred): {ROOT}")
    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {model_name}")

    device = resolve_device()
    print(f"Device: {device}")

    if debug:
        print(f"=== RUN MODE: DEBUG (first {DEBUG_SAMPLE_SIZE} sequences after filters) ===")
    else:
        print("=== RUN MODE: FULL DATASET ===")

    df = load_dataset(input_csv)
    df = strip_dataframe_column_names(df)
    for req in (ID_COLUMN, SEQUENCE_COLUMN):
        if req not in df.columns:
            raise ValueError(
                f"Required column {req!r} not found after stripping whitespace from headers. "
                f"Columns: {list(df.columns)!r}"
            )

    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    label_columns = resolve_label_columns(df)
    print(f"Final label columns ({len(label_columns)}): {label_columns}")

    seq_series = df[SEQUENCE_COLUMN].astype(str).str.strip()
    kept_mask = seq_series.str.len() > 0
    dropped = int((~kept_mask).sum())
    if dropped:
        print(f"Warning: dropping {dropped} rows with empty sequences.")

    df_work = df.loc[kept_mask].reset_index(drop=True)

    if debug:
        df_work = df_work.iloc[:DEBUG_SAMPLE_SIZE].copy().reset_index(drop=True)
        print(f"Debug slice applied: {len(df_work):,} row(s) (cap={DEBUG_SAMPLE_SIZE:,}).")

    sequences = df_work[SEQUENCE_COLUMN].astype(str).str.strip().tolist()
    accessions = df_work[ID_COLUMN].astype(str).tolist()
    multilabel_targets = label_matrix_from_df(df_work, label_columns)

    print(f"Batch size: {batch_size}")

    model, tokenizer, effective_max = load_model_and_tokenizer(model_name, device, max_length)
    print(f"Tokenizer effective max_length: {effective_max}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_embed_start = time.perf_counter()
    embeddings, targets_out, acc_arr, skipped_oom = embed_with_oom_handling(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sequences=sequences,
        accessions=accessions,
        targets=multilabel_targets,
        batch_size=batch_size,
        max_length=effective_max,
    )
    t_embed_end = time.perf_counter()

    print(f"Embedding tensor shape: {embeddings.shape}  (samples × embedding_dim)")
    if skipped_oom:
        print(f"Warning: skipped {skipped_oom} sequence(s) due to OOM.")

    n = embeddings.shape[0]
    if acc_arr.shape[0] != n or targets_out.shape[0] != n:
        raise RuntimeError(
            "Row count mismatch after embedding: "
            f"embeddings={n}, accessions={acc_arr.shape[0]}, targets={targets_out.shape[0]}"
        )
    if targets_out.shape[1] != len(label_columns):
        raise RuntimeError(
            f"Target width mismatch: array has {targets_out.shape[1]} cols "
            f"but {len(label_columns)} label names."
        )

    embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else None
    time_taken = time.perf_counter() - t_run_start
    embedding_seconds = t_embed_end - t_embed_start

    peak_gpu_bytes: int | None = None
    if device.type == "cuda":
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated())

    generation_date = datetime.now(timezone.utc).isoformat()

    metadata: dict[str, Any] = {
        "model_name": model_name,
        "model_short_name": model_short_name_from_hf_id(model_name),
        "embedding_dim": embedding_dim,
        "generation_date": generation_date,
        "device_used": str(device),
        "time_taken_seconds": round(time_taken, 3),
        "embedding_phase_seconds": round(embedding_seconds, 3),
        "max_length": effective_max,
        "batch_size": batch_size,
        "embedding_shape": list(embeddings.shape),
        "hidden_dim": embedding_dim,
        "num_sequences": int(n),
        "skipped_due_to_oom": skipped_oom,
        "multilabel_targets_shape": list(targets_out.shape),
        "num_labels": len(label_columns),
        "label_columns": label_columns,
        "input_csv": str(input_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "id_column": ID_COLUMN,
        "sequence_column": SEQUENCE_COLUMN,
        "artifacts": {
            "embeddings": "embeddings.npy",
            "accessions": "accessions.npy",
            "multilabel_targets": "multilabel_targets.npy",
            "metadata": "metadata.json",
            "label_columns": "label_columns.json",
        },
    }
    if peak_gpu_bytes is not None:
        metadata["peak_gpu_memory_bytes"] = peak_gpu_bytes
        metadata["peak_gpu_memory_mib"] = round(peak_gpu_bytes / (1024 * 1024), 2)

    save_outputs(
        output_dir,
        embeddings,
        acc_arr,
        targets_out,
        label_columns,
        metadata,
    )

    print(f"Wall time (total run): {time_taken:.2f}s")
    print("Done.")

    if verify:
        verify_saved_arrays(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate ESM-2 protein embeddings for DeepLoc multilabel data.")
    p.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id (default: %(default)s)",
    )
    p.add_argument(
        "--input_csv",
        type=str,
        default=str(DEFAULT_INPUT_REL),
        help="Input CSV path (default: %(default)s, resolved under project root if relative)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed/embeddings/{model_short_name}/)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size (default: %(default)s)",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Max sequence length for tokenizer (default: %(default)s)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=f"Process only the first {DEBUG_SAMPLE_SIZE} rows after filters",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After saving, reload arrays and print shape/dtype/NaN/Inf/min/max checks",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    batch_size = args.batch_size
    env_batch = os.environ.get("PROTEIN_EMBEDDING_BATCH")
    if env_batch is not None:
        batch_size = max(1, int(env_batch))
        print(f"Batch size overridden by PROTEIN_EMBEDDING_BATCH={batch_size}")

    input_csv = resolve_input_csv(args.input_csv)
    output_dir = resolve_output_dir(args.model_name, args.output_dir)

    run(
        input_csv=input_csv,
        output_dir=output_dir,
        model_name=args.model_name,
        batch_size=batch_size,
        max_length=args.max_length,
        debug=args.debug,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
