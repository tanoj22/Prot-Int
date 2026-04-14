"""
Smoke-test src/data/dataset.py against a real embeddings folder.

From project root:
    python scripts/test_dataset.py
    python scripts/test_dataset.py --embeddings-dir data/processed/embeddings/esm2_t33_650M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import (  # noqa: E402
    ProteinLocalizationDataset,
    compute_class_weights,
    create_dataloaders,
    create_splits,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Test ProteinLocalizationDataset pipeline.")
    p.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "embeddings" / "esm2_t12_35M",
        help="Folder containing embeddings.npy, accessions.npy, multilabel_targets.npy, label_columns.json",
    )
    args = p.parse_args()
    emb_dir = args.embeddings_dir
    if not emb_dir.is_absolute():
        emb_dir = (ROOT / emb_dir).resolve()

    print(f"Embeddings dir: {emb_dir}")
    if not emb_dir.is_dir():
        print("ERROR: directory does not exist. Run embeddings.py first or pass --embeddings-dir.")
        sys.exit(1)

    ds = ProteinLocalizationDataset(emb_dir)
    print(f"len(dataset) = {len(ds)}")
    print(f"embedding_dim = {ds.embedding_dim}")
    print(f"num_labels = {ds.num_labels}")
    print(f"label_names ({len(ds.label_names)}): {ds.label_names}")

    e, t, acc = ds[0]
    print(f"\nSample [0]: accession={acc!r}, emb shape={tuple(e.shape)}, target shape={tuple(t.shape)}")

    train_ds, val_ds, test_ds = create_splits(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    print(f"\nSplits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    loaders = create_dataloaders(train_ds, val_ds, test_ds, batch_size=8, num_workers=0)
    emb_b, tgt_b, acc_b = next(iter(loaders["train"]))
    print(f"\nTrain batch: emb {tuple(emb_b.shape)}, tgt {tuple(tgt_b.shape)}, {len(acc_b)} accessions")

    w = compute_class_weights(train_ds)
    print(f"\npos_weight tensor shape: {tuple(w.shape)}")

    print("\nOK — dataset.py smoke test passed.")


if __name__ == "__main__":
    main()
