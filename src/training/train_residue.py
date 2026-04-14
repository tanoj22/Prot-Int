"""
Entry point for training the residue-level localization classifier.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset

from src.data.dataset import compute_class_weights, create_splits
from src.data.residue_dataset import ResidueDataset, create_residue_dataloaders
from src.models.residue_classifier import ResidueLocalizationClassifier
from src.training.losses import get_loss_function
from src.training.trainer import Trainer

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EMBEDDINGS_DIR = ROOT / "data" / "processed" / "residue_embeddings" / "esm2_t33_650M"


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residue-level multilabel localization classifier.")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help="Directory with per-protein .npz residue embeddings and index files.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--experiment_name", type=str, default="residue_classifier")
    parser.add_argument("--run_name", type=str, default=None, help="Defaults to residue_baseline.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_heads", type=int, default=4, help="Reserved / API compatibility for the classifier.")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    return parser.parse_args()


def print_summary(dataset: ResidueDataset, device: torch.device, config: Dict[str, Any]) -> None:
    y = dataset._targets.numpy()
    pos_counts = y.sum(axis=0)
    pos_pct = 100.0 * pos_counts / max(len(dataset), 1)

    print("\n=== Residue training summary ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Embedding dim: {dataset.embedding_dim}")
    print(f"Num labels: {dataset.num_labels}")
    print(f"Device: {device}")
    print("Hyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("Class distribution (positives):")
    for i, name in enumerate(dataset.label_names):
        print(f"  {name}: {int(pos_counts[i])} ({pos_pct[i]:.2f}%)")
    print("================================\n")


def _safe_filename(name: str) -> str:
    out = re.sub(r"[^\w.\-]+", "_", str(name).strip())
    return (out or "protein")[:120]


def make_attention_artifact_hook(test_subset: Subset) -> Callable[[Trainer], None]:
    """Log attention visualizations for the first three test proteins (MLflow run must be active)."""

    def post_eval_hook(trainer: Trainer) -> None:
        try:
            import mlflow
        except ImportError:
            return
        if mlflow.active_run() is None:
            return

        model = trainer.model
        if not hasattr(model, "get_attention_weights"):
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        device = trainer.device
        out_dir = trainer.checkpoint_dir / "attention_plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_show = min(3, len(test_subset))
        for i in range(n_show):
            emb, _tgt, mask, acc = test_subset[i]
            acc_str = str(acc)
            emb_b = emb.unsqueeze(0).to(device, non_blocking=True)
            mask_b = mask.unsqueeze(0).to(device, non_blocking=True)

            model.eval()
            with torch.no_grad():
                _, attn = model.get_attention_weights(emb_b, mask_b)  # type: ignore[union-attr]

            attn_np = attn[0].detach().float().cpu().numpy()
            valid = mask_b[0].to(dtype=torch.bool).cpu().numpy()
            seq_len = int(valid.sum())
            attn_trim = attn_np[:seq_len]
            positions = np.arange(1, seq_len + 1)

            top_k = min(10, seq_len)
            top_local = np.argsort(-attn_trim)[:top_k]

            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(positions, attn_trim, color="steelblue", linewidth=1.2)
            ax.scatter(positions[top_local], attn_trim[top_local], color="darkorange", s=22, zorder=3)
            ax.set_xlabel("Residue index (1-based)")
            ax.set_ylabel("Attention weight")
            top_pos_str = ", ".join(str(int(positions[j])) for j in top_local)
            ax.set_title(f"Attention — {acc_str}\nTop-{top_k} residues (1-based pos): {top_pos_str}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            path = out_dir / f"attention_{_safe_filename(acc_str)}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)

            mlflow.log_artifact(str(path), artifact_path="attention")

    return post_eval_hook


def main() -> None:
    args = parse_args()
    run_name = args.run_name or "residue_baseline"

    embeddings_dir = Path(args.embeddings_dir).expanduser()
    if not embeddings_dir.is_absolute():
        embeddings_dir = (ROOT / embeddings_dir).resolve()

    dataset = ResidueDataset(residue_embeddings_dir=embeddings_dir, max_seq_len=args.max_seq_len)
    device = resolve_device()

    train_ds, val_ds, test_ds = create_splits(dataset)
    dataloaders = create_residue_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=args.batch_size,
        num_workers=0,
    )

    pos_weights = compute_class_weights(train_ds).to(device)

    model = ResidueLocalizationClassifier(
        embedding_dim=dataset.embedding_dim,
        num_labels=dataset.num_labels,
        num_heads=args.num_heads,
        dropout=args.dropout,
        label_names=dataset.label_names,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    loss_fn = get_loss_function(name=args.loss, pos_weights=pos_weights, device=device)

    checkpoint_dir = ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        "task": "residue_localization",
        "embeddings_dir": str(embeddings_dir),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "loss": args.loss,
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "checkpoint_dir": str(checkpoint_dir),
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "max_seq_len": args.max_seq_len,
        "label_smoothing": False,
    }

    print_summary(dataset=dataset, device=device, config=config)

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device,
        label_names=dataset.label_names,
        forward_with_mask=True,
    )

    train_info = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience,
        experiment_name=args.experiment_name,
        run_name=run_name,
        post_eval_hook=make_attention_artifact_hook(test_ds),
    )
    final_test_metrics = trainer.evaluate(dataset_name="test")

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_residue_path = models_dir / "best_residue_model.pt"

    if trainer.best_checkpoint_path and Path(trainer.best_checkpoint_path).is_file():
        ckpt = torch.load(trainer.best_checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
    else:
        state_dict = trainer.model.state_dict()

    torch.save(
        {
            "state_dict": state_dict,
            "label_names": dataset.label_names,
            "embedding_dim": dataset.embedding_dim,
            "num_labels": dataset.num_labels,
            "dropout": args.dropout,
            "num_heads": args.num_heads,
            "model_type": "residue_localization",
            "train_info": train_info,
            "final_test_metrics": final_test_metrics,
            "config": config,
        },
        best_residue_path,
    )
    print(f"Saved best residue model to: {best_residue_path}")

    metrics_path = models_dir / "final_test_metrics_residue.json"
    metrics_path.write_text(json.dumps(final_test_metrics, indent=2, default=str), encoding="utf-8")
    print(f"Saved final test metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
