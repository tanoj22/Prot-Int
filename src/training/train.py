"""
Entry point for training protein localization classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import ProteinLocalizationDataset, compute_class_weights, create_dataloaders, create_splits
from src.models.classifier import ProteinLocalizationClassifier
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.training.visualize import generate_all_plots, generate_comparison_plots


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EMBEDDINGS_DIR = ROOT / "data" / "processed" / "embeddings" / "esm2_t33_650M"

CONFIG_PRESETS: Dict[str, Dict[str, Any]] = {
    "v1": {
        "lr": 1e-3,
        "patience": 10,
        "dropout_rates": (0.3, 0.3, 0.2),
        "hidden_dims": (512, 256, 128),
        "weight_decay": 0.0,
        "label_smoothing": False,
    },
    "v2": {
        "lr": 5e-4,
        "patience": 15,
        "dropout_rates": (0.5, 0.4, 0.3),
        "hidden_dims": (256, 128, 64),
        "weight_decay": 1e-4,
        "label_smoothing": True,
    },
}


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multilabel protein localization classifier.")
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--experiment_name", type=str, default="protein_localization")
    parser.add_argument("--run_name", type=str, default=None, help="Defaults: baseline_v1 (v1) or regularized_v2 (v2).")
    parser.add_argument("--run_version", type=str, default="v1", help="Metadata tag for tracking (e.g. v1, v2).")
    parser.add_argument("--config_preset", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument("--lr", type=float, default=None, help="If set, overrides preset learning rate.")
    parser.add_argument("--patience", type=int, default=None, help="If set, overrides preset early-stopping patience.")
    return parser.parse_args()


def print_summary(
    dataset: ProteinLocalizationDataset,
    device: torch.device,
    config: Dict[str, Any],
) -> None:
    y = dataset._targets.numpy()
    pos_counts = y.sum(axis=0)
    pos_pct = 100.0 * pos_counts / max(len(dataset), 1)

    print("\n=== Training Summary ===")
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
    print("========================\n")


def _apply_preset(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    preset = CONFIG_PRESETS[args.config_preset]
    lr = float(args.lr) if args.lr is not None else float(preset["lr"])
    patience = int(args.patience) if args.patience is not None else int(preset["patience"])
    run_name = args.run_name or (
        "baseline_v1" if args.config_preset == "v1" else "regularized_v2"
    )
    return run_name, {
        "lr": lr,
        "patience": patience,
        "dropout_rates": tuple(preset["dropout_rates"]),
        "hidden_dims": tuple(int(x) for x in preset["hidden_dims"]),
        "weight_decay": float(preset["weight_decay"]),
        "label_smoothing": bool(preset["label_smoothing"]),
    }


def _save_run_artifacts(
    run_dir: Path,
    train_info: Mapping[str, Any],
    final_test_metrics: Mapping[str, Any],
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    history = dict(train_info.get("history", {}))
    payload = {
        **history,
        "best_epoch": train_info.get("best_epoch"),
        "best_val_macro_f1": train_info.get("best_val_macro_f1"),
    }
    (run_dir / "train_history.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps(dict(final_test_metrics), indent=2, default=str),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        y_pred_binary=y_pred,
    )


def main() -> None:
    args = parse_args()
    run_name, hyper = _apply_preset(args)

    embeddings_dir = Path(args.embeddings_dir).expanduser()
    if not embeddings_dir.is_absolute():
        embeddings_dir = (ROOT / embeddings_dir).resolve()

    dataset = ProteinLocalizationDataset(embeddings_dir=embeddings_dir)
    device = resolve_device()

    train_ds, val_ds, test_ds = create_splits(dataset)
    dataloaders = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=args.batch_size,
        num_workers=0,
    )

    pos_weights = compute_class_weights(train_ds).to(device)

    model = ProteinLocalizationClassifier(
        embedding_dim=dataset.embedding_dim,
        num_labels=dataset.num_labels,
        label_names=dataset.label_names,
        dropout_rates=hyper["dropout_rates"],
        hidden_dims=hyper["hidden_dims"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    loss_fn = get_loss_function(name=args.loss, pos_weights=pos_weights, device=device)

    checkpoint_dir = ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_artifacts_dir = ROOT / "models" / "run_artifacts" / run_name
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        "embeddings_dir": str(embeddings_dir),
        "batch_size": args.batch_size,
        "lr": hyper["lr"],
        "epochs": args.epochs,
        "patience": hyper["patience"],
        "loss": args.loss,
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "run_version": args.run_version,
        "config_preset": args.config_preset,
        "checkpoint_dir": str(checkpoint_dir),
        "weight_decay": hyper["weight_decay"],
        "label_smoothing": hyper["label_smoothing"],
        "label_smooth_low": 0.05,
        "label_smooth_high": 0.95,
        "dropout_rates": list(hyper["dropout_rates"]),
        "hidden_dims": list(hyper["hidden_dims"]),
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
    )

    train_info = trainer.train(
        num_epochs=args.epochs,
        patience=hyper["patience"],
        experiment_name=args.experiment_name,
        run_name=run_name,
    )
    final_test_metrics = trainer.evaluate(dataset_name="test")

    y_true_t, y_proba_t, y_bin_t = trainer.get_test_predictions("test")

    _save_run_artifacts(
        run_artifacts_dir,
        train_info,
        final_test_metrics,
        y_true_t,
        y_proba_t,
        y_bin_t,
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_model.pt"

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
            "dropout_rates": list(hyper["dropout_rates"]),
            "hidden_dims": list(hyper["hidden_dims"]),
            "train_info": train_info,
            "final_test_metrics": final_test_metrics,
            "config": config,
        },
        best_model_path,
    )
    print(f"Saved best model to: {best_model_path}")

    run_model_path = run_artifacts_dir / "best_model.pt"
    torch.save(
        {
            "state_dict": state_dict,
            "label_names": dataset.label_names,
            "embedding_dim": dataset.embedding_dim,
            "num_labels": dataset.num_labels,
            "dropout_rates": list(hyper["dropout_rates"]),
            "hidden_dims": list(hyper["hidden_dims"]),
            "train_info": train_info,
            "final_test_metrics": final_test_metrics,
            "config": config,
        },
        run_model_path,
    )
    print(f"Saved run checkpoint to: {run_model_path}")

    metrics_path = models_dir / "final_test_metrics.json"
    metrics_path.write_text(json.dumps(final_test_metrics, indent=2, default=str), encoding="utf-8")
    print(f"Saved final test metrics to: {metrics_path}")

    history = train_info.get("history", {})
    plots_dir = run_artifacts_dir / "plots"
    generate_all_plots(
        output_dir=plots_dir,
        train_losses=history.get("train_loss", []),
        val_losses=history.get("val_loss", []),
        epoch_metrics_list=history.get("val_per_class_f1", []),
        label_names=dataset.label_names,
        test_metrics=final_test_metrics,
        y_true=y_true_t,
        y_pred=y_bin_t,
        y_pred_proba=y_proba_t,
        best_epoch=(
            int(be)
            if (be := train_info.get("best_epoch")) is not None and int(be) >= 1
            else None
        ),
        mlflow_log=True,
    )

    if args.config_preset == "v2":
        v1_dir = ROOT / "models" / "run_artifacts" / "baseline_v1"
        v1_hist = v1_dir / "train_history.json"
        v1_metrics = v1_dir / "final_test_metrics.json"
        v1_npz = v1_dir / "test_predictions.npz"
        if v1_hist.is_file() and v1_metrics.is_file() and v1_npz.is_file():
            cmp_dir = ROOT / "models" / "run_artifacts" / "comparison_v1_vs_v2"
            generate_comparison_plots(
                v1_artifact_dir=v1_dir,
                v2_artifact_dir=run_artifacts_dir,
                label_names=dataset.label_names,
                output_dir=cmp_dir,
                mlflow_log=True,
            )
            print(f"Saved v1 vs v2 comparison plots to: {cmp_dir}")
        else:
            print(
                "Skipping v1 vs v2 comparison plots: expected baseline_v1 artifacts at "
                f"{v1_dir} (train_history.json, final_test_metrics.json, test_predictions.npz)."
            )


if __name__ == "__main__":
    main()
