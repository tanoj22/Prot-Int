"""
Training loop and evaluation utilities for multilabel protein localization.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment, misc]

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .metrics import compute_metrics, find_optimal_thresholds, format_metrics_table


def _safe_metric_name(name: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "label"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        device: torch.device | str,
        label_names: List[str],
        *,
        forward_with_mask: bool = False,
    ) -> None:
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = dict(config)
        self.device = torch.device(device)
        self.label_names = list(label_names)
        self.forward_with_mask = bool(forward_with_mask)

        self.model.to(self.device)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        root = Path(__file__).resolve().parent.parent.parent
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", root / "models" / "checkpoints")).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_macro_f1 = -float("inf")
        self.best_checkpoint_path: Path | None = None
        self.best_epoch: int = -1
        self.optimal_thresholds: Dict[str, float] | None = None
        self.final_test_metrics: Dict[str, Any] | None = None

        self.label_smoothing: bool = bool(self.config.get("label_smoothing", False))
        self._ls_low: float = float(self.config.get("label_smooth_low", 0.05))
        self._ls_high: float = float(self.config.get("label_smooth_high", 0.95))

    def _smooth_targets_for_loss(self, targets: torch.Tensor) -> torch.Tensor:
        """Map hard 0/1 labels to [low, high] for training loss only (multilabel)."""
        if not self.label_smoothing:
            return targets
        lo, hi = self._ls_low, self._ls_high
        return targets * (hi - lo) + lo

    def _autocast_context(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)

    def _run_loader_eval(
        self,
        split: str,
        thresholds: Dict[str, float] | None = None,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        loader = self.dataloaders[split]
        self.model.eval()

        total_loss = 0.0
        total_count = 0
        y_true_chunks: List[np.ndarray] = []
        y_proba_chunks: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                if self.forward_with_mask:
                    embeddings, targets, mask, _ = batch
                    mask = mask.to(self.device, non_blocking=True)
                else:
                    embeddings, targets, _ = batch
                    mask = None
                embeddings = embeddings.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with self._autocast_context():
                    if self.forward_with_mask:
                        logits = self.model(embeddings, mask=mask)
                    else:
                        logits = self.model(embeddings)
                    loss = self._compute_loss(logits, targets)

                bsz = embeddings.shape[0]
                total_loss += float(loss.item()) * bsz
                total_count += bsz

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                y = targets.detach().cpu().numpy()
                y_proba_chunks.append(probs)
                y_true_chunks.append(y)

        avg_loss = total_loss / max(total_count, 1)
        y_true = np.vstack(y_true_chunks)
        y_proba = np.vstack(y_proba_chunks)

        if thresholds is None:
            y_bin = (y_proba >= 0.5).astype(np.int64)
        else:
            thr = np.asarray([float(thresholds.get(n, 0.5)) for n in self.label_names], dtype=np.float32)
            y_bin = (y_proba >= thr[None, :]).astype(np.int64)

        return avg_loss, y_true, y_proba, y_bin

    def _save_checkpoint(self, epoch: int, val_macro_f1: float) -> Path:
        path = self.checkpoint_dir / "best_model_checkpoint.pt"
        payload = {
            "epoch": epoch,
            "val_macro_f1": float(val_macro_f1),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "label_names": self.label_names,
            "config": self.config,
        }
        torch.save(payload, path)
        return path

    def _load_best_checkpoint_into_model(self) -> None:
        if self.best_checkpoint_path is None or not self.best_checkpoint_path.is_file():
            return
        ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def train(
        self,
        num_epochs: int,
        patience: int,
        experiment_name: str,
        run_name: str,
        post_eval_hook: Optional[Callable[["Trainer"], None]] = None,
    ) -> Dict[str, Any]:
        use_mlflow = mlflow is not None
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
            "val_micro_f1": [],
            "val_per_class_f1": [],
        }

        epochs_without_improvement = 0
        train_loader = self.dataloaders["train"]

        run_ctx = mlflow.start_run(run_name=run_name) if use_mlflow else nullcontext()
        with run_ctx:
            model_params = sum(p.numel() for p in self.model.parameters())
            embedding_dim = (
                int(getattr(getattr(self.model, "net", [None])[0], "in_features", -1))
                if hasattr(self.model, "net")
                else -1
            )
            params_to_log = dict(self.config)
            params_to_log.update(
                {
                    "num_epochs": int(num_epochs),
                    "patience": int(patience),
                    "device": str(self.device),
                    "model_architecture": str(self.model),
                    "embedding_dim": int(embedding_dim),
                    "num_params": int(model_params),
                    "loss_function_name": self.loss_fn.__class__.__name__,
                }
            )
            if use_mlflow:
                mlflow.log_params({k: str(v) for k, v in params_to_log.items()})
                mlflow.set_tags(
                    {
                        "run_version": str(self.config.get("run_version", "")),
                        "config_preset": str(self.config.get("config_preset", "")),
                        "run_name": str(self.config.get("run_name", "")),
                    }
                )

            for epoch in range(1, num_epochs + 1):
                self.model.train()
                running_loss = 0.0
                running_count = 0

                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
                for batch in pbar:
                    if self.forward_with_mask:
                        embeddings, targets, mask, _ = batch
                        mask = mask.to(self.device, non_blocking=True)
                    else:
                        embeddings, targets, _ = batch
                        mask = None
                    embeddings = embeddings.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad(set_to_none=True)
                    with self._autocast_context():
                        if self.forward_with_mask:
                            logits = self.model(embeddings, mask=mask)
                        else:
                            logits = self.model(embeddings)
                        targets_loss = self._smooth_targets_for_loss(targets)
                        loss = self._compute_loss(logits, targets_loss)

                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                    bsz = embeddings.shape[0]
                    running_loss += float(loss.item()) * bsz
                    running_count += bsz
                    pbar.set_postfix(loss=f"{(running_loss / max(running_count,1)):.4f}")

                train_loss = running_loss / max(running_count, 1)
                val_loss, y_true_val, y_proba_val, y_bin_val = self._run_loader_eval("val", thresholds=None)
                val_metrics = compute_metrics(
                    y_true=y_true_val,
                    y_pred_proba=y_proba_val,
                    y_pred_binary=y_bin_val,
                    label_names=self.label_names,
                )

                val_macro_f1 = float(val_metrics["macro_f1"])
                val_micro_f1 = float(val_metrics["micro_f1"])
                val_hamming = float(val_metrics["hamming_loss"])

                self.scheduler.step(val_macro_f1)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_macro_f1"].append(val_macro_f1)
                history["val_micro_f1"].append(val_micro_f1)
                history["val_per_class_f1"].append(
                    {name: float(val_metrics["per_class"][name]["f1"]) for name in self.label_names}
                )

                epoch_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "val_micro_f1": val_micro_f1,
                    "val_hamming_loss": val_hamming,
                }
                for name in self.label_names:
                    key = f"val_f1_{_safe_metric_name(name)}"
                    epoch_metrics[key] = float(val_metrics["per_class"][name]["f1"])
                if use_mlflow:
                    mlflow.log_metrics(epoch_metrics, step=epoch)

                improved = val_macro_f1 > (self.best_val_macro_f1 + 1e-12)
                if improved:
                    self.best_val_macro_f1 = val_macro_f1
                    self.best_epoch = epoch
                    self.best_checkpoint_path = self._save_checkpoint(epoch=epoch, val_macro_f1=val_macro_f1)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_macro_f1={val_macro_f1:.4f} val_micro_f1={val_micro_f1:.4f}"
                )

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no val_macro_f1 improvement for {patience} epoch(s))."
                    )
                    break

            self._load_best_checkpoint_into_model()
            test_metrics = self.evaluate(dataset_name="test")
            self.final_test_metrics = test_metrics

            if post_eval_hook is not None:
                post_eval_hook(self)

            if self.optimal_thresholds is not None:
                thresholds_path = self.checkpoint_dir / "optimal_thresholds.json"
                thresholds_path.write_text(json.dumps(self.optimal_thresholds, indent=2), encoding="utf-8")
                if use_mlflow:
                    mlflow.log_artifact(str(thresholds_path), artifact_path="artifacts")

            test_metrics_path = self.checkpoint_dir / "final_test_metrics.json"
            test_metrics_path.write_text(json.dumps(test_metrics, indent=2, default=str), encoding="utf-8")
            if use_mlflow:
                if self.best_checkpoint_path is not None and self.best_checkpoint_path.is_file():
                    mlflow.log_artifact(str(self.best_checkpoint_path), artifact_path="model")
                mlflow.log_artifact(str(test_metrics_path), artifact_path="artifacts")
                mlflow.log_metric("best_val_macro_f1", float(self.best_val_macro_f1))
                mlflow.log_metric("best_epoch", float(self.best_epoch))
                if "macro_f1" in test_metrics:
                    mlflow.log_metric("test_macro_f1", float(test_metrics["macro_f1"]))
                if "micro_f1" in test_metrics:
                    mlflow.log_metric("test_micro_f1", float(test_metrics["micro_f1"]))

        return {
            "best_val_macro_f1": float(self.best_val_macro_f1),
            "best_epoch": int(self.best_epoch),
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "history": history,
        }

    def evaluate(self, dataset_name: str = "test") -> Dict[str, Any]:
        if dataset_name not in self.dataloaders:
            raise ValueError(f"Unknown dataset_name={dataset_name!r}. Expected one of {list(self.dataloaders)}")

        _, y_true_val, y_proba_val, _ = self._run_loader_eval("val", thresholds=None)
        thresholds = find_optimal_thresholds(
            y_true=y_true_val,
            y_pred_proba=y_proba_val,
            label_names=self.label_names,
        )
        self.optimal_thresholds = thresholds

        eval_loss, y_true_eval, y_proba_eval, y_bin_eval = self._run_loader_eval(dataset_name, thresholds=thresholds)
        metrics = compute_metrics(
            y_true=y_true_eval,
            y_pred_proba=y_proba_eval,
            y_pred_binary=y_bin_eval,
            label_names=self.label_names,
        )
        metrics["loss"] = float(eval_loss)
        metrics["thresholds"] = thresholds

        print(f"\n[{dataset_name.upper()}] loss={eval_loss:.4f}")
        print(format_metrics_table(metrics, self.label_names))

        nan_auroc_labels = [
            name for name, d in metrics["per_class"].items() if isinstance(d.get("auroc"), float) and math.isnan(d["auroc"])
        ]
        if nan_auroc_labels:
            print(f"Note: AUROC undefined for classes with no positives/negatives: {nan_auroc_labels}")

        return metrics

    def get_test_predictions(self, dataset_name: str = "test") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``y_true``, ``y_pred_proba``, ``y_pred_binary`` on ``dataset_name`` using optimal thresholds."""
        if self.optimal_thresholds is None:
            raise RuntimeError("evaluate() must be called before get_test_predictions().")
        _, y_true, y_proba, y_bin = self._run_loader_eval(dataset_name, self.optimal_thresholds)
        return y_true, y_proba, y_bin
