"""
trainer.py
==========
Main training loop orchestration: wires together the model, optimizer,
scheduler, loss, dataloaders, checkpointing, TensorBoard logging, and
progress bars into complete train / validate / test loops, with resume,
early stopping, gradient clipping, and mixed precision support.

Run via:
    python -m training.trainer
    python -m training.trainer --resume
    python -m training.trainer --resume weights/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from configs.config_loader import load_config
from evaluation.evaluator import ModelEvaluator
from evaluation.report_generator import EvaluationReportGenerator
from models.model_factory import build_model
from training.checkpoint_manager import CheckpointManager
from training.gpu_monitor import read_gpu_utilization
from training.loss_factory import build_loss, compute_class_weights
from training.metrics import compute_classification_metrics
from training.optimizer_factory import build_optimizer
from training.scheduler import build_scheduler
from training.tensorboard_logger import TensorboardLogger
from utils.gpu_utils import apply_gpu_settings, get_device
from utils.logger import get_logger, log_gpu_info
from utils.seed import set_global_seed

logger = get_logger(__name__)


class EarlyStopping:
    """Stops training when a monitored validation metric stops improving
    for `patience` consecutive validation rounds."""

    def __init__(self, patience: int = 15, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.num_bad_rounds = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update state with the latest metric value; returns True if
        training should stop."""
        improved = (
            self.best_value is None
            or (self.mode == "min" and value < self.best_value - self.min_delta)
            or (self.mode == "max" and value > self.best_value + self.min_delta)
        )
        if improved:
            self.best_value = value
            self.num_bad_rounds = 0
        else:
            self.num_bad_rounds += 1
            if self.num_bad_rounds >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """Encapsulates one full training run driven by the merged application config."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        set_global_seed(self.cfg.training.seed)
        apply_gpu_settings(self.cfg.gpu.allow_tf32, self.cfg.gpu.cudnn_benchmark)
        log_gpu_info()

        self.device = get_device(
            use_cuda_if_available=self.cfg.gpu.use_cuda_if_available,
            device_index=self.cfg.gpu.device_index,
            fallback_to_cpu=self.cfg.gpu.fallback_to_cpu,
        )

        self.model = build_model(self.cfg.model).to(self.device)
        self.optimizer = build_optimizer(self.model, self.cfg.training)
        self.scheduler = build_scheduler(self.optimizer, self.cfg.training)
        self.loss_fn = None  # built lazily in fit() once class weights (if any) are known

        self.checkpoint_manager = CheckpointManager(
            self.cfg.training.checkpoint_dir,
            self.cfg.training.save_top_k,
            metric_mode="max" if "accuracy" in self.cfg.training.get("early_stopping_metric", "val_loss") or
            "f1" in self.cfg.training.get("early_stopping_metric", "val_loss") else "min",
        )
        self.early_stopping = EarlyStopping(
            patience=self.cfg.training.early_stopping_patience,
            mode=self.checkpoint_manager.metric_mode,
        )

        self.tb_logger = None
        if self.cfg.training.get("tensorboard", {}).get("enabled", True) if hasattr(self.cfg.training, "get") else True:
            try:
                self.tb_logger = TensorboardLogger(
                    log_dir=self.cfg.training.tensorboard.log_dir if hasattr(self.cfg.training, "tensorboard") else "logs/tensorboard"
                )
            except ImportError:
                logger.warning("TensorBoard not available; continuing without TB logging.")

        self.use_amp = bool(self.cfg.training.mixed_precision) and self.device.type == "cuda"
        self.scaler = None

        self.start_epoch = 1
        self.global_step = 0
        self._history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader=None, val_loader=None, resume_path: Optional[str] = None) -> None:
        """Run the full training loop with validation, checkpointing,
        early stopping, and TensorBoard logging.

        If `train_loader` is None, builds dataloaders automatically from
        `configs/datasets.yaml`. Pass `resume_path` (or set
        `training.resume_from_checkpoint` in configs/training.yaml) to
        continue from a saved checkpoint.
        """
        import torch

        if train_loader is None or val_loader is None:
            from training.dataset_loader import build_dataloaders

            built_train, built_val, _ = build_dataloaders(
                self.cfg.training, self.cfg.datasets, self.cfg.model, self.cfg.augmentation
            )
            train_loader = train_loader or built_train
            val_loader = val_loader or built_val

        class_weights = self._resolve_class_weights(train_loader)
        self.loss_fn = build_loss(self.cfg.model, self.cfg.training, class_weights=class_weights)
        self.loss_fn = self.loss_fn.to(self.device) if hasattr(self.loss_fn, "to") else self.loss_fn

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        resume_path = resume_path or self.cfg.training.resume_from_checkpoint
        if resume_path:
            self._resume(resume_path)

        logger.info(
            "Starting training: epochs=%d, batch_size=%d, device=%s, mixed_precision=%s",
            self.cfg.training.epochs, self.cfg.training.batch_size, self.device, self.use_amp,
        )

        for epoch in range(self.start_epoch, self.cfg.training.epochs + 1):
            train_metrics = self._train_one_epoch(train_loader, epoch)

            val_metrics = {}
            if epoch % self.cfg.training.validate_every_n_epochs == 0:
                val_metrics = self._validate(val_loader, epoch)
                self._checkpoint_epoch(epoch, val_metrics)

                if self.scheduler is not None:
                    self._step_scheduler(val_metrics.get("val_loss"))

                monitored = val_metrics.get(self.cfg.training.early_stopping_metric, val_metrics.get("val_loss"))
                if monitored is not None and self.early_stopping.step(monitored):
                    logger.info(
                        "Early stopping triggered at epoch %d (no improvement in %s for %d rounds).",
                        epoch, self.cfg.training.early_stopping_metric, self.early_stopping.patience,
                    )
                    break

            self._log_epoch_summary(epoch, train_metrics, val_metrics)

        if self._history:
            from evaluation.report_generator import EvaluationReportGenerator

            EvaluationReportGenerator(self.cfg.evaluation.report_output_dir).save_training_curves(
                self._history, experiment_name=self.cfg.training.experiment_name
            )

        if self.tb_logger:
            self.tb_logger.close()

    def test(self, test_loader=None, checkpoint_path: Optional[str] = None) -> Dict:
        """Run the full evaluation suite on the held-out test set,
        optionally loading a specific checkpoint first (defaults to the
        current in-memory model / the best saved checkpoint if present)."""
        if test_loader is None:
            from training.dataset_loader import build_dataloaders

            _, _, test_loader = build_dataloaders(
                self.cfg.training, self.cfg.datasets, self.cfg.model, self.cfg.augmentation
            )

        if checkpoint_path:
            from utils.checkpoint_utils import load_model_weights

            load_model_weights(self.model, checkpoint_path, map_location=str(self.device))

        evaluator = ModelEvaluator(self.model, self.device, num_classes=self.cfg.model.num_classes)
        metrics = evaluator.evaluate(test_loader)

        report_generator = EvaluationReportGenerator(self.cfg.evaluation.report_output_dir)
        report_generator.save(metrics, experiment_name=f"{self.cfg.training.experiment_name}_test")
        return metrics

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        import torch

        self.model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        iterator = self._progress_bar(train_loader, desc=f"Epoch {epoch} [train]")

        for step, (inputs, targets) in enumerate(iterator):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip_norm)
                self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.cfg.training.log_every_n_steps == 0:
                self._log_step(loss.item(), epoch)

            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(1, num_batches)
        elapsed = time.time() - epoch_start
        logger.info("Epoch %d train complete: loss=%.4f, time=%.1fs", epoch, mean_loss, elapsed)
        return {"train_loss": mean_loss, "epoch_time_sec": elapsed}

    def _validate(self, val_loader, epoch: int) -> Dict[str, float]:
        import torch

        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        all_preds, all_targets, all_probs = [], [], []

        iterator = self._progress_bar(val_loader, desc=f"Epoch {epoch} [val]")

        with torch.no_grad():
            for inputs, targets in iterator:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item()
                num_batches += 1

                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())

        import torch as _torch

        preds_cat = _torch.cat(all_preds) if all_preds else _torch.tensor([])
        targets_cat = _torch.cat(all_targets) if all_targets else _torch.tensor([])
        probs_cat = _torch.cat(all_probs) if all_probs else None

        classification_metrics = compute_classification_metrics(
            preds_cat, targets_cat, num_classes=self.cfg.model.num_classes, probabilities=probs_cat
        )

        val_metrics = {"val_loss": running_loss / max(1, num_batches)}
        for key, value in classification_metrics.items():
            if key != "confusion_matrix":
                val_metrics[f"val_{key}"] = value

        gpu_snapshot = read_gpu_utilization(self.cfg.gpu.device_index)
        if gpu_snapshot.gpu_utilization_percent is not None:
            val_metrics["gpu_utilization_percent"] = gpu_snapshot.gpu_utilization_percent

        if self.tb_logger:
            self.tb_logger.log_scalars("val", val_metrics, epoch)

        return val_metrics

    # ------------------------------------------------------------------
    # Checkpointing / resume
    # ------------------------------------------------------------------

    def _checkpoint_epoch(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        state = self.checkpoint_manager.build_state(
            self.model, self.optimizer, self.scheduler, epoch, val_metrics, config=dict(self.cfg.training)
        )
        self.checkpoint_manager.save_last({**state, "epoch": epoch})

        monitored = val_metrics.get(self.cfg.training.early_stopping_metric, val_metrics.get("val_loss"))
        if monitored is not None:
            self.checkpoint_manager.maybe_save_best(state, epoch, monitored)

    def _resume(self, resume_path: str) -> None:
        path = Path(resume_path) if resume_path != "last" else self.checkpoint_manager.checkpoint_dir / "last.pt"
        checkpoint = self.checkpoint_manager.load_for_resume(str(path))

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info("Resumed training from epoch %d.", self.start_epoch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_class_weights(self, train_loader):
        loss_params = self.cfg.training.get("loss_params", {}) if hasattr(self.cfg.training, "get") else {}
        configured = loss_params.get("class_weights") if loss_params else None
        if configured != "auto":
            return None

        logger.info("Computing automatic inverse-frequency class weights from the training split...")
        labels = []
        for _, targets in train_loader:
            labels.extend(targets.tolist())
        return compute_class_weights(labels, self.cfg.model.num_classes)

    def _step_scheduler(self, val_loss: Optional[float]) -> None:
        import torch.optim.lr_scheduler as lr_scheduler

        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _log_step(self, loss_value: float, epoch: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_scalars("train", {"loss": loss_value}, self.global_step)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.tb_logger.log_learning_rate(current_lr, self.global_step)

    def _log_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        summary = {"epoch": epoch, **train_metrics, **val_metrics}
        self._history.append(summary)
        logger.info("Epoch %d summary: %s", epoch, summary)

    def _progress_bar(self, iterable, desc: str):
        if not (self.cfg.training.get("progress_bar", True) if hasattr(self.cfg.training, "get") else True):
            return iterable
        try:
            from tqdm import tqdm

            return tqdm(iterable, desc=desc, leave=False)
        except ImportError:
            return iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the StrokeRehabAI movement-assessment model.")
    parser.add_argument("--resume", nargs="?", const="last", default=None, help="Resume from a checkpoint (default: last.pt).")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run the test loop only.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load for --test-only.")
    args = parser.parse_args()

    trainer = Trainer()
    if args.test_only:
        trainer.test(checkpoint_path=args.checkpoint)
    else:
        trainer.fit(resume_path=args.resume)
        trainer.test()


if __name__ == "__main__":
    main()
