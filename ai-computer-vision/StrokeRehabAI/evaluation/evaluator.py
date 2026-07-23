"""
evaluator.py
============
Runs a trained model over a held-out test set and computes the
configured evaluation metrics (see configs/evaluation.yaml): accuracy,
precision, recall, F1, ROC AUC, confusion matrix, plus inference-speed
(FPS / latency) and GPU-utilization benchmarking.

Run via:
    python -m evaluation.evaluator --checkpoint weights/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from training.gpu_monitor import InferenceBenchmark, read_gpu_utilization
from training.metrics import compute_classification_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates a model against a test DataLoader and computes the full
    configured metric suite (see configs/evaluation.yaml -> metrics)."""

    def __init__(self, model, device, num_classes: int):
        self.model = model
        self.device = device
        self.num_classes = num_classes

    def evaluate(self, test_loader, benchmark_speed: bool = True) -> Dict:
        import torch

        self.model.eval()
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []
        sample_batch = None

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if sample_batch is None:
                    sample_batch = batch_x[:1].clone()

                batch_x = batch_x.to(self.device)
                probs = self.model.predict_proba(batch_x)
                preds = torch.argmax(probs, dim=-1).cpu().numpy()

                all_preds.extend(preds.tolist())
                all_targets.extend(batch_y.numpy().tolist())
                all_probs.append(probs.cpu().numpy())

        probabilities = np.concatenate(all_probs, axis=0) if all_probs else None
        metrics = compute_classification_metrics(
            np.array(all_preds), np.array(all_targets), num_classes=self.num_classes, probabilities=probabilities
        )

        gpu_snapshot = read_gpu_utilization()
        metrics["gpu_utilization_percent"] = gpu_snapshot.gpu_utilization_percent
        metrics["gpu_memory_used_mb"] = gpu_snapshot.memory_used_mb

        if benchmark_speed and sample_batch is not None:
            benchmark = InferenceBenchmark(self.model, self.device)
            metrics.update(benchmark.run(sample_batch))

        logger.info(
            "Evaluation complete: %s",
            {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        )
        return metrics


def main() -> None:
    from configs.config_loader import load_config
    from evaluation.report_generator import EvaluationReportGenerator
    from models.model_factory import build_model
    from training.dataset_loader import build_dataloaders
    from utils.checkpoint_utils import load_model_weights
    from utils.gpu_utils import get_device

    parser = argparse.ArgumentParser(description="Evaluate a trained StrokeRehabAI model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint (.pt)")
    args = parser.parse_args()

    cfg = load_config()
    device = get_device(cfg.gpu.use_cuda_if_available, cfg.gpu.device_index, cfg.gpu.fallback_to_cpu)
    model = build_model(cfg.model).to(device)
    load_model_weights(model, args.checkpoint, map_location=str(device))

    _, _, test_loader = build_dataloaders(cfg.training, cfg.datasets, cfg.model, cfg.get("augmentation"))
    evaluator = ModelEvaluator(model, device, num_classes=cfg.model.num_classes)
    metrics = evaluator.evaluate(test_loader)

    EvaluationReportGenerator(cfg.evaluation.report_output_dir).save(metrics, experiment_name="cli_eval")


if __name__ == "__main__":
    main()
