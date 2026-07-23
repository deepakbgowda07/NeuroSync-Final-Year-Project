"""
report_generator.py
====================
Persists evaluation results to outputs/evaluation_reports/, matching
configs/evaluation.yaml -> report_output_dir and save_confusion_matrix_plot:
metrics as both JSON and CSV, a confusion-matrix image, and (given a
per-epoch training history) loss/learning-curve plots.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.csv_export import write_dicts_to_csv
from utils.file_io import ensure_dir
from utils.json_export import write_json
from utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReportGenerator:
    def __init__(self, output_dir: str = "outputs/evaluation_reports"):
        self.output_dir = ensure_dir(output_dir)

    def save(self, metrics: Dict, experiment_name: str = "eval_run") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{experiment_name}_{timestamp}"
        report_path = self.output_dir / f"{base_name}.json"
        write_json(metrics, report_path)
        logger.info("Evaluation report (JSON) saved: %s", report_path)

        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, type(None)))}
        if scalar_metrics:
            csv_path = self.output_dir / f"{base_name}.csv"
            write_dicts_to_csv([scalar_metrics], csv_path)

        if metrics.get("confusion_matrix") is not None:
            self._save_confusion_matrix_plot(metrics["confusion_matrix"], report_path.with_suffix(".png"))

        return report_path

    def save_training_curves(
        self, history: List[Dict], experiment_name: str = "training_run"
    ) -> Optional[Path]:
        """Plot loss/accuracy learning curves from a list of per-epoch
        metric dicts (e.g. [{"epoch": 1, "train_loss": .., "val_loss": ..}, ...])
        and save both the plot and the raw history as CSV/JSON."""
        if not history:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{experiment_name}_curves_{timestamp}"

        write_json(history, self.output_dir / f"{base_name}.json")
        write_dicts_to_csv(history, self.output_dir / f"{base_name}.csv")

        try:
            import matplotlib.pyplot as plt

            epochs = [row.get("epoch", i) for i, row in enumerate(history)]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            for key in ("train_loss", "val_loss"):
                if any(key in row for row in history):
                    axes[0].plot(epochs, [row.get(key) for row in history], label=key)
            axes[0].set_title("Loss Curves")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()

            for key in ("val_accuracy", "val_f1_score"):
                if any(key in row for row in history):
                    axes[1].plot(epochs, [row.get(key) for row in history], label=key)
            axes[1].set_title("Learning Curves")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Score")
            axes[1].legend()

            fig.tight_layout()
            plot_path = self.output_dir / f"{base_name}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info("Training curves plot saved: %s", plot_path)
            return plot_path
        except ImportError:
            logger.warning("matplotlib not installed; skipping training-curve plot.")
            return None

    def _save_confusion_matrix_plot(self, matrix, output_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(np.array(matrix), cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
            logger.info("Confusion matrix plot saved: %s", output_path)
        except ImportError:
            logger.warning("matplotlib not installed; skipping confusion matrix plot.")
