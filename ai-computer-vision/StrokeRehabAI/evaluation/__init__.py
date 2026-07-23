"""Evaluation package: offline model evaluation, metrics reports, and confusion matrices."""

from .evaluator import ModelEvaluator
from .report_generator import EvaluationReportGenerator

__all__ = ["ModelEvaluator", "EvaluationReportGenerator"]
