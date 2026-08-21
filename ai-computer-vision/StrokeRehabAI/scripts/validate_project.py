"""
validate_project.py
======================
Final validation: verifies that every package imports cleanly, every
YAML config file parses and merges without error, key dependencies are
installed, the model can be built, the dashboard database initializes,
and the dashboard-inference integration path (session logging) works
end to end. Run this after installation or before a demo to catch
environment problems early with clear, specific error messages rather
than a confusing failure mid-session.

Run via:
    python main.py validate
    python -m scripts.validate_project
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List

from utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

CORE_PACKAGES = [
    "camera", "mediapipe_pipeline", "preprocessing", "feature_extraction",
    "datasets", "models", "training", "inference", "evaluation",
    "visualization", "dashboard", "utils", "configs",
]

REQUIRED_THIRD_PARTY = [
    "torch", "cv2", "mediapipe", "numpy", "pandas", "yaml", "streamlit",
    "plotly", "sklearn", "matplotlib", "reportlab", "openpyxl",
]


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(ValidationCheck(name, passed, detail))

    def print_summary(self) -> None:
        print("\n=== StrokeRehabAI Final Validation ===\n")
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            icon = "\u2713" if check.passed else "\u2717"
            line = f"[{status}] {icon} {check.name}"
            if check.detail and not check.passed:
                line += f" — {check.detail}"
            try:
                print(line)
            except UnicodeEncodeError:
                icon_ascii = "+" if check.passed else "x"
                line_ascii = f"[{status}] {icon_ascii} {check.name}"
                if check.detail and not check.passed:
                    line_ascii += f" — {check.detail}"
                print(line_ascii)

        passed_count = sum(1 for c in self.checks if c.passed)
        print(f"\n{passed_count}/{len(self.checks)} checks passed.")
        if self.all_passed:
            print("All validation checks passed. The project is ready to run.\n")
        else:
            print("Some checks failed — see details above and the log file for more.\n")


def check_package_imports(report: ValidationReport) -> None:
    for package_name in CORE_PACKAGES:
        try:
            importlib.import_module(package_name)
            report.add(f"import {package_name}", True)
        except Exception as exc:  # noqa: BLE001 - want to catch and report any import failure
            report.add(f"import {package_name}", False, str(exc))


def check_third_party_dependencies(report: ValidationReport) -> None:
    for module_name in REQUIRED_THIRD_PARTY:
        try:
            importlib.import_module(module_name)
            report.add(f"dependency: {module_name}", True)
        except ImportError as exc:
            report.add(f"dependency: {module_name}", False, str(exc))


def check_configuration(report: ValidationReport) -> None:
    try:
        from configs.config_loader import load_config

        cfg = load_config(force_reload=True)
        required_sections = {
            "camera", "training", "datasets", "model", "dashboard", "visualization",
            "evaluation", "gpu", "logging", "augmentation", "features", "exercises", "calibration",
        }
        missing = required_sections - set(cfg.keys())
        if missing:
            report.add("configuration sections present", False, f"Missing: {missing}")
        else:
            report.add("configuration sections present", True)

        report.add(
            "exercise library has 10 exercises", len(cfg.exercises.definitions) == 10,
            f"Found {len(cfg.exercises.definitions)}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("configuration loads", False, str(exc))


def check_knowledge_base(report: ValidationReport) -> None:
    try:
        from inference.knowledge_base import PhysiotherapyKnowledgeBase

        kb = PhysiotherapyKnowledgeBase()
        report.add("knowledge base loads all 10 exercises", len(kb.keys()) == 10, f"Found {len(kb.keys())}")
    except Exception as exc:  # noqa: BLE001
        report.add("knowledge base loads", False, str(exc))


def check_model_builds(report: ValidationReport) -> None:
    try:
        from configs.config_loader import load_config
        from models.model_factory import build_model

        cfg = load_config()
        cfg.model.stgcn.channels = [8, 8]
        cfg.model.stgcn.strides = [1, 1]
        model = build_model(cfg.model)
        report.add("ST-GCN model builds", model is not None)
    except Exception as exc:  # noqa: BLE001
        report.add("ST-GCN model builds", False, str(exc))


def check_gpu_status(report: ValidationReport) -> None:
    try:
        from utils.gpu_utils import get_gpu_info

        info = get_gpu_info()
        report.add(
            "GPU status check ran (CUDA optional)", True,
            f"CUDA available: {info.available}" + (f", GPU: {info.device_name}" if info.available else " (CPU fallback OK)"),
        )
    except Exception as exc:  # noqa: BLE001
        report.add("GPU status check ran", False, str(exc))


def check_dashboard_database_integration(report: ValidationReport) -> None:
    """Verifies the dashboard <-> inference integration path: database
    initializes, a session can be logged (as the real-time engine
    would), and analytics/reports can read it back."""
    try:
        from dashboard.db import init_db, get_connection
        from dashboard.analytics import RecoveryAnalyticsEngine
        from inference.demo_mode import generate_demo_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/validate.db"
            init_db(db_path)
            report.add("dashboard database initializes", True)

            patient_id = generate_demo_dataset(db_path, num_sessions=2)
            report.add("demo session data can be written", patient_id is not None)

            engine = RecoveryAnalyticsEngine(db_path)
            analytics = engine.compute_patient_analytics(patient_id)
            report.add("recovery analytics compute from stored sessions", analytics.num_sessions == 2)

            from dashboard.report_generator import ReportGenerator

            report_gen = ReportGenerator(db_path, output_dir=f"{tmpdir}/reports")
            report_path = report_gen.generate(patient_id, report_format="json")
            report.add("report generation succeeds", report_path.exists())
    except Exception as exc:  # noqa: BLE001
        report.add("dashboard <-> inference integration", False, str(exc))


def check_error_handling(report: ValidationReport) -> None:
    try:
        from utils.error_handling import StrokeRehabAIError, format_user_facing_message

        exc = StrokeRehabAIError("Test message.")
        report.add("error handling module functions", format_user_facing_message(exc) == "Test message.")
    except Exception as exc:  # noqa: BLE001
        report.add("error handling module functions", False, str(exc))


def run_validation() -> bool:
    configure_logging()
    report = ValidationReport()

    check_package_imports(report)
    check_third_party_dependencies(report)
    check_configuration(report)
    check_knowledge_base(report)
    check_model_builds(report)
    check_gpu_status(report)
    check_error_handling(report)
    check_dashboard_database_integration(report)

    report.print_summary()
    return report.all_passed


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
