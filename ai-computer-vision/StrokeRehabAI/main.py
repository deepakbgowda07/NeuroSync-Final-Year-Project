"""
main.py
=======
Top-level convenience CLI for the three main entry points:

    python main.py train        # runs training/trainer.py
    python main.py infer        # runs inference/realtime_pipeline.py (live camera)
    python main.py dashboard    # prints the streamlit launch command
    python main.py check-gpu    # prints detected GPU/CUDA info
    python main.py check-data   # runs datasets/dataset_checker.py

This is intentionally a thin dispatcher — each subcommand's real logic
lives in its respective package.
"""

from __future__ import annotations

import argparse
import sys

from utils.logger import configure_logging, get_logger, log_gpu_info

logger = get_logger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    from training.trainer import Trainer

    trainer = Trainer()
    trainer.fit()


def cmd_infer(args: argparse.Namespace) -> None:
    from inference.realtime_pipeline import RealtimeInferencePipeline

    pipeline = RealtimeInferencePipeline(checkpoint_path=args.checkpoint)
    pipeline.run()


def cmd_dashboard(args: argparse.Namespace) -> None:
    print("Streamlit apps must be launched via the `streamlit` CLI, not `python`:\n")
    print("    streamlit run dashboard/app.py\n")


def cmd_check_gpu(args: argparse.Namespace) -> None:
    log_gpu_info()


def cmd_check_data(args: argparse.Namespace) -> None:
    from datasets.dataset_checker import DatasetChecker

    DatasetChecker().check_all()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StrokeRehabAI project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run the training pipeline.")

    infer_parser = subparsers.add_parser("infer", help="Run the real-time inference pipeline.")
    infer_parser.add_argument("--checkpoint", default=None, help="Path to a trained model checkpoint (.pt)")

    subparsers.add_parser("dashboard", help="Show how to launch the Streamlit dashboard.")
    subparsers.add_parser("check-gpu", help="Print detected GPU/CUDA information.")
    subparsers.add_parser("check-data", help="Check configured dataset availability.")

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "infer": cmd_infer,
        "dashboard": cmd_dashboard,
        "check-gpu": cmd_check_gpu,
        "check-data": cmd_check_data,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
