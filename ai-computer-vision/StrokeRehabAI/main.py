"""
main.py
=======
Top-level convenience CLI for StrokeRehabAI's main entry points:

    python main.py train                       # runs training/trainer.py
    python main.py infer                        # runs the real-time inference engine (live camera)
    python main.py infer --patient-id 3          # associate the session with a patient
    python main.py infer --skip-calibration       # skip the pre-exercise calibration step
    python main.py dashboard                    # prints the streamlit launch command
    python main.py check-gpu                     # prints detected GPU/CUDA info
    python main.py check-data                    # runs datasets/dataset_checker.py
    python main.py demo --mode webcam             # demonstration mode: live webcam
    python main.py demo --mode video --source path/to/video.mp4
    python main.py demo --mode session --source 1  # replay a stored session by ID
    python main.py demo --generate-data            # populate the dashboard with synthetic demo data
    python main.py validate                       # run final integration/environment validation checks

This is intentionally a thin dispatcher — each subcommand's real logic
lives in its respective package.
"""

from __future__ import annotations

import argparse
import sys

# Mask tensorflow to prevent import errors with incompatible protobuf versions in the environment
sys.modules["tensorflow"] = None

from utils.logger import configure_logging, get_logger, log_gpu_info

logger = get_logger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    from training.trainer import Trainer

    trainer = Trainer()
    trainer.fit()


def cmd_infer(args: argparse.Namespace) -> None:
    from inference.realtime_pipeline import RealtimeInferencePipeline

    pipeline = RealtimeInferencePipeline(patient_id=args.patient_id, skip_calibration=args.skip_calibration)
    pipeline.run()


def cmd_dashboard(args: argparse.Namespace) -> None:
    print("Streamlit apps must be launched via the `streamlit` CLI, not `python`:\n")
    print("    streamlit run dashboard/app.py\n")


def cmd_check_gpu(args: argparse.Namespace) -> None:
    log_gpu_info()


def cmd_check_data(args: argparse.Namespace) -> None:
    from datasets.dataset_checker import DatasetChecker

    DatasetChecker().check_all()


def cmd_demo(args: argparse.Namespace) -> None:
    from configs.config_loader import load_config
    from inference.demo_mode import DemoModeRunner, DemoSourceMode, generate_demo_dataset

    cfg = load_config()

    if args.generate_data:
        patient_id = generate_demo_dataset(cfg.dashboard.database_path, num_sessions=args.num_sessions)
        print(f"Demo dataset generated: patient_id={patient_id}. Open the dashboard's Recovery Analytics "
              f"or Reports page and select the new demo patient to explore it.")
        return

    runner = DemoModeRunner(cfg=cfg)
    runner.run(DemoSourceMode(args.mode), source_path=args.source, skip_calibration=args.skip_calibration)


def cmd_validate(args: argparse.Namespace) -> None:
    from scripts.validate_project import run_validation

    success = run_validation()
    sys.exit(0 if success else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StrokeRehabAI project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run the training pipeline.")

    infer_parser = subparsers.add_parser("infer", help="Run the real-time inference engine.")
    infer_parser.add_argument("--patient-id", type=int, default=None, help="Patient ID to associate this session with.")
    infer_parser.add_argument("--skip-calibration", action="store_true", help="Skip the pre-exercise calibration step.")

    subparsers.add_parser("dashboard", help="Show how to launch the Streamlit dashboard.")
    subparsers.add_parser("check-gpu", help="Print detected GPU/CUDA information.")
    subparsers.add_parser("check-data", help="Check configured dataset availability.")

    demo_parser = subparsers.add_parser("demo", help="Run demonstration mode (webcam, sample video, or stored session replay).")
    demo_parser.add_argument("--mode", choices=["webcam", "video", "session"], default="webcam")
    demo_parser.add_argument("--source", default=None, help="Video file path (mode=video) or session ID (mode=session).")
    demo_parser.add_argument("--skip-calibration", action="store_true")
    demo_parser.add_argument("--generate-data", action="store_true", help="Populate the dashboard with a synthetic demo patient and sessions.")
    demo_parser.add_argument("--num-sessions", type=int, default=8, help="Number of synthetic sessions to generate with --generate-data.")

    subparsers.add_parser("validate", help="Run final integration/environment validation checks.")

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
        "demo": cmd_demo,
        "validate": cmd_validate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
