"""
download_dataset.py
====================
Guides the user through obtaining each supported dataset. None of the
following datasets are auto-downloaded, because each has its own
license / registration requirements:

    UI-PRMD              -> https://www.webpages.uidaho.edu/ui-prmd/
    KIMORE                -> https://vrai.dii.univpm.it/content/kimore-dataset
    IntelliRehabDS         -> https://zenodo.org/records/4610859
    Stroke Rehab (Mendeley/NIAID) -> https://data.niaid.nih.gov/resources?id=mendeley_ygpdzx52g2

This script instead prints instructions and expected local directory
layout, matching `configs/datasets.yaml`.

Run: python -m datasets.download_dataset --dataset UI-PRMD
"""

from __future__ import annotations

import argparse

from configs.config_loader import load_config
from utils.file_io import ensure_dir, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)


def print_instructions(dataset_name: str) -> None:
    cfg = load_config()
    sources = cfg.datasets.sources

    if dataset_name not in sources:
        logger.error("Unknown dataset '%s'. Options: %s", dataset_name, list(sources.keys()))
        return

    entry = sources[dataset_name]
    local_path = resolve_path(entry["local_path"])
    ensure_dir(local_path)

    print(f"\n=== {dataset_name} ===")
    print(f"Source URL     : {entry['url']}")
    print(f"License note   : {entry['license_note']}")
    print(f"Expected local path: {local_path}")
    print(
        "\nThis dataset cannot be downloaded automatically due to its "
        "license/registration terms. Please download it manually from "
        "the URL above and place the extracted contents at the path "
        "shown, then run:\n"
        f"    python -m datasets.dataset_checker --dataset {dataset_name}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset acquisition instructions.")
    parser.add_argument(
        "--dataset",
        default="UI-PRMD",
        choices=["UI-PRMD", "KIMORE", "IntelliRehabDS", "StrokeRehab_Mendeley"],
    )
    args = parser.parse_args()
    print_instructions(args.dataset)


if __name__ == "__main__":
    main()
