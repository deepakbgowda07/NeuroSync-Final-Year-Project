"""
file_io.py
==========
Filesystem helpers: safe directory creation, path resolution, and
generic read/write wrappers used across the dataset and reporting
pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from utils.logger import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the StrokeRehabAI project root (two levels above this file)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(relative_path: PathLike) -> Path:
    """Resolve a path relative to the project root, if not already absolute."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return project_root() / p


def file_size_mb(path: PathLike) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


def list_files(directory: PathLike, extensions: tuple = (".csv", ".json", ".npy")) -> list:
    """List files in a directory filtered by extension."""
    p = Path(directory)
    if not p.exists():
        logger.warning("Directory does not exist: %s", p)
        return []
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in extensions and f.is_file())
