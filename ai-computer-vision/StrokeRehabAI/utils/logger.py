"""
logger.py
=========
Centralized logging setup using the standard `logging` module plus
`rich` for readable console output and pretty tracebacks.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Camera initialized")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.traceback import install as install_rich_traceback

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is a required dependency but degrade gracefully
    _RICH_AVAILABLE = False

_CONFIGURED = False
_LOG_DIR = Path("logs")


def _build_log_filename(pattern: str = "strokerehab_{timestamp}.log") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return pattern.format(timestamp=timestamp)


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    error_log_filename: str = "errors.log",
    rotate_max_bytes: int = 5 * 1024 * 1024,
    rotate_backup_count: int = 5,
    rich_traceback: bool = True,
) -> None:
    """Configure the root logger once for the whole application.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _CONFIGURED, _LOG_DIR
    if _CONFIGURED:
        return

    _LOG_DIR = Path(log_dir) if log_dir else _LOG_DIR
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        if _RICH_AVAILABLE:
            console_handler = RichHandler(rich_tracebacks=rich_traceback, show_path=False)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if file_output:
        log_path = _LOG_DIR / _build_log_filename()
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=rotate_max_bytes, backupCount=rotate_backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        error_handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / error_log_filename,
            maxBytes=rotate_max_bytes,
            backupCount=rotate_backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)

    if _RICH_AVAILABLE and rich_traceback:
        install_rich_traceback(show_locals=False)

    _CONFIGURED = True
    root_logger.debug("Logging configured (level=%s, dir=%s)", level, _LOG_DIR)


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger, configuring global logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def log_gpu_info(logger: Optional[logging.Logger] = None) -> None:
    """Log GPU/CUDA availability info. Kept separate to avoid importing
    torch at module import time (torch import is comparatively slow)."""
    logger = logger or get_logger(__name__)
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            logger.info(
                "CUDA available | GPU: %s | VRAM: %.1f GB | CUDA: %s | Torch: %s",
                name,
                vram_gb,
                torch.version.cuda,
                torch.__version__,
            )
        else:
            logger.warning("CUDA not available. Falling back to CPU. Torch: %s", torch.__version__)
    except ImportError:
        logger.warning("PyTorch not installed; cannot report GPU info.")
