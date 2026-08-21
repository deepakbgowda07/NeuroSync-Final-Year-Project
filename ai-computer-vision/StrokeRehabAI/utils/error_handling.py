"""
error_handling.py
====================
Centralized error handling: custom exception types for the failure
modes the system must handle gracefully (camera disconnection, CUDA
unavailable, missing datasets, missing model weights, corrupted files,
invalid configurations, low memory, unexpected runtime exceptions),
plus helpers that turn a raw exception into a clear, user-facing
message while still logging full details for debugging.

Every custom exception carries both a short `user_message` (safe to
show a clinician/patient in the dashboard or console) and the original
technical detail (logged, not surfaced) — see `handle_gracefully()`.
"""

from __future__ import annotations

import functools
import traceback
from typing import Callable, Optional, TypeVar

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class StrokeRehabAIError(Exception):
    """Base class for all custom StrokeRehabAI exceptions.

    Args:
        user_message: a clear, non-technical message safe to show the
            person using the system.
        detail: technical detail for logs only (defaults to str(cause)).
        cause: the original exception, if any, for full traceback logging.
    """

    def __init__(self, user_message: str, detail: Optional[str] = None, cause: Optional[BaseException] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail or (str(cause) if cause else "")
        self.cause = cause


class CameraError(StrokeRehabAIError):
    """Camera disconnected, unavailable, or failed to open."""


class CUDAUnavailableError(StrokeRehabAIError):
    """CUDA was required but is not available (informational; the
    system should fall back to CPU rather than raise this in most
    paths — see utils/gpu_utils.py's fallback_to_cpu behavior. Raised
    only where CPU fallback isn't viable.)"""


class DatasetError(StrokeRehabAIError):
    """A required dataset directory/file is missing or malformed."""


class ModelWeightsError(StrokeRehabAIError):
    """A required model checkpoint is missing or failed to load."""


class CorruptedFileError(StrokeRehabAIError):
    """A file (video, config, checkpoint, database) is present but unreadable/corrupted."""


class InvalidConfigurationError(StrokeRehabAIError):
    """A configuration file is missing required fields or has invalid values."""


class LowMemoryError(StrokeRehabAIError):
    """The system detected insufficient GPU or system memory to proceed safely."""


def format_user_facing_message(exc: BaseException) -> str:
    """Return a clear, non-technical message for any exception —
    StrokeRehabAIError subclasses already carry one; other exceptions
    get a generic-but-honest fallback message."""
    if isinstance(exc, StrokeRehabAIError):
        return exc.user_message

    fallback_messages = {
        FileNotFoundError: "A required file could not be found. Check the path in your configuration.",
        PermissionError: "Permission was denied accessing a required file or device.",
        MemoryError: "The system ran out of memory. Try closing other applications or reducing batch size.",
        ConnectionError: "A network or device connection failed.",
    }
    for exc_type, message in fallback_messages.items():
        if isinstance(exc, exc_type):
            return message

    return "An unexpected error occurred. See the log file for technical details."


def log_full_detail(exc: BaseException, context: str = "") -> None:
    """Logs the full technical traceback (for developers), while
    `format_user_facing_message` is what gets shown to the person using
    the system."""
    prefix = f"[{context}] " if context else ""
    logger.error("%sUnhandled exception: %s\n%s", prefix, exc, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def handle_gracefully(
    default_return: T = None,
    context: str = "",
    reraise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: catches any exception raised by the wrapped function,
    logs full technical detail, and either returns `default_return` or
    re-raises as a StrokeRehabAIError with a clear user-facing message.

    Usage:
        @handle_gracefully(default_return=None, context="camera capture")
        def read_frame(...):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except StrokeRehabAIError:
                raise  # already has a clear user_message; let it propagate as-is
            except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
                log_full_detail(exc, context=context or func.__name__)
                if reraise:
                    raise StrokeRehabAIError(format_user_facing_message(exc), cause=exc) from exc
                return default_return

        return wrapper

    return decorator


def check_available_memory_mb(min_required_mb: float = 500.0, device_index: int = 0) -> None:
    """Raises LowMemoryError if free GPU memory (when CUDA is available)
    falls below `min_required_mb` — call before starting a memory-heavy
    operation (e.g. training, batch inference) to fail with a clear
    message rather than an opaque CUDA OOM traceback."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
        free_mb = free_bytes / (1024 ** 2)
        if free_mb < min_required_mb:
            raise LowMemoryError(
                f"Low GPU memory: only {free_mb:.0f}MB free (need at least {min_required_mb:.0f}MB). "
                "Close other GPU applications or reduce batch size.",
                detail=f"free_mb={free_mb:.1f}",
            )
    except ImportError:
        return
