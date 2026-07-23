"""
gpu_utils.py
============
GPU / CUDA detection and device management helpers.

Automatically detects CUDA availability and gracefully falls back to
CPU if no compatible GPU is found. Tuned defaults in configs/gpu.yaml
assume an NVIDIA RTX 3050 Laptop GPU (6GB VRAM), but this module works
on any CUDA-capable device or CPU-only machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    available: bool
    device_name: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    total_vram_gb: Optional[float] = None
    device_index: int = 0


def get_gpu_info(device_index: int = 0) -> GPUInfo:
    """Inspect the current machine and return structured GPU information."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed; reporting no GPU available.")
        return GPUInfo(available=False)

    if not torch.cuda.is_available():
        return GPUInfo(available=False, torch_version=torch.__version__)

    props = torch.cuda.get_device_properties(device_index)
    return GPUInfo(
        available=True,
        device_name=torch.cuda.get_device_name(device_index),
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
        total_vram_gb=round(props.total_memory / (1024 ** 3), 2),
        device_index=device_index,
    )


def get_device(use_cuda_if_available: bool = True, device_index: int = 0, fallback_to_cpu: bool = True):
    """Return a `torch.device`, preferring CUDA when available.

    TODO: extend to support Apple MPS backend for cross-platform dev machines.
    """
    import torch

    if use_cuda_if_available and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(device_index))
        return device

    if not fallback_to_cpu:
        raise RuntimeError("CUDA unavailable and fallback_to_cpu=False.")

    logger.warning("CUDA unavailable, falling back to CPU.")
    return torch.device("cpu")


def apply_gpu_settings(allow_tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    """Apply global torch/cuDNN performance flags. Call once at startup."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        logger.debug(
            "Applied GPU settings: allow_tf32=%s, cudnn_benchmark=%s", allow_tf32, cudnn_benchmark
        )


def estimate_safe_batch_size(vram_gb: Optional[float], default: int = 16) -> int:
    """Rough heuristic to suggest a batch size given available VRAM.

    This is intentionally conservative and meant as a starting point,
    not a substitute for empirical tuning.

    TODO: replace with a proper memory-profiling based estimator once
    the model architecture is finalized.
    """
    if vram_gb is None:
        return default
    if vram_gb <= 4:
        return 8
    if vram_gb <= 6:
        return 16
    if vram_gb <= 8:
        return 24
    return 32
