"""
gpu_monitor.py
================
Lightweight GPU utilization and inference-speed monitoring used during
training (periodic logging) and evaluation (FPS / inference-time
benchmarking, per configs/evaluation.yaml's `fps` / `inference_time`
metrics).

GPU utilization is read via `nvidia-smi` when available (works
regardless of which ML framework is active) with a `torch.cuda` memory
fallback if `nvidia-smi` isn't on PATH (e.g. inside some containers).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GPUUtilizationSnapshot:
    gpu_utilization_percent: Optional[float]
    memory_used_mb: Optional[float]
    memory_total_mb: Optional[float]
    source: str  # "nvidia-smi" | "torch" | "unavailable"


def read_gpu_utilization(device_index: int = 0) -> GPUUtilizationSnapshot:
    """Best-effort GPU utilization snapshot. Prefers `nvidia-smi` (gives
    true compute utilization); falls back to torch's memory-only view."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        util_str, used_str, total_str = [s.strip() for s in output.split(",")]
        return GPUUtilizationSnapshot(
            gpu_utilization_percent=float(util_str),
            memory_used_mb=float(used_str),
            memory_total_mb=float(total_str),
            source="nvidia-smi",
        )
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
        pass

    try:
        import torch

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
            total = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 2)
            return GPUUtilizationSnapshot(
                gpu_utilization_percent=None,  # torch doesn't expose compute utilization directly
                memory_used_mb=used, memory_total_mb=total, source="torch",
            )
    except ImportError:
        pass

    return GPUUtilizationSnapshot(None, None, None, source="unavailable")


class InferenceBenchmark:
    """Measures average inference latency and throughput (FPS) for a
    model over a representative batch of inputs — used by
    evaluation/evaluator.py to report the `fps` / `inference_time`
    metrics from configs/evaluation.yaml."""

    def __init__(self, model, device, warmup_iters: int = 5, measured_iters: int = 30):
        self.model = model
        self.device = device
        self.warmup_iters = warmup_iters
        self.measured_iters = measured_iters

    def run(self, sample_input) -> dict:
        import torch

        self.model.eval()
        sample_input = sample_input.to(self.device)

        with torch.no_grad():
            for _ in range(self.warmup_iters):
                self.model(sample_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.measured_iters):
                self.model(sample_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        mean_latency_ms = (elapsed / self.measured_iters) * 1000
        fps = self.measured_iters / elapsed if elapsed > 0 else 0.0

        logger.info("Inference benchmark: %.2f ms/batch, %.1f batches/sec", mean_latency_ms, fps)
        return {"mean_latency_ms": mean_latency_ms, "fps": fps, "measured_iters": self.measured_iters}
