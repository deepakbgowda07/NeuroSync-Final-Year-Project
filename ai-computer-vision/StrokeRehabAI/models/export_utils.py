"""
export_utils.py
==================
Model optimization and export utilities targeting the RTX 3050 Laptop
GPU (6GB VRAM): TorchScript export, ONNX export, dynamic quantization
(CPU inference path), and GPU memory optimization helpers. Complements
`training/gpu_monitor.py`'s `InferenceBenchmark` (already implements
FPS/latency benchmarking) and `configs/model.yaml`'s existing
`onnx_export` section.

Run via:
    python -m models.export_utils --checkpoint weights/checkpoints/best.pt --format torchscript
    python -m models.export_utils --checkpoint weights/checkpoints/best.pt --format onnx
    python -m models.export_utils --checkpoint weights/checkpoints/best.pt --format quantized
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def _sample_input_for(model_cfg, batch_size: int = 1):
    """Builds a representative dummy input tensor matching the active
    architecture's expected shape (see models.model_factory.data_representation_for)."""
    import torch

    from models.model_factory import data_representation_for

    representation = data_representation_for(model_cfg)
    if representation == "graph":
        return torch.randn(batch_size, model_cfg.stgcn.in_channels, 90, model_cfg.stgcn.num_joints)
    return torch.randn(batch_size, 90, model_cfg.lstm.input_dim)


def export_torchscript(model, model_cfg, output_path: str) -> Path:
    """Export a trained model to TorchScript (via tracing) for
    dependency-light deployment (no Python/model-definition code needed
    to run inference — see docs/deployment_guide.md)."""
    import torch

    model.eval()
    sample_input = _sample_input_for(model_cfg)

    with torch.no_grad():
        traced = torch.jit.trace(model, sample_input)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    traced.save(str(output_path))
    logger.info("Exported TorchScript model to %s", output_path)
    return output_path


def export_onnx(model, model_cfg, output_path: str, opset_version: int = 17) -> Path:
    """Export a trained model to ONNX, per configs/model.yaml's
    model.onnx_export settings — enables ONNX Runtime (CPU or
    CUDA-EP) inference independent of PyTorch."""
    import torch

    model.eval()
    sample_input = _sample_input_for(model_cfg)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    torch.onnx.export(
        model, sample_input, str(output_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset_version,
        dynamo=False,  # legacy TorchScript-based exporter; avoids requiring the optional onnxscript package
    )
    logger.info("Exported ONNX model to %s (opset %d)", output_path, opset_version)
    return output_path


def export_quantized(model, model_cfg, output_path: str) -> Path:
    """Dynamic quantization (int8 weights) for CPU inference — reduces
    model size and CPU inference latency; not applicable to CUDA
    inference (where mixed precision, not quantization, is the RTX
    3050's optimization path — see training/trainer.py's
    `mixed_precision` support).
    """
    import torch
    from torch.quantization import quantize_dynamic

    model.eval().cpu()
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    torch.save(quantized_model.state_dict(), output_path)
    logger.info("Exported dynamically-quantized model (state_dict) to %s", output_path)
    return output_path


def optimize_gpu_memory(device_index: int = 0) -> None:
    """Frees cached (but unused) CUDA memory and logs current
    allocation — call between training/inference phases or after
    releasing a large batch, to keep the RTX 3050's 6GB budget healthy."""
    import torch

    if not torch.cuda.is_available():
        logger.info("CUDA not available; skipping GPU memory optimization.")
        return

    torch.cuda.empty_cache()
    allocated_mb = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
    logger.info("GPU memory after cache clear: allocated=%.1fMB, reserved=%.1fMB", allocated_mb, reserved_mb)


def main() -> None:
    from configs.config_loader import load_config
    from models.model_factory import build_model
    from utils.checkpoint_utils import load_model_weights

    parser = argparse.ArgumentParser(description="Export a trained StrokeRehabAI model for deployment.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint (.pt)")
    parser.add_argument("--format", required=True, choices=["torchscript", "onnx", "quantized"])
    parser.add_argument("--output", default=None, help="Output path (defaults to weights/exported/<format>.*)")
    args = parser.parse_args()

    cfg = load_config()
    model = build_model(cfg.model)
    load_model_weights(model, args.checkpoint, map_location="cpu")

    defaults = {
        "torchscript": ("weights/exported/model.torchscript.pt", export_torchscript),
        "onnx": (cfg.model.onnx_export.output_path, export_onnx),
        "quantized": ("weights/exported/model.quantized.pt", export_quantized),
    }
    default_path, export_fn = defaults[args.format]
    output_path = args.output or default_path

    export_fn(model, cfg.model, output_path)


if __name__ == "__main__":
    main()
