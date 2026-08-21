"""Tests for models.export_utils (TorchScript/ONNX/quantization export)."""

import pytest

torch = pytest.importorskip("torch")

from configs.config_loader import load_config
from models.export_utils import export_onnx, export_quantized, export_torchscript, optimize_gpu_memory
from models.model_factory import build_model


@pytest.fixture
def tiny_model_and_cfg():
    cfg = load_config(force_reload=True)
    cfg.model.stgcn.channels = [8, 8]
    cfg.model.stgcn.strides = [1, 1]
    model = build_model(cfg.model)
    model.eval()
    return model, cfg.model


def test_export_torchscript_creates_loadable_file(tiny_model_and_cfg, tmp_path):
    model, model_cfg = tiny_model_and_cfg
    output_path = export_torchscript(model, model_cfg, str(tmp_path / "model.ts.pt"))
    assert output_path.exists()

    loaded = torch.jit.load(str(output_path))
    sample_input = torch.randn(1, model_cfg.stgcn.in_channels, 90, model_cfg.stgcn.num_joints)
    output = loaded(sample_input)
    assert output.shape == (1, model_cfg.num_classes)


def test_export_onnx_creates_valid_file(tiny_model_and_cfg, tmp_path):
    onnx = pytest.importorskip("onnx")
    model, model_cfg = tiny_model_and_cfg
    output_path = export_onnx(model, model_cfg, str(tmp_path / "model.onnx"))
    assert output_path.exists()
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def test_onnx_output_matches_pytorch_output(tiny_model_and_cfg, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model, model_cfg = tiny_model_and_cfg
    output_path = export_onnx(model, model_cfg, str(tmp_path / "model.onnx"))

    sample_input = torch.randn(1, model_cfg.stgcn.in_channels, 90, model_cfg.stgcn.num_joints)
    with torch.no_grad():
        torch_output = model(sample_input).numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": sample_input.numpy()})[0]

    import numpy as np
    np.testing.assert_allclose(torch_output, onnx_output, atol=1e-3, rtol=1e-3)


def test_export_quantized_creates_state_dict(tiny_model_and_cfg, tmp_path):
    model, model_cfg = tiny_model_and_cfg
    output_path = export_quantized(model, model_cfg, str(tmp_path / "model.q.pt"))
    assert output_path.exists()
    state_dict = torch.load(output_path, weights_only=False)
    assert len(state_dict) > 0


def test_optimize_gpu_memory_does_not_raise_without_cuda():
    optimize_gpu_memory()  # should log and return cleanly on CPU-only sandbox
