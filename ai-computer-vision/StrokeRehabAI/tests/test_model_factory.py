"""Tests for models.model_factory. Skipped automatically if torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from configs.config_loader import load_config
from models.model_factory import build_model, data_representation_for


def test_build_stgcn_default_architecture_produces_correct_output_shape():
    cfg = load_config(force_reload=True)
    assert cfg.model.architecture == "stgcn"
    model = build_model(cfg.model)

    batch = torch.randn(2, cfg.model.stgcn.in_channels, 20, cfg.model.stgcn.num_joints)
    output = model(batch)
    assert output.shape == (2, cfg.model.num_classes)


def test_stgcn_model_summary_reports_architecture():
    cfg = load_config(force_reload=True)
    model = build_model(cfg.model)
    summary = model.model_summary()
    assert summary["architecture"] == "STGCN"
    assert summary["num_joints"] == cfg.model.stgcn.num_joints


def test_data_representation_for_stgcn_is_graph():
    cfg = load_config(force_reload=True)
    assert data_representation_for(cfg.model) == "graph"


def test_build_lstm_fallback_architecture():
    cfg = load_config(force_reload=True)
    cfg.model.architecture = "lstm"
    model = build_model(cfg.model)

    batch = torch.randn(2, 10, cfg.model.lstm.input_dim)
    output = model(batch)
    assert output.shape == (2, cfg.model.num_classes)
    assert data_representation_for(cfg.model) == "flat"


def test_unknown_architecture_raises():
    cfg = load_config(force_reload=True)
    cfg.model.architecture = "not_a_real_architecture"
    with pytest.raises(ValueError):
        build_model(cfg.model)
