"""Tests for configs.config_loader."""

from configs.config_loader import ConfigLoader, load_config


def test_load_config_returns_all_sections():
    cfg = load_config(force_reload=True)
    expected_sections = {
        "camera", "training", "datasets", "model",
        "dashboard", "visualization", "evaluation", "gpu", "logging",
    }
    assert expected_sections.issubset(set(cfg.keys()))


def test_dot_access_matches_dict_access():
    cfg = load_config()
    assert cfg.training.batch_size == cfg["training"]["batch_size"]


def test_deep_merge_combines_nested_dicts():
    base = {"a": {"x": 1, "y": 2}}
    incoming = {"a": {"y": 99, "z": 3}}
    merged = ConfigLoader._deep_merge(base, incoming)
    assert merged == {"a": {"x": 1, "y": 99, "z": 3}}
