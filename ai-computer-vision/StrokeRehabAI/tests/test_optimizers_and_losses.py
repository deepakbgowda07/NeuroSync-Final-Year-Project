"""Tests for training.optimizer_factory, training.optimizers.ranger, and training.loss_factory."""

import pytest

torch = pytest.importorskip("torch")

from configs.config_loader import load_config
from training.loss_factory import build_loss, compute_class_weights
from training.optimizer_factory import build_optimizer


@pytest.fixture
def tiny_model():
    return torch.nn.Linear(10, 5)


@pytest.mark.parametrize("optimizer_name", ["adamw", "adam", "sgd", "ranger"])
def test_build_optimizer_supports_all_configured_names(tiny_model, optimizer_name):
    cfg = load_config(force_reload=True)
    cfg.training.optimizer = optimizer_name
    optimizer = build_optimizer(tiny_model, cfg.training)

    x = torch.randn(4, 10)
    loss = tiny_model(x).sum()
    loss.backward()
    optimizer.step()  # should not raise
    optimizer.zero_grad()


def test_build_optimizer_unknown_name_raises(tiny_model):
    cfg = load_config(force_reload=True)
    cfg.training.optimizer = "not_a_real_optimizer"
    with pytest.raises(ValueError):
        build_optimizer(tiny_model, cfg.training)


@pytest.mark.parametrize("loss_name,expected_type", [
    ("cross_entropy", torch.nn.CrossEntropyLoss),
    ("mse", torch.nn.MSELoss),
    ("huber", torch.nn.HuberLoss),
])
def test_build_loss_returns_correct_type(loss_name, expected_type):
    cfg = load_config(force_reload=True)
    cfg.training.loss = loss_name
    loss_fn = build_loss(cfg.model, cfg.training)
    assert isinstance(loss_fn, expected_type)


def test_compute_class_weights_inverse_frequency():
    labels = [0, 0, 0, 0, 1]  # class 0 is 4x more frequent than class 1
    weights = compute_class_weights(labels, num_classes=2)
    assert weights[1] > weights[0]
