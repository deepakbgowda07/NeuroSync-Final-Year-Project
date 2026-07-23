"""Tests for training.checkpoint_manager."""

import pytest

torch = pytest.importorskip("torch")

from training.checkpoint_manager import CheckpointManager


@pytest.fixture
def tiny_model():
    return torch.nn.Linear(4, 2)


def test_save_last_creates_last_pt(tmp_path, tiny_model):
    manager = CheckpointManager(str(tmp_path), save_top_k=2)
    state = manager.build_state(tiny_model, None, None, epoch=1, metrics={"val_loss": 0.5})
    path = manager.save_last(state)
    assert path.exists()
    assert path.name == "last.pt"


def test_maybe_save_best_keeps_top_k(tmp_path, tiny_model):
    manager = CheckpointManager(str(tmp_path), save_top_k=2, metric_mode="min")
    for epoch, val_loss in enumerate([0.9, 0.5, 0.7, 0.3], start=1):
        state = manager.build_state(tiny_model, None, None, epoch=epoch, metrics={"val_loss": val_loss})
        manager.maybe_save_best(state, epoch, val_loss)

    best_files = [p for p in tmp_path.glob("best_epoch*.pt")]
    assert len(best_files) == 2  # only top-2 (lowest loss) retained


def test_resume_loads_epoch_and_state(tmp_path, tiny_model):
    manager = CheckpointManager(str(tmp_path), save_top_k=1)
    state = manager.build_state(tiny_model, None, None, epoch=5, metrics={"val_loss": 0.2})
    manager.save_last(state)

    checkpoint = manager.load_for_resume()
    assert checkpoint["epoch"] == 5
    assert "model_state_dict" in checkpoint


def test_metric_mode_max_keeps_highest_values(tmp_path, tiny_model):
    manager = CheckpointManager(str(tmp_path), save_top_k=1, metric_mode="max")
    state_low = manager.build_state(tiny_model, None, None, epoch=1, metrics={"val_accuracy": 0.5})
    state_high = manager.build_state(tiny_model, None, None, epoch=2, metrics={"val_accuracy": 0.9})

    manager.maybe_save_best(state_low, 1, 0.5)
    saved = manager.maybe_save_best(state_high, 2, 0.9)
    assert saved is not None  # higher accuracy should replace the lower one
