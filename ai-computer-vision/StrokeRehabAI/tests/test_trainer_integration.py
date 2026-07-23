"""End-to-end integration test for training.trainer.Trainer against a
tiny synthetic dataset — covers dataloader construction, one training
epoch, checkpoint save, validation metrics, resume, and the test loop.
Marked slow; still fast enough (<30s) to run in the default suite with
a deliberately tiny model/config.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _write_synthetic_samples(root, num_samples=16, num_subjects=5):
    rng = np.random.default_rng(0)
    for i in range(num_samples):
        subject = f"subj{i % num_subjects}"
        num_frames = int(rng.integers(70, 100))
        landmarks = rng.uniform(0, 1, size=(num_frames, 33, 3))
        label = int(rng.integers(0, 5))
        metadata = {"sample_id": f"s{i}", "subject_id": subject, "exercise_type": "ex", "source_dataset": "TEST"}
        np.savez_compressed(root / f"s{i}.npz", landmarks=landmarks, label=label, metadata=metadata)


@pytest.fixture
def tiny_cfg(tmp_path):
    from configs.config_loader import load_config

    _write_synthetic_samples(tmp_path)
    cfg = load_config(force_reload=True)
    cfg.datasets.processed_dir = str(tmp_path)
    cfg.datasets.sequence_length = 30
    cfg.datasets.stride = 15
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    cfg.training.epochs = 1
    cfg.training.mixed_precision = False
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.training.tensorboard.enabled = False
    cfg.training.progress_bar = False
    cfg.gpu.use_cuda_if_available = False
    cfg.model.stgcn.channels = [8, 8]
    cfg.model.stgcn.strides = [1, 1]
    return cfg


def test_trainer_fit_and_test_end_to_end(tiny_cfg):
    from training.trainer import Trainer

    trainer = Trainer(tiny_cfg)
    trainer.fit()  # builds dataloaders internally
    assert (tiny_cfg.training.checkpoint_dir and
            (__import__("pathlib").Path(tiny_cfg.training.checkpoint_dir) / "last.pt").exists())

    metrics = trainer.test()
    assert "accuracy" in metrics


def test_trainer_resume_continues_from_saved_epoch(tiny_cfg):
    from training.trainer import Trainer

    tiny_cfg.training.epochs = 2
    trainer = Trainer(tiny_cfg)
    trainer.fit()

    trainer2 = Trainer(tiny_cfg)
    trainer2.fit(resume_path="last")
    assert trainer2.start_epoch == 3  # resumed after the 2 completed epochs
