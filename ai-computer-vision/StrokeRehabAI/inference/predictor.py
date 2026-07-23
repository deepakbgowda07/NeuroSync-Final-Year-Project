"""
predictor.py
============
Loads a trained (or, currently, untrained/scaffold) model checkpoint
and exposes a simple predict() interface consumed by the real-time
inference pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.model_factory import build_model, data_representation_for
from utils.checkpoint_utils import load_model_weights
from utils.gpu_utils import get_device
from utils.logger import get_logger

logger = get_logger(__name__)


class RehabPredictor:
    """Wraps a trained model for inference on windowed pose sequences.

    Transparently reshapes each window to whatever tensor layout the
    configured architecture expects: (T, C*V) for the LSTM baseline, or
    (C, T, V) for the default ST-GCN model (see
    models.model_factory.data_representation_for).
    """

    def __init__(self, model_cfg, gpu_cfg, checkpoint_path: Optional[str] = None):
        self.device = get_device(
            use_cuda_if_available=gpu_cfg.use_cuda_if_available,
            device_index=gpu_cfg.device_index,
            fallback_to_cpu=gpu_cfg.fallback_to_cpu,
        )
        self.model_cfg = model_cfg
        self.data_representation = data_representation_for(model_cfg)
        self.model = build_model(model_cfg).to(self.device)
        self.model.eval()
        self.checkpoint_loaded = False

        if checkpoint_path:
            try:
                load_model_weights(self.model, checkpoint_path, map_location=str(self.device))
                self.checkpoint_loaded = True
            except FileNotFoundError:
                logger.warning(
                    "No checkpoint found at %s — running with randomly initialized "
                    "weights. Predictions are NOT clinically meaningful until a "
                    "trained checkpoint is provided (see training/trainer.py).",
                    checkpoint_path,
                )

    def _reshape_window(self, window: np.ndarray) -> np.ndarray:
        """Reshape a flat (T, num_joints*in_channels) window into whatever
        layout the active architecture expects."""
        if self.data_representation == "flat":
            return window  # already (T, feature_dim), what RehabLSTM expects

        # "graph": ST-GCN expects (C, T, V). `window` arrives as
        # (T, num_joints * in_channels); un-flatten then permute.
        num_joints = self.model_cfg.stgcn.num_joints
        in_channels = self.model_cfg.stgcn.in_channels
        t = window.shape[0]
        reshaped = window.reshape(t, num_joints, in_channels)  # (T, V, C)
        return reshaped.transpose(2, 0, 1)  # (C, T, V)

    def predict(self, window: np.ndarray) -> dict:
        """Run inference on a single windowed pose sequence.

        `window` is expected as a flat (sequence_length, num_joints *
        in_channels) array (what preprocessing.sequence_builder produces);
        it is reshaped internally to match the active architecture.

        Returns a dict with predicted class index, per-class probabilities,
        and a `checkpoint_loaded` flag so callers can distinguish real
        predictions from scaffold/random-weight output.
        """
        import torch

        model_input = self._reshape_window(window)

        with torch.no_grad():
            x = torch.tensor(model_input, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.model.predict_proba(x).squeeze(0).cpu().numpy()

        predicted_class = int(np.argmax(probs))
        return {
            "predicted_class": predicted_class,
            "class_probabilities": probs.tolist(),
            "checkpoint_loaded": self.checkpoint_loaded,
        }
