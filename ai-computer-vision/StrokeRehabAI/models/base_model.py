"""
base_model.py
=============
Abstract base class all StrokeRehabAI models implement, so training,
inference, and ONNX export code can work with any concrete
architecture (LSTM today; TCN/Transformer variants planned).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict


class BaseRehabModel:
    """Mixin-style base defining the contract concrete models must satisfy.

    NOTE: intentionally does not subclass torch.nn.Module directly in
    this stub so the file can be imported without torch installed
    (useful for static analysis / docs generation). Concrete models
    (e.g. RehabLSTM) subclass both this and nn.Module.
    """

    @abstractmethod
    def forward(self, x):
        """Forward pass: x is (batch, sequence_length, input_dim)."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x):
        """Return class probabilities for a batch of input sequences."""
        raise NotImplementedError

    def model_summary(self) -> Dict:
        """Return a small dict describing the model for logging/reports.
        Concrete subclasses should override with real hyperparameters."""
        return {"architecture": self.__class__.__name__}
