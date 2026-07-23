"""
lstm_model.py
=============
Baseline (Bi)LSTM classifier for exercise correctness / severity
assessment from a windowed sequence of pose features.

This is the reference architecture named in configs/model.yaml
("rehab_lstm_v1"). It is intentionally a solid, unglamorous baseline —
the actual clinical validation and hyperparameter tuning is left as
the next development phase, per project scope.

TODO (next development phase):
- Add attention pooling over the LSTM outputs instead of last-hidden-state.
- Add a multi-task head (correctness classification + ROM regression).
- Benchmark against a Temporal Convolutional Network (TCN) baseline.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - allows import-time introspection without torch
    torch = None
    nn = object  # type: ignore

from models.base_model import BaseRehabModel


class RehabLSTM(BaseRehabModel, nn.Module if torch is not None else object):
    """Bidirectional LSTM classifier over pose-landmark sequences."""

    def __init__(
        self,
        input_dim: int = 99,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to instantiate RehabLSTM.")

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """x: (batch, sequence_length, input_dim) -> logits: (batch, num_classes)"""
        lstm_out, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # Concatenate final forward + backward hidden states of the last layer.
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]

        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def model_summary(self):
        return {
            "architecture": "RehabLSTM",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters()) if torch is not None else None,
        }
