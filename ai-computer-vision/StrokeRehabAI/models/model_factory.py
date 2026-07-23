"""
model_factory.py
=================
Config-driven model instantiation, so training/inference code never
hardcodes an architecture class. `architecture: "stgcn"` (the default,
see configs/model.yaml) builds the Spatial-Temporal Graph Convolutional
Network; `architecture: "lstm"` builds the flattened-feature-vector
LSTM baseline kept for comparison/fallback.
"""

from __future__ import annotations

from models.lstm_model import RehabLSTM
from models.stgcn_model import STGCN
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_stgcn(model_cfg):
    stgcn_cfg = model_cfg.stgcn
    model = STGCN(
        num_joints=stgcn_cfg.num_joints,
        in_channels=stgcn_cfg.in_channels,
        num_classes=model_cfg.num_classes,
        graph_strategy=stgcn_cfg.graph_strategy,
        max_hop=stgcn_cfg.max_hop,
        edge_importance_weighting=stgcn_cfg.edge_importance_weighting,
        temporal_kernel_size=stgcn_cfg.temporal_kernel_size,
        dropout=stgcn_cfg.dropout,
        channels=list(stgcn_cfg.channels),
        strides=list(stgcn_cfg.strides),
    )

    pretrained_cfg = stgcn_cfg.get("pretrained_weights") if hasattr(stgcn_cfg, "get") else None
    if pretrained_cfg and pretrained_cfg.get("enabled"):
        logger.warning(
            "model.stgcn.pretrained_weights.enabled=true, but no compatible "
            "pretrained checkpoint is auto-loaded (see the config note on "
            "skeleton-graph mismatch). Load weights explicitly via "
            "utils.checkpoint_utils.load_model_weights() if you have a "
            "compatible checkpoint."
        )
    return model


def _build_lstm(model_cfg):
    lstm_cfg = model_cfg.lstm
    return RehabLSTM(
        input_dim=lstm_cfg.input_dim,
        hidden_dim=lstm_cfg.hidden_dim,
        num_layers=lstm_cfg.num_layers,
        num_classes=model_cfg.num_classes,
        dropout=lstm_cfg.dropout,
        bidirectional=lstm_cfg.bidirectional,
    )


_MODEL_BUILDERS = {
    "stgcn": _build_stgcn,
    "lstm": _build_lstm,
    # TODO: register future architectures here, e.g. "motionbert": _build_motionbert
    # (see docs/architecture.md for the documented MotionBERT fallback plan).
}


def build_model(model_cfg):
    """Instantiate a model from the `model` section of the application config
    (see configs/model.yaml)."""
    architecture = model_cfg.architecture.lower()
    builder = _MODEL_BUILDERS.get(architecture)

    if builder is None:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Available: {list(_MODEL_BUILDERS.keys())}"
        )

    model = builder(model_cfg)
    logger.info("Built model: %s", model.model_summary())
    return model


def data_representation_for(model_cfg) -> str:
    """Return the expected input tensor layout for the configured
    architecture: 'graph' (N, C, T, V) for ST-GCN, or 'flat' (N, T, C*V)
    for the LSTM baseline. Used by datasets/rehab_dataset.py and
    training/dataset_loader.py to shape batches correctly."""
    return "graph" if model_cfg.architecture.lower() == "stgcn" else "flat"
