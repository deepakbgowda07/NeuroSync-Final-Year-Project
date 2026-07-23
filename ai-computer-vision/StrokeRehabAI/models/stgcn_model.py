"""
stgcn_model.py
===============
Spatial-Temporal Graph Convolutional Network (ST-GCN), the project's
default movement-assessment model, per Yan, Xiong & Lin (2018).

Operates directly on the skeleton graph (MediaPipe's 33 joints) rather
than a flattened feature vector, so the network can learn spatial
(joint-to-joint) relationships jointly with temporal dynamics — a
better inductive bias for movement-quality assessment than treating
joints as independent input channels (as the LSTM baseline does).

Input shape:  (N, C, T, V)      N=batch, C=in_channels (xyz[,vis]),
                                  T=sequence_length, V=num_joints (33)
Output shape: (N, num_classes)

TODO (next development phase):
- Add a person dimension (M) for multi-person clips if ever needed;
  current implementation assumes a single tracked patient (M=1).
- Benchmark `graph_strategy: "uniform"` vs `"distance"` vs `"spatial"`
  once labeled data is available (spatial is ST-GCN's reported best).
"""

from __future__ import annotations

from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - allow import-time introspection without torch
    torch = None
    nn = object  # type: ignore
    F = None

from models.base_model import BaseRehabModel
from models.graph_utils import Graph
from utils.logger import get_logger

logger = get_logger(__name__)


class SpatialGraphConv(nn.Module if torch is not None else object):
    """Graph convolution: applies a 1x1 conv to expand channels by the
    number of adjacency subsets, then contracts each subset against its
    corresponding adjacency matrix (einsum), summing the results."""

    def __init__(self, in_channels: int, out_channels: int, num_subsets: int):
        super().__init__()
        self.num_subsets = num_subsets
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x, A):
        """x: (N, C, T, V); A: (num_subsets, V, V) -> (N, out_channels, T, V)"""
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.num_subsets, kc // self.num_subsets, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous()


class STGCNBlock(nn.Module if torch is not None else object):
    """One spatial-graph-conv + temporal-conv block with a residual
    connection, matching the standard ST-GCN basic block layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.3,
        residual: bool = True,
    ):
        super().__init__()
        padding = (temporal_kernel_size - 1) // 2

        self.gcn = SpatialGraphConv(in_channels, out_channels, num_subsets)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(BaseRehabModel, nn.Module if torch is not None else object):
    """Full ST-GCN network: a stack of STGCNBlocks over a fixed skeleton
    graph, global-average-pooled and classified by a linear head."""

    def __init__(
        self,
        num_joints: int = 33,
        in_channels: int = 3,
        num_classes: int = 5,
        graph_strategy: str = "spatial",
        max_hop: int = 1,
        edge_importance_weighting: bool = True,
        temporal_kernel_size: int = 9,
        dropout: float = 0.3,
        channels: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to instantiate STGCN.")

        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.num_classes = num_classes

        graph = Graph(num_node=num_joints, strategy=graph_strategy, max_hop=max_hop)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer("A", A)
        num_subsets = A.shape[0]

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        channels = channels or [64, 64, 64, 128, 128, 128, 256, 256, 256]
        strides = strides or [1] * len(channels)
        if len(channels) != len(strides):
            raise ValueError("model.stgcn.channels and .strides must have the same length.")

        self.blocks = nn.ModuleList()
        prev_channels = in_channels
        for i, (out_channels, stride) in enumerate(zip(channels, strides)):
            self.blocks.append(
                STGCNBlock(
                    prev_channels, out_channels, num_subsets,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=stride, dropout=dropout,
                    residual=(i != 0),  # first block has no residual (channel mismatch is common)
                )
            )
            prev_channels = out_channels

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(A.shape)) for _ in self.blocks]
            )
        else:
            self.edge_importance = [1.0] * len(self.blocks)

        self.final_channels = prev_channels
        self.classifier = nn.Linear(prev_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (N, C, T, V) -> logits: (N, num_classes)"""
        n, c, t, v = x.size()

        # Normalize jointly over the (channel, joint) dimension per the
        # original ST-GCN data-BN step, then restore (N, C, T, V).
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        for block, importance in zip(self.blocks, self.edge_importance):
            A_weighted = self.A * importance if isinstance(importance, torch.Tensor) else self.A
            x = block(x, A_weighted)

        # Global average pool over time and joints -> (N, C)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(n, self.final_channels)
        x = self.dropout(x)
        return self.classifier(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def model_summary(self):
        return {
            "architecture": "STGCN",
            "num_joints": self.num_joints,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "num_blocks": len(self.blocks),
            "final_channels": self.final_channels,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
