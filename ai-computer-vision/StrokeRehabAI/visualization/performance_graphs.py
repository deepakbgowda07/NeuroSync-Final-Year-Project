"""
performance_graphs.py
======================
Rolling-window performance graphs (e.g. joint angle over time, model
confidence over time) rendered with matplotlib/plotly for the
dashboard's Live Session and Recovery Analytics pages.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class PerformanceGraphRenderer:
    """Maintains rolling time-series buffers and produces plotly figures
    for embedding in the Streamlit dashboard."""

    def __init__(self, window_seconds: int = 30, fps: float = 30.0):
        self.max_points = int(window_seconds * fps)
        self._series: Dict[str, Deque[float]] = {}

    def add_point(self, series_name: str, value: float) -> None:
        buffer = self._series.setdefault(series_name, deque(maxlen=self.max_points))
        buffer.append(value)

    def build_figure(self, series_names: List[str]):
        import plotly.graph_objects as go

        fig = go.Figure()
        for name in series_names:
            values = list(self._series.get(name, []))
            fig.add_trace(go.Scatter(y=values, mode="lines", name=name))

        fig.update_layout(
            title="Session Performance", xaxis_title="Frame", yaxis_title="Value",
            template="plotly_white", height=350,
        )
        return fig
