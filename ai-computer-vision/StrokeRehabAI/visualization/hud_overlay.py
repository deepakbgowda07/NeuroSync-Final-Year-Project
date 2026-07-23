"""
hud_overlay.py
===============
Heads-up-display overlay rendered on top of the live camera feed:
current exercise, current phase, rep counter, movement quality,
exercise timer, FPS, CUDA status, and model confidence — everything
the "Visual Feedback" spec requires beyond the skeleton itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class HUDState:
    fps: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    exercise_display_name: Optional[str] = None
    phase: Optional[str] = None
    rep_count: int = 0
    movement_quality: Optional[float] = None
    session_elapsed_seconds: float = 0.0
    cuda_available: bool = False
    model_confidence: Optional[float] = None
    view: Optional[str] = None


class HUDOverlay:
    def __init__(self, visualization_cfg):
        self.font_scale = visualization_cfg.hud_font_scale

    def draw(self, frame: np.ndarray, state: HUDState) -> np.ndarray:
        import cv2

        left_lines = [f"FPS: {state.fps:.1f}"]
        for stage, ms in state.stage_timings.items():
            left_lines.append(f"{stage}: {ms:.1f}ms")
        left_lines.append(f"CUDA: {'ON' if state.cuda_available else 'OFF'}")

        right_lines = []
        if state.exercise_display_name:
            right_lines.append(f"Exercise: {state.exercise_display_name}")
        if state.phase:
            right_lines.append(f"Phase: {state.phase}")
        right_lines.append(f"Reps: {state.rep_count}")
        if state.movement_quality is not None:
            right_lines.append(f"Quality: {state.movement_quality * 100:.0f}%")
        if state.model_confidence is not None:
            right_lines.append(f"Confidence: {state.model_confidence * 100:.0f}%")
        if state.view:
            right_lines.append(f"View: {state.view}")
        right_lines.append(self._format_timer(state.session_elapsed_seconds))

        self._draw_lines(frame, left_lines, origin=(10, 24), color=(255, 255, 255))
        self._draw_lines(frame, right_lines, origin=(max(10, frame.shape[1] - 260), 24), color=(0, 255, 255))

        return frame

    @staticmethod
    def _format_timer(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"Time: {minutes:02d}:{secs:02d}"

    def _draw_lines(self, frame: np.ndarray, lines, origin, color) -> None:
        import cv2

        x, y = origin
        for line in lines:
            cv2.putText(
                frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale, color, 1, cv2.LINE_AA,
            )
            y += 22
