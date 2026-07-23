"""
session_report.py
==================
Builds a structured summary for a single exercise session: duration,
mean prediction confidence, joint-angle ROM summary, and feedback
messages issued — the data backing both the dashboard's Reports page
and any future PDF export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from utils.json_export import write_json
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionReport:
    patient_name: Optional[str]
    exercise_name: str
    started_at: str
    ended_at: Optional[str] = None
    mean_confidence: Optional[float] = None
    joint_angle_summary: Dict = field(default_factory=dict)
    feedback_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "patient_name": self.patient_name,
            "exercise_name": self.exercise_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "mean_confidence": self.mean_confidence,
            "joint_angle_summary": self.joint_angle_summary,
            "feedback_log": self.feedback_log,
        }

    def save(self, output_dir: str = "outputs/evaluation_reports") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self.exercise_name.replace(' ', '_').lower()}_{timestamp}.json"
        path = write_json(self.to_dict(), f"{output_dir}/{filename}")
        logger.info("Session report saved: %s", path)
        return str(path)
