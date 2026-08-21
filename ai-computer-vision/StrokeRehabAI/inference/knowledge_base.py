"""
knowledge_base.py
====================
Loads the physiotherapy knowledge base (`knowledge_base/exercises_knowledge.yaml`)
— clinical reference data (target joints/muscles, normal and functional
ROM ranges, expected movement sequence, known compensation patterns,
plain-language clinical descriptions, and correction rules) — kept
separate from `configs/exercises.yaml`'s *runtime* thresholds (which
drive rep detection and error thresholds in the real-time engine).

This is the reference layer `inference/clinical_reasoning.py` and the
dashboard's recommendation/report modules can cite when explaining
*why* a threshold matters clinically, not just that it was crossed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "exercises_knowledge.yaml"


@dataclass
class CompensationPattern:
    name: str
    description: str


@dataclass
class ExerciseKnowledge:
    exercise_key: str
    target_joints: List[str]
    target_muscles: List[str]
    normal_rom_deg: Tuple[float, float]
    functional_rom_deg: Tuple[float, float]
    expected_movement_sequence: List[str]
    compensation_patterns: List[CompensationPattern]
    clinical_description: str
    correction_rules: List[str]


class PhysiotherapyKnowledgeBase:
    """Loads and exposes ExerciseKnowledge entries from the editable
    YAML knowledge base file."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else DEFAULT_KNOWLEDGE_BASE_PATH
        self._entries: Dict[str, ExerciseKnowledge] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            logger.error("Knowledge base file not found: %s", self.path)
            return

        with open(self.path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        exercises = data.get("knowledge_base", {}).get("exercises", {})
        for key, entry in exercises.items():
            self._entries[key] = ExerciseKnowledge(
                exercise_key=key,
                target_joints=entry.get("target_joints", []),
                target_muscles=entry.get("target_muscles", []),
                normal_rom_deg=tuple(entry.get("normal_rom_deg", [0, 0])),
                functional_rom_deg=tuple(entry.get("functional_rom_deg", [0, 0])),
                expected_movement_sequence=entry.get("expected_movement_sequence", []),
                compensation_patterns=[
                    CompensationPattern(name=p["name"], description=p["description"])
                    for p in entry.get("compensation_patterns", [])
                ],
                clinical_description=(entry.get("clinical_description") or "").strip(),
                correction_rules=entry.get("correction_rules", []),
            )

        logger.info("Loaded physiotherapy knowledge base: %d exercises from %s", len(self._entries), self.path)

    def get(self, exercise_key: str) -> Optional[ExerciseKnowledge]:
        return self._entries.get(exercise_key)

    def keys(self) -> List[str]:
        return list(self._entries.keys())

    def all(self) -> Dict[str, ExerciseKnowledge]:
        return dict(self._entries)

    def find_compensation_pattern(self, exercise_key: str, pattern_name: str) -> Optional[CompensationPattern]:
        knowledge = self.get(exercise_key)
        if knowledge is None:
            return None
        return next((p for p in knowledge.compensation_patterns if p.name == pattern_name), None)
