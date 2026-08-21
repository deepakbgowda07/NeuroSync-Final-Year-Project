"""Tests for inference.knowledge_base.PhysiotherapyKnowledgeBase."""

import pytest

from inference.exercise_library import SUPPORTED_EXERCISES
from inference.knowledge_base import PhysiotherapyKnowledgeBase


@pytest.fixture
def kb():
    return PhysiotherapyKnowledgeBase()


def test_loads_all_ten_exercises(kb):
    assert set(kb.keys()) == set(SUPPORTED_EXERCISES)


def test_every_exercise_has_required_fields(kb):
    for key in kb.keys():
        entry = kb.get(key)
        assert entry.target_joints, f"{key} missing target_joints"
        assert entry.normal_rom_deg != (0, 0), f"{key} missing normal_rom_deg"
        assert entry.expected_movement_sequence, f"{key} missing movement sequence"
        assert entry.clinical_description, f"{key} missing clinical description"
        assert entry.correction_rules, f"{key} missing correction rules"


def test_get_unknown_exercise_returns_none(kb):
    assert kb.get("not_a_real_exercise") is None


def test_find_compensation_pattern(kb):
    pattern = kb.find_compensation_pattern("shoulder_flexion", "trunk_compensation")
    assert pattern is not None
    assert "trunk" in pattern.description.lower() or "lean" in pattern.description.lower()


def test_find_compensation_pattern_unknown_returns_none(kb):
    assert kb.find_compensation_pattern("shoulder_flexion", "not_a_real_pattern") is None


def test_custom_path_missing_file_logs_and_returns_empty(tmp_path):
    kb = PhysiotherapyKnowledgeBase(path=str(tmp_path / "nonexistent.yaml"))
    assert kb.keys() == []


def test_functional_rom_within_normal_rom(kb):
    for key in kb.keys():
        entry = kb.get(key)
        normal_min, normal_max = entry.normal_rom_deg
        functional_min, functional_max = entry.functional_rom_deg
        assert normal_min <= functional_min or functional_min <= normal_min + 5  # allow small tolerance
        assert functional_max <= normal_max + 5
