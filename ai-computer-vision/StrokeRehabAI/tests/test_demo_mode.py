"""Tests for inference.demo_mode."""

import pytest

from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.db import get_connection
from dashboard.session_manager import SessionManager
from inference.demo_mode import DemoModeRunner, DemoSourceMode, generate_demo_dataset


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "demo.db")


def test_generate_demo_dataset_creates_patient_and_sessions(db_path):
    patient_id = generate_demo_dataset(db_path, num_sessions=5)
    assert patient_id is not None

    manager = SessionManager(db_path)
    sessions = manager.list_sessions(patient_id=patient_id)
    assert len(sessions) == 5


def test_generate_demo_dataset_produces_valid_analytics(db_path):
    patient_id = generate_demo_dataset(db_path, num_sessions=6)
    engine = RecoveryAnalyticsEngine(db_path)
    analytics = engine.compute_patient_analytics(patient_id)
    assert analytics.num_sessions == 6
    assert 0 <= analytics.recovery_score <= 100


def test_generate_demo_dataset_creates_active_rehabilitation_plan(db_path):
    patient_id = generate_demo_dataset(db_path, num_sessions=3)
    with get_connection(db_path) as conn:
        plan = conn.execute(
            "SELECT * FROM rehabilitation_plans WHERE patient_id = ? AND is_active = 1", (patient_id,)
        ).fetchone()
    assert plan is not None


def test_generate_demo_dataset_is_callable_repeatedly(db_path):
    id1 = generate_demo_dataset(db_path, num_sessions=2)
    id2 = generate_demo_dataset(db_path, num_sessions=2)
    assert id1 != id2  # each call creates a distinct demo patient, not overwriting


def test_demo_runner_session_replay_mode(db_path, capsys):
    patient_id = generate_demo_dataset(db_path, num_sessions=2)
    manager = SessionManager(db_path)
    sessions = manager.list_sessions(patient_id=patient_id)
    session_id = sessions[0].session_id

    from configs.config_loader import load_config

    cfg = load_config(force_reload=True)
    cfg.dashboard.database_path = db_path
    runner = DemoModeRunner(cfg=cfg)
    runner.run(DemoSourceMode.SESSION, source_path=str(session_id))

    captured = capsys.readouterr()
    assert "quality=" in captured.out


def test_demo_runner_video_mode_missing_file_raises(db_path):
    from configs.config_loader import load_config

    cfg = load_config(force_reload=True)
    cfg.dashboard.database_path = db_path
    runner = DemoModeRunner(cfg=cfg)
    with pytest.raises(FileNotFoundError):
        runner.run(DemoSourceMode.VIDEO, source_path="/nonexistent/video.mp4")


def test_demo_runner_video_mode_no_path_raises(db_path):
    from configs.config_loader import load_config

    cfg = load_config(force_reload=True)
    cfg.dashboard.database_path = db_path
    runner = DemoModeRunner(cfg=cfg)
    with pytest.raises(ValueError):
        runner.run(DemoSourceMode.VIDEO, source_path=None)


def test_demo_runner_session_mode_no_id_raises(db_path):
    from configs.config_loader import load_config

    cfg = load_config(force_reload=True)
    cfg.dashboard.database_path = db_path
    runner = DemoModeRunner(cfg=cfg)
    with pytest.raises(ValueError):
        runner.run(DemoSourceMode.SESSION, source_path=None)
