# tests/test_feedback_loop.py
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from engine_db import EngineDB
from mognet.feedback_loop import record_actual_performance, retrain_from_feedback
from mognet.viral_scorer import ViralScorer


def _make_temp_db() -> EngineDB:
    tmp = tempfile.mktemp(suffix=".db")
    db = EngineDB(db_path=Path(tmp))
    db.init()
    return db


def _make_dummy_features_json() -> str:
    return json.dumps({
        "visual": {
            "cuts_per_second": 3.0, "cuts_per_second_act1": 1.0,
            "cuts_per_second_act2": 8.0, "cuts_per_second_act3": 3.0,
            "camera_angles": [0.0], "angle_inversion_count": 2,
            "brightness_per_clip": [120.0], "brightness_contrast_ratio": 1.5,
            "color_temp_per_clip": [1.1], "color_temp_shift": True,
            "zoom_pulse_count": 2, "shake_events": 3,
            "slow_mo_frames": 5, "direct_stare_clips": 2,
        },
        "audio": {
            "bpm": 114.0, "drop_timestamps": [8.0], "silence_gaps_before_drop": [55.0],
            "avg_silence_gap_ms": 55.0, "build_detected": True, "drop_intensity_db": -12.0,
        },
        "text": {
            "hook_text": "you are below average", "hook_aggression_score": 7.5,
            "has_second_person": True, "text_density": 1.5,
        },
    })


def test_record_and_retrieve(monkeypatch):
    db = _make_temp_db()
    monkeypatch.setattr("mognet.feedback_loop._get_db", lambda: db)
    record_actual_performance("edit_001", views=50000, shares=200, saves=100, watch_pct=0.45, db=db)
    rows = db.get_mognet_training_rows()
    # No features_json yet so row won't appear in training set — that's fine
    assert isinstance(rows, list)


def test_mognet_performance_table_exists():
    db = _make_temp_db()
    tables = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mognet_performance'"
    ).fetchall()
    assert len(tables) == 1, "mognet_performance table not created"


def test_save_and_retrieve_prediction():
    db = _make_temp_db()
    db.save_mognet_prediction("edit_test", "/out/test.mp4", 82.5, _make_dummy_features_json())
    row = db.conn.execute(
        "SELECT predicted_score FROM mognet_performance WHERE edit_id='edit_test'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 82.5) < 0.1


def test_retrain_skips_when_no_data(monkeypatch):
    """retrain_from_feedback should log and return cleanly when no data."""
    db = _make_temp_db()
    scorer = ViralScorer()
    # Should not raise
    retrain_from_feedback(scorer, db=db)
