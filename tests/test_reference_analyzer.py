# tests/test_reference_analyzer.py
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.reference_analyzer import analyze_gold_library, _synthetic_engagement_for


def test_synthetic_engagement_is_positive():
    score = _synthetic_engagement_for("bp_masterx_7589676463071268118.mp4", 0.85)
    assert score
    assert score["views"] > 0


def test_synthetic_engagement_scales_with_fidelity():
    low  = _synthetic_engagement_for("some_video.mp4", 0.3)
    high = _synthetic_engagement_for("some_video.mp4", 0.9)
    low_eng  = low["views"] * low["watch_pct"] + low["shares"] * 10 + low["saves"] * 5
    high_eng = high["views"] * high["watch_pct"] + high["shares"] * 10 + high["saves"] * 5
    assert high_eng > low_eng


def test_analyze_gold_library_runs():
    """analyze_gold_library should complete without error (may skip if no gold)."""
    gold_dir = Path(__file__).resolve().parent.parent / "input" / "gold"
    if not gold_dir.exists() or not list(gold_dir.glob("*.mp4")):
        pytest.skip("No gold videos available")
    model_path = Path(__file__).resolve().parent / "test_viral_scorer.pkl"
    scorer = analyze_gold_library(model_path=str(model_path))
    assert scorer is not None
    model_path.unlink(missing_ok=True)
