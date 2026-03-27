# tests/test_viral_scorer.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.viral_scorer import ViralScorer


def _make_dummy_features(n: int = 10) -> list[dict]:
    return [
        {
            "visual": {
                "cuts_per_second": 2.0 + i * 0.3,
                "cuts_per_second_act1": 1.0,
                "cuts_per_second_act2": 8.0 + i,
                "cuts_per_second_act3": 3.0,
                "camera_angles": [0.0, 15.0, -10.0],
                "angle_inversion_count": i % 3,
                "brightness_per_clip": [120.0, 80.0, 150.0],
                "brightness_contrast_ratio": 1.5 + i * 0.1,
                "color_temp_per_clip": [1.2, 0.9, 0.8],
                "color_temp_shift": bool(i % 2),
                "zoom_pulse_count": i % 4,
                "shake_events": i % 5,
                "slow_mo_frames": i % 3,
                "direct_stare_clips": i % 4,
            },
            "audio": {
                "bpm": 108.0 + i,
                "drop_timestamps": [8.0, 11.0],
                "silence_gaps_before_drop": [50.0 + i * 5],
                "avg_silence_gap_ms": 50.0 + i * 5,
                "build_detected": bool(i % 2),
                "drop_intensity_db": -15.0 + i,
            },
            "text": {
                "hook_text": "you are below average",
                "hook_aggression_score": 5.0 + i * 0.3,
                "has_second_person": True,
                "text_density": 1.5,
            },
        }
        for i in range(n)
    ]


def _make_dummy_metrics(n: int = 10) -> list[dict]:
    return [
        {"views": 10000 * (i + 1), "watch_pct": 0.4 + i * 0.05,
         "shares": 100 * i, "saves": 50 * i}
        for i in range(n)
    ]


def test_train_and_predict():
    scorer = ViralScorer()
    features = _make_dummy_features(10)
    metrics  = _make_dummy_metrics(10)
    scorer.train(features, metrics)
    result = scorer.predict(features[0])
    assert "score" in result
    assert "confidence" in result
    assert "breakdown" in result
    assert 0.0 <= result["score"] <= 100.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_without_train_raises():
    scorer = ViralScorer()
    with pytest.raises(RuntimeError, match="not trained"):
        scorer.predict(_make_dummy_features(1)[0])


def test_save_and_load(tmp_path):
    scorer = ViralScorer()
    scorer.train(_make_dummy_features(10), _make_dummy_metrics(10))
    pkl_path = str(tmp_path / "scorer.pkl")
    scorer.save(pkl_path)
    scorer2 = ViralScorer()
    scorer2.load(pkl_path)
    result = scorer2.predict(_make_dummy_features(1)[0])
    assert 0.0 <= result["score"] <= 100.0


def test_breakdown_has_expected_keys():
    scorer = ViralScorer()
    scorer.train(_make_dummy_features(10), _make_dummy_metrics(10))
    result = scorer.predict(_make_dummy_features(1)[0])
    bd = result["breakdown"]
    assert "cut_rate_contribution" in bd
    assert "audio_contribution" in bd
    assert "hook_contribution" in bd
