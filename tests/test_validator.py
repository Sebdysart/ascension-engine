# tests/test_validator.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.validator import validate_edit, _apply_rule_critiques
from mognet.viral_scorer import ViralScorer


def _make_scorer(score_override: float = 80.0) -> ViralScorer:
    """Return a trained scorer that always predicts score_override."""
    class _MockScorer(ViralScorer):
        def predict(self, feats):
            return {
                "score": score_override,
                "confidence": 0.8,
                "breakdown": {
                    "cut_rate_contribution": 0.4,
                    "audio_contribution": 0.35,
                    "hook_contribution": 0.25,
                },
            }
    s = _MockScorer()
    s._trained = True
    return s


def _make_good_features() -> dict:
    return {
        "visual": {
            "cuts_per_second": 4.0, "cuts_per_second_act1": 1.2,
            "cuts_per_second_act2": 8.5, "cuts_per_second_act3": 3.0,
            "camera_angles": [30.0, 5.0, -5.0], "angle_inversion_count": 2,
            "brightness_per_clip": [130.0, 70.0, 160.0],
            "brightness_contrast_ratio": 1.6, "color_temp_per_clip": [1.2, 0.85],
            "color_temp_shift": True, "zoom_pulse_count": 2,
            "shake_events": 3, "slow_mo_frames": 10, "direct_stare_clips": 4,
        },
        "audio": {
            "bpm": 114.0, "drop_timestamps": [8.0, 11.0],
            "silence_gaps_before_drop": [55.0], "avg_silence_gap_ms": 55.0,
            "build_detected": True, "drop_intensity_db": -10.0,
        },
        "text": {
            "hook_text": "you are below average", "hook_aggression_score": 7.5,
            "has_second_person": True, "text_density": 1.8,
        },
    }


def _make_bad_features() -> dict:
    f = _make_good_features()
    f["audio"]["avg_silence_gap_ms"] = 20.0    # too short
    f["visual"]["angle_inversion_count"] = 0    # missing inversion
    f["text"]["hook_aggression_score"] = 4.0    # too passive
    f["visual"]["brightness_contrast_ratio"] = 1.1   # not washed out
    f["visual"]["cuts_per_second_act2"] = 3.0   # too slow
    return f


def test_approve_on_good_features(tmp_path):
    """High scorer + good features → APPROVE."""
    from unittest.mock import patch
    scorer = _make_scorer(82.0)
    good = _make_good_features()
    with patch("mognet.validator.extract_video_features", return_value=good):
        result = validate_edit(str(tmp_path / "edit.mp4"), scorer)
    assert result["decision"] == "APPROVE"
    assert result["viral_score"] == 82.0


def test_reject_on_low_score(tmp_path):
    """Score < 75 → auto-REJECT regardless of rules."""
    from unittest.mock import patch
    scorer = _make_scorer(60.0)
    good = _make_good_features()
    with patch("mognet.validator.extract_video_features", return_value=good):
        result = validate_edit(str(tmp_path / "edit.mp4"), scorer)
    assert result["decision"] == "REJECT"


def test_critiques_on_bad_features():
    """All 5 rule critiques fire on bad features."""
    bad = _make_bad_features()
    critiques, warnings = _apply_rule_critiques(bad)
    # Should have 2 CRITICALs and 3 WARNINGs
    combined = critiques + warnings
    assert any("silence gap" in c.lower() for c in combined)
    assert any("angle" in c.lower() or "power hierarchy" in c.lower() for c in combined)
    assert any("hook" in c.lower() or "passive" in c.lower() for c in combined)
    assert any("victim" in c.lower() or "washed" in c.lower() or "brightness" in c.lower() for c in combined)
    assert any("awakening" in c.lower() or "act2" in c.lower() or "slow" in c.lower() for c in combined)


def test_result_keys():
    """validate_edit returns all required keys."""
    from unittest.mock import patch
    scorer = _make_scorer(80.0)
    with patch("mognet.validator.extract_video_features", return_value=_make_good_features()):
        result = validate_edit("/fake/path.mp4", scorer)
    for key in ["decision", "viral_score", "strengths", "warnings", "critiques", "features"]:
        assert key in result, f"Missing key: {key}"


def test_reject_on_critical_rule(tmp_path):
    """CRITICAL rule should force REJECT even with high score."""
    from unittest.mock import patch
    scorer = _make_scorer(85.0)
    bad = _make_good_features()
    bad["visual"]["angle_inversion_count"] = 0  # CRITICAL
    bad["visual"]["cuts_per_second_act2"] = 2.0  # CRITICAL
    with patch("mognet.validator.extract_video_features", return_value=bad):
        result = validate_edit(str(tmp_path / "edit.mp4"), scorer)
    assert result["decision"] == "REJECT"
