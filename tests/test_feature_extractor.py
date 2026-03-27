# tests/test_feature_extractor.py
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.feature_extractor import extract_video_features

GOLD_DIR = Path(__file__).resolve().parent.parent / "input" / "gold"
GOLD_VIDEOS = list(GOLD_DIR.glob("*.mp4"))


def test_extract_returns_required_keys():
    """Feature extractor must return all three top-level keys."""
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    assert "visual" in feats
    assert "audio" in feats
    assert "text" in feats


def test_visual_keys_present():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    v = feats["visual"]
    for key in ["cuts_per_second", "cuts_per_second_act1", "cuts_per_second_act2",
                "cuts_per_second_act3", "camera_angles", "angle_inversion_count",
                "brightness_per_clip", "brightness_contrast_ratio",
                "color_temp_per_clip", "color_temp_shift",
                "zoom_pulse_count", "shake_events", "slow_mo_frames",
                "direct_stare_clips"]:
        assert key in v, f"Missing visual key: {key}"


def test_audio_keys_present():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    a = feats["audio"]
    for key in ["bpm", "drop_timestamps", "silence_gaps_before_drop",
                "avg_silence_gap_ms", "build_detected", "drop_intensity_db"]:
        assert key in a, f"Missing audio key: {key}"


def test_text_keys_present():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    t = feats["text"]
    for key in ["hook_text", "hook_aggression_score", "has_second_person", "text_density"]:
        assert key in t, f"Missing text key: {key}"


def test_numeric_values_are_floats_or_ints():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    assert isinstance(feats["visual"]["cuts_per_second"], float)
    assert isinstance(feats["audio"]["bpm"], float)
    assert isinstance(feats["text"]["hook_aggression_score"], float)


def test_invalid_path_raises():
    with pytest.raises(Exception):
        extract_video_features("/nonexistent/path.mp4")
