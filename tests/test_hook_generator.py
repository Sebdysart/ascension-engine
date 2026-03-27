# tests/test_hook_generator.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))

from hook_generator import (
    HookSpec, select_flaw_clip, generate_hook_spec, _estimate_brightness,
)


def test_hookspec_defaults():
    spec = HookSpec(
        flaw_clip_id="test_clip",
        flaw_clip_path="library/victim_contrast/test.mp4",
        flaw_clip_brightness=0.8,
    )
    assert spec.zoom_start_pct == 150.0
    assert spec.zoom_end_pct == 100.0
    assert spec.zoom_duration_sec == 0.5
    assert spec.shake_peak == 100.0
    assert spec.zoom_pulse_peak_pct == 115.0
    assert spec.grade_start == "warm"
    assert spec.grade_end == "cold"


def test_hookspec_to_json():
    spec = HookSpec(flaw_clip_id="x", flaw_clip_path="p", flaw_clip_brightness=0.5)
    j = spec.to_json()
    import json
    d = json.loads(j)
    assert d["flaw_clip_id"] == "x"
    assert d["zoom_start_pct"] == 150.0


def test_estimate_brightness_inversion():
    clip_high = {"mog_score": 0.1}  # low mog = high brightness for victim
    clip_low  = {"mog_score": 0.9}  # high mog = darker, better lighting
    assert _estimate_brightness(clip_high) > _estimate_brightness(clip_low)


def test_select_flaw_clip_picks_victim():
    clips = [
        {"clip_id": "a", "mog_track": "good_parts", "mog_score": 0.8, "tags": []},
        {"clip_id": "b", "mog_track": "victim_contrast", "mog_score": 0.2, "tags": []},
        {"clip_id": "c", "mog_track": "victim_contrast", "mog_score": 0.3, "tags": []},
    ]
    flaw = select_flaw_clip(clips)
    assert flaw is not None
    assert flaw["clip_id"] == "b"  # lowest mog_score = highest brightness


def test_select_flaw_clip_prefers_soft_tags():
    clips = [
        {"clip_id": "a", "mog_track": "victim_contrast", "mog_score": 0.2, "tags": []},
        {"clip_id": "b", "mog_track": "victim_contrast", "mog_score": 0.25,
         "tags": ["flat_lighting_cope"]},
    ]
    flaw = select_flaw_clip(clips)
    assert flaw is not None
    # b has soft tag bonus, even with slightly higher mog_score
    assert flaw["clip_id"] == "b"


def test_generate_hook_spec_no_clips():
    spec = generate_hook_spec(clips=[], beats=[])
    assert spec is None


def test_generate_hook_spec_with_clips():
    clips = [
        {"clip_id": "victim_1", "mog_track": "victim_contrast", "mog_score": 0.2, "tags": [],
         "file_path": "library/victim_contrast/v1.mp4"},
    ]
    spec = generate_hook_spec(clips=clips, beats=[0.25, 0.5, 1.0])
    assert spec is not None
    assert spec.flaw_clip_id == "victim_1"
    assert spec.bass_hit_time_sec == 0.25
