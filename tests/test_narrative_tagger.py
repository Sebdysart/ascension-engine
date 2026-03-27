# tests/test_narrative_tagger.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))

from narrative_tagger import (
    classify_narrative_role, tag_clips_by_narrative_role,
    NARRATIVE_ROLES,
)


def test_narrative_roles_defined():
    assert "victim_act1" in NARRATIVE_ROLES
    assert "awakening_transition" in NARRATIVE_ROLES
    assert "ascension_reveal" in NARRATIVE_ROLES
    assert "jawline_pop" in NARRATIVE_ROLES
    assert "full_walk" in NARRATIVE_ROLES


def test_victim_act1_classification():
    clip = {
        "mog_track": "victim_contrast",
        "mog_score": 0.3,
        "tags": ["face_closeup"],
    }
    role = classify_narrative_role(clip)
    assert role == "victim_act1"


def test_ascension_reveal_classification():
    clip = {
        "mog_track": "good_parts",
        "mog_score": 0.75,
        "tags": ["hunter_eyes", "dark_cinema_mood"],
    }
    role = classify_narrative_role(clip)
    assert role == "ascension_reveal"


def test_jawline_pop_classification():
    clip = {
        "mog_track": "good_parts",
        "mog_score": 0.7,
        "tags": ["jawline_pop", "high_contrast"],
    }
    role = classify_narrative_role(clip)
    assert role == "jawline_pop"


def test_full_walk_classification():
    clip = {
        "mog_track": "mid_tier",
        "mog_score": 0.5,
        "tags": ["full_body", "motion_toward_camera"],
    }
    role = classify_narrative_role(clip)
    assert role == "full_walk"


def test_awakening_transition_classification():
    clip = {
        "mog_track": "mid_tier",
        "mog_score": 0.5,
        "tags": ["high_energy_cut", "motion"],
    }
    role = classify_narrative_role(clip)
    assert role == "awakening_transition"


def test_tag_clips_updates_roles():
    clips = [
        {"clip_id": "c1", "mog_track": "victim_contrast", "mog_score": 0.3, "tags": []},
        {"clip_id": "c2", "mog_track": "good_parts", "mog_score": 0.75,
         "tags": ["hunter_eyes"]},
    ]
    result = tag_clips_by_narrative_role(clips, dry_run=True)
    assert result["c1"] == "victim_act1"
    assert result["c2"] == "ascension_reveal"
