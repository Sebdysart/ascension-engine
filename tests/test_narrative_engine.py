# tests/test_narrative_engine.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))

from narrative_engine import (
    build_narrative, get_act, describe_narrative,
    ACT_BOUNDARIES, GRADE_SPEC,
)


def test_get_act_boundaries():
    assert get_act(0.0) == "victim"
    assert get_act(2.9) == "victim"
    assert get_act(3.0) == "awakening"
    assert get_act(7.9) == "awakening"
    assert get_act(8.0) == "ascension"
    assert get_act(14.9) == "ascension"


def test_build_narrative_slot_count():
    n = build_narrative(bpm=114.0)
    assert len(n.slots) > 0


def test_no_victim_in_act3():
    n = build_narrative(bpm=114.0)
    for slot in n.slots:
        if slot.act == "ascension":
            assert not slot.is_victim_slot, f"Slot {slot.index} at {slot.start_sec:.2f}s is ascension but marked victim"


def test_act1_all_victim():
    n = build_narrative(bpm=114.0)
    act1 = [s for s in n.slots if s.act == "victim"]
    assert len(act1) > 0
    for s in act1:
        assert s.is_victim_slot


def test_angle_inversion_after_victim():
    n = build_narrative(bpm=114.0)
    for i, slot in enumerate(n.slots):
        if slot.is_victim_slot and i + 1 < len(n.slots):
            assert n.slots[i + 1].angle_inversion_required, \
                f"Slot {i+1} after victim slot {i} should require angle inversion"


def test_grade_specs_defined():
    for act in ("victim", "awakening", "ascension"):
        assert act in GRADE_SPEC
        assert "css_filter" in GRADE_SPEC[act]


def test_drop_times_default():
    n = build_narrative(bpm=114.0, beats=[])
    assert 8.0 in n.drop_times or any(abs(t - 8.0) < 0.5 for t in n.drop_times)


def test_pre_drop_silence():
    n = build_narrative(bpm=114.0)
    pre_drops = [s for s in n.slots if s.is_pre_drop]
    for s in pre_drops:
        assert 0.04 <= s.pre_drop_silence_sec <= 0.08


def test_act3_reveal_hierarchy():
    n = build_narrative(bpm=114.0)
    act3 = [s for s in n.slots if s.act == "ascension"]
    assert len(act3) > 0
    for s in act3:
        assert s.reveal_tier in ("eyes", "jawline", "frame_shoulders", "full_walk")


def test_describe_narrative():
    n = build_narrative(bpm=114.0)
    desc = describe_narrative(n)
    assert "MogNarrative" in desc
    assert "Victim" in desc
    assert "Awakening" in desc
    assert "Ascension" in desc
