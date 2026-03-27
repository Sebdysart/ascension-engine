# tests/test_sequencer_narrative.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))

from sequencer import build_narrative_sequence, NarrativeAwareSlot


def test_returns_narrative_aware_slots():
    slots = build_narrative_sequence(bpm=114.0)
    assert len(slots) > 0
    assert all(hasattr(s, "act") for s in slots)
    assert all(hasattr(s, "narrative_role") for s in slots)


def test_act1_clips_from_victim_pool():
    slots = build_narrative_sequence(bpm=114.0)
    act1 = [s for s in slots if s.act == "victim"]
    for s in act1:
        assert s.preferred_pool in ("victim_contrast", "low_cinematic")


def test_act3_clips_s_tier_only():
    slots = build_narrative_sequence(bpm=114.0)
    act3 = [s for s in slots if s.act == "ascension"]
    for s in act3:
        assert s.preferred_pool == "good_parts"


def test_no_victim_in_act3():
    slots = build_narrative_sequence(bpm=114.0)
    for s in slots:
        if s.act == "ascension":
            assert not s.is_victim_slot


def test_pre_drop_silence_present():
    slots = build_narrative_sequence(bpm=114.0)
    pre_drops = [s for s in slots if s.pre_drop_silence_sec > 0]
    # Should have silence slots before drops
    assert len(pre_drops) > 0
    for s in pre_drops:
        assert 0.04 <= s.pre_drop_silence_sec <= 0.08


def test_backwards_compat_build_sequence():
    from sequencer import build_sequence, Slot
    slots = build_sequence(bpm=114.0)
    assert len(slots) > 0
    assert isinstance(slots[0], Slot)
