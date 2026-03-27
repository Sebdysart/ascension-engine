#!/usr/bin/env python3
"""
Ascension Engine — Narrative Arc Engine

Three-act MogNarrative structure:
  Act 1 "Victim"     (0–3s):  warm grade, ~1 cut/sec, victim clips
  Act 2 "Awakening"  (3–8s):  rapid 8–12 cuts/sec, desaturation ramps, shake builds
  Act 3 "Ascension"  (8–15s): cold grade, S-tier only, reveal hierarchy

Phonk sync map:
  0–3s  → intro/verse  → Act 1
  3–8s  → build        → Act 2
  8–11s → drop         → Act 3 head
  11–13s→ drop cont.   → Act 3 peak
  13–15s→ outro        → Act 3 close

Usage:
    from data.narrative_engine import build_narrative, describe_narrative
    n = build_narrative(bpm=114.0)
    print(describe_narrative(n))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

Act = Literal["victim", "awakening", "ascension"]

# ── Act boundaries ─────────────────────────────────────────────────────────────
ACT_BOUNDARIES: list[tuple[float, float, Act]] = [
    (0.0,  3.0,  "victim"),
    (3.0,  8.0,  "awakening"),
    (8.0,  15.0, "ascension"),
]

# ── Grade specs per act ────────────────────────────────────────────────────────
GRADE_SPEC: dict[str, dict] = {
    "victim": {
        "brightness_adj": +0.40,
        "saturation_adj": +0.10,
        "color_temp": "warm",
        "css_filter": "brightness(1.4) saturate(1.1) sepia(0.15)",
    },
    "awakening": {
        "brightness_adj": 0.0,
        "saturation_adj": -0.15,
        "color_temp": "neutral",
        "css_filter": "brightness(1.0) saturate(0.85) contrast(1.1)",
    },
    "ascension": {
        "brightness_adj": -0.30,
        "saturation_adj": -0.30,
        "color_temp": "cold",
        "css_filter": "brightness(0.7) saturate(0.7) contrast(1.35) hue-rotate(10deg)",
    },
}

PRE_DROP_SILENCE_DEFAULT = 0.06   # seconds; range 0.04–0.08
ASCENSION_REVEAL_ORDER = ["eyes", "jawline", "frame_shoulders", "full_walk"]


@dataclass
class PhonkSection:
    name: str
    start_sec: float
    end_sec: float
    act: Act
    is_drop: bool = False


@dataclass
class NarrativeSlot:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    act: Act
    section: str
    is_victim_slot: bool = False
    is_pre_drop: bool = False
    angle_inversion_required: bool = False
    pre_drop_silence_sec: float = 0.0
    slow_mo: bool = False
    reveal_tier: str = ""
    shake_intensity: float = 0.0
    zoom_pulse: bool = False


@dataclass
class MogNarrative:
    total_duration_sec: float
    bpm: float
    slots: list[NarrativeSlot] = field(default_factory=list)
    drop_times: list[float] = field(default_factory=list)
    phonk_sections: list[PhonkSection] = field(default_factory=list)
    pre_drop_victim_indices: list[int] = field(default_factory=list)


def get_act(time_sec: float) -> Act:
    for start, end, act in ACT_BOUNDARIES:
        if start <= time_sec < end:
            return act
    return "ascension"


def build_phonk_sections(total_sec: float = 15.0) -> list[PhonkSection]:
    return [
        PhonkSection("intro",  0.0,  3.0,  "victim",    is_drop=False),
        PhonkSection("build",  3.0,  8.0,  "awakening", is_drop=False),
        PhonkSection("drop",   8.0,  11.0, "ascension", is_drop=True),
        PhonkSection("drop",   11.0, 13.0, "ascension", is_drop=True),
        PhonkSection("outro",  13.0, min(15.0, total_sec), "ascension", is_drop=False),
    ]


def detect_drop_times(beats: list[float], total_sec: float = 15.0) -> list[float]:
    drop_anchors = [8.0, 11.0]
    if not beats:
        return drop_anchors
    result = []
    for anchor in drop_anchors:
        candidates = [b for b in beats if anchor <= b <= anchor + 1.5]
        result.append(min(candidates) if candidates else anchor)
    return result


def _slot_duration_for_act(act: Act, bpm: float) -> float:
    half_beat = 120.0 / bpm
    if act == "victim":
        return 2.0 * half_beat
    elif act == "awakening":
        eighth = 30.0 / bpm
        return max(0.08, min(eighth, 0.13))
    else:
        return half_beat


def _shake_for_act(act: Act, t: float, sec_start: float, sec_end: float) -> float:
    if act == "victim":
        return 0.0
    elif act == "awakening":
        if sec_end <= sec_start:
            return 0.0
        progress = (t - sec_start) / (sec_end - sec_start)
        return round(progress * 60.0, 1)
    else:
        return 40.0


def build_narrative(
    bpm: float = 114.0,
    total_sec: float = 15.0,
    beats: list[float] | None = None,
    slow_mo_count: int = 2,
) -> MogNarrative:
    """
    Build a complete MogNarrative arc.

    Parameters
    ----------
    bpm:           Song BPM (108–120 typical for phonk)
    total_sec:     Total edit duration in seconds
    beats:         Detected beat timestamps from librosa
    slow_mo_count: Strategic slow-mo frames in Act 3 (1–2)
    """
    _beats = beats or []
    sections = build_phonk_sections(total_sec)
    drop_times = detect_drop_times(_beats, total_sec)

    narrative = MogNarrative(
        total_duration_sec=total_sec,
        bpm=bpm,
        drop_times=drop_times,
        phonk_sections=sections,
    )

    slots: list[NarrativeSlot] = []
    idx = 0

    for section in sections:
        act = section.act
        sec_start = section.start_sec
        sec_end = min(section.end_sec, total_sec)
        slot_dur = _slot_duration_for_act(act, bpm)
        t = sec_start
        while t < sec_end - 1e-6:
            end = min(t + slot_dur, sec_end)
            actual = end - t
            if actual < slot_dur * 0.25:
                t += slot_dur
                continue
            slot = NarrativeSlot(
                index=idx,
                start_sec=t,
                end_sec=end,
                duration_sec=actual,
                act=act,
                section=section.name,
                shake_intensity=_shake_for_act(act, t, sec_start, sec_end),
            )
            slots.append(slot)
            idx += 1
            t += slot_dur

    # Mark victim slots (Act 1 = all victim)
    for slot in slots:
        if slot.act == "victim":
            slot.is_victim_slot = True

    # Mark 2 beats before each drop as victim (unless in Act 3)
    half_beat = 120.0 / bpm
    for drop_t in drop_times:
        for slot in slots:
            if slot.act != "ascension" and drop_t - 2 * half_beat <= slot.start_sec < drop_t:
                slot.is_victim_slot = True
                slot.is_pre_drop = True
                slot.pre_drop_silence_sec = PRE_DROP_SILENCE_DEFAULT

    # NEVER victim in Act 3
    for slot in slots:
        if slot.act == "ascension":
            slot.is_victim_slot = False
            slot.is_pre_drop = False
            slot.pre_drop_silence_sec = 0.0

    # Angle inversion after every victim slot
    for i, slot in enumerate(slots):
        if slot.is_victim_slot and i + 1 < len(slots):
            slots[i + 1].angle_inversion_required = True

    # Zoom pulse on drop beats
    for slot in slots:
        for drop_t in drop_times:
            if abs(slot.start_sec - drop_t) < 0.15:
                slot.zoom_pulse = True

    # Strategic slow-mo in Act 3 — evenly spaced up to slow_mo_count
    act3 = [s for s in slots if s.act == "ascension" and not s.zoom_pulse]
    placed = 0
    if act3 and slow_mo_count > 0:
        # Space placements evenly: pick every (len/count)-th slot
        step = max(1, len(act3) // max(1, slow_mo_count))
        for i, slot in enumerate(act3):
            if i % step == (step // 2) and placed < slow_mo_count:
                slot.slow_mo = True
                placed += 1

    # Ascension reveal hierarchy
    act3_all = [s for s in slots if s.act == "ascension"]
    for i, slot in enumerate(act3_all):
        slot.reveal_tier = ASCENSION_REVEAL_ORDER[i % len(ASCENSION_REVEAL_ORDER)]

    narrative.pre_drop_victim_indices = [i for i, s in enumerate(slots) if s.is_pre_drop]
    narrative.slots = slots
    return narrative


def describe_narrative(narrative: MogNarrative) -> str:
    act_counts: dict[str, int] = {"victim": 0, "awakening": 0, "ascension": 0}
    victim_slots = []
    for s in narrative.slots:
        act_counts[s.act] = act_counts.get(s.act, 0) + 1
        if s.is_victim_slot:
            victim_slots.append(s.index)

    lines = [
        f"MogNarrative  BPM={narrative.bpm}  dur={narrative.total_duration_sec}s  slots={len(narrative.slots)}",
        f"  Act 1 (Victim):     {act_counts['victim']} slots",
        f"  Act 2 (Awakening):  {act_counts['awakening']} slots",
        f"  Act 3 (Ascension):  {act_counts['ascension']} slots",
        f"  Drop times:         {narrative.drop_times}",
        f"  Victim slots:       {victim_slots}",
    ]
    for s in narrative.slots:
        flags = []
        if s.is_victim_slot:            flags.append("VICTIM")
        if s.is_pre_drop:               flags.append("pre-drop")
        if s.zoom_pulse:                flags.append("zoom-pulse")
        if s.slow_mo:                   flags.append("slow-mo")
        if s.angle_inversion_required:  flags.append("invert-angle")
        flag_str = " [" + ",".join(flags) + "]" if flags else ""
        lines.append(
            f"  [{s.index:2d}] {s.start_sec:5.2f}→{s.end_sec:5.2f}s  {s.act:10s}  "
            f"shake={s.shake_intensity:4.0f}  {s.reveal_tier}{flag_str}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    bpm = float(sys.argv[1]) if len(sys.argv) > 1 else 114.0
    print(describe_narrative(build_narrative(bpm=bpm)))
