#!/usr/bin/env python3
"""
Ascension Engine — Beat-Grid Sequencer

Builds a cut schedule (list of slots with duration and section type) from a BPM
and a section layout.  Each slot maps to one clip in the final edit.

Beat-grid rules
───────────────
These edits use a half-time phonk feel where the *felt* pulse runs at BPM/2.
One "half-time beat" = 120 / bpm  seconds.

  • drop / buildup  → cut every half-time beat  (120 / bpm  sec per slot)
  • verse           → cut every 2nd half-time beat  (240 / bpm  sec per slot)

BUG that was here: slot_duration was pinned to the manifest clip duration (3 s),
producing only ~4 slots per 15-second edit (≈ 0.31 cuts/sec).  The fix below
derives slot duration from the beat grid so a 15-second edit at 108–120 BPM
lands in the 10–13 slots / 0.7–1.0 cuts·s⁻¹ window.

Usage
─────
  from data.sequencer import build_sequence

  slots = build_sequence(bpm=114, total_sec=15.0)
  # [{'start': 0.0, 'end': 2.105, 'duration': 2.105, 'section': 'verse'}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SectionType = Literal["verse", "buildup", "drop"]


@dataclass
class Section:
    name: SectionType
    duration_sec: float


@dataclass
class Slot:
    start: float
    end: float
    duration: float
    section: SectionType
    clip_index: int = 0  # filled in by the caller / playlist builder


# ── Default section layout for a 15-second edit ───────────────────────────────
# verse 3 s → buildup 3 s → drop 6 s → verse 3 s  =  15 s
DEFAULT_SECTIONS: list[Section] = [
    Section("verse",   3.0),
    Section("buildup", 3.0),
    Section("drop",    6.0),
    Section("verse",   3.0),
]


def _slot_duration(section: SectionType, bpm: float) -> float:
    """
    Return the cut interval in seconds for the given section type.

    Half-time beat = 120 / bpm.
      drop / buildup : 1 × half-time beat  →  120 / bpm  seconds
      verse          : 2 × half-time beat  →  240 / bpm  seconds
    """
    half_beat = 120.0 / bpm
    if section in ("drop", "buildup"):
        return half_beat          # e.g. 1.0 s at 120 BPM, 1.053 s at 114 BPM
    else:  # verse
        return 2.0 * half_beat    # e.g. 2.0 s at 120 BPM, 2.105 s at 114 BPM


def build_sequence(
    bpm: float = 114.0,
    sections: list[Section] | None = None,
    total_sec: float | None = None,
) -> list[Slot]:
    """
    Build a beat-grid cut schedule.

    Parameters
    ----------
    bpm:
        Song BPM (108–120 typical for phonk edits).
    sections:
        Ordered list of Section objects.  Defaults to DEFAULT_SECTIONS (15 s).
    total_sec:
        If given, overrides section list with a flat drop-paced grid of that
        duration (useful for quick testing).

    Returns
    -------
    List of Slot objects in chronological order.
    """
    if total_sec is not None:
        # Flat mode: fill `total_sec` with drop-paced slots
        dur = _slot_duration("drop", bpm)
        secs = sections or []
        slots: list[Slot] = []
        t = 0.0
        idx = 0
        while t + dur <= total_sec + 1e-6:
            end = min(t + dur, total_sec)
            slots.append(Slot(start=t, end=end, duration=end - t, section="drop", clip_index=idx))
            t = end
            idx += 1
        return slots

    layout = sections or DEFAULT_SECTIONS
    slots = []
    cursor = 0.0
    clip_idx = 0

    for sec in layout:
        slot_dur = _slot_duration(sec.name, bpm)
        sec_end = cursor + sec.duration_sec
        t = cursor

        while t < sec_end - 1e-6:
            end = min(t + slot_dur, sec_end)
            actual_dur = end - t
            # Skip runt slots shorter than 25 % of a full slot
            if actual_dur >= slot_dur * 0.25:
                slots.append(Slot(
                    start=t,
                    end=end,
                    duration=actual_dur,
                    section=sec.name,
                    clip_index=clip_idx,
                ))
                clip_idx += 1
            t += slot_dur

        cursor = sec_end

    return slots


def describe(slots: list[Slot], bpm: float | None = None) -> str:
    """Return a human-readable summary of the cut schedule."""
    if not slots:
        return "No slots."
    total = slots[-1].end
    n = len(slots)
    rate = n / total if total > 0 else 0
    lines = [
        f"Slots: {n}   Total: {total:.2f}s   Rate: {rate:.2f} cuts/s"
        + (f"   BPM: {bpm}" if bpm else ""),
        "",
    ]
    for i, s in enumerate(slots):
        lines.append(
            f"  [{i+1:2d}] {s.start:5.2f}→{s.end:5.2f}s  ({s.duration:.3f}s)  {s.section}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    bpm = float(sys.argv[1]) if len(sys.argv) > 1 else 114.0
    slots = build_sequence(bpm=bpm)
    print(describe(slots, bpm=bpm))
