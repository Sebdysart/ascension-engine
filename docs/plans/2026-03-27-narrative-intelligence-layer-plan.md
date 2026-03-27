# Narrative Intelligence Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a three-act MogNarrative system that gives every edit psychological structure — victim placement, awakening arc, ascension reveal — on top of the existing beat-grid and dual-track mog system.

**Architecture:** Six components. Three new Python modules (`narrative_engine.py`, `hook_generator.py`, `narrative_tagger.py`), two updated files (`sequencer.py`, `generate_batch.py`), one new Remotion composition (`BrutalBeatMontage.tsx`). All new Python modules are importable standalone and wired into `generate_batch.py` as the new default generation path.

**Tech Stack:** Python 3.11+, dataclasses, SQLite via `engine_db.py`, FFmpeg (silence gap injection), Remotion + React + TypeScript, pytest (new test suite in `tests/`).

---

## Existing file map — READ BEFORE EDITING

| File | Role |
|------|------|
| `data/sequencer.py` | Beat-grid builder — `Slot` dataclass, `build_sequence()` |
| `data/generate_batch.py` | Top-level orchestrator — `generate_from_template()` |
| `data/vision_tagger.py` | Mog scoring — `classify_mog_track()`, `MOG_S_TIER=0.65`, `MOG_VICTIM=0.45` |
| `data/engine_db.py` | SQLite — `clips.track CHECK IN ('unclassified','good_parts','victim_contrast','archived')` |
| `library/tags/index.json` | Tags store — `{"tags": {"tag_name": ["clip_id", ...]}}` |
| `remotion/Root.tsx` | Remotion composition registry |
| `remotion/compositions/MogEdit.tsx` | Soft archetype — DO NOT MODIFY |
| `clip-manifest.json` | Master flat clip list |

---

## Task 1: Create `data/narrative_engine.py` — Three-act arc core

**Files:**
- Create: `data/narrative_engine.py`
- Create: `tests/test_narrative_engine.py`

### Step 1: Create `tests/test_narrative_engine.py` (failing)

```python
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
```

### Step 2: Run test to verify it fails

```bash
cd /Users/sebastiandysart/Desktop/ascension-engine/.claude/worktrees/magical-hypatia
python -m pytest tests/test_narrative_engine.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'narrative_engine'`

### Step 3: Create `data/narrative_engine.py`

```python
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

    # Strategic slow-mo in Act 3
    act3 = [s for s in slots if s.act == "ascension" and not s.zoom_pulse]
    placed = 0
    for i, slot in enumerate(act3):
        if i % 4 == 2 and placed < slow_mo_count:
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
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_narrative_engine.py -v
```

Expected: All 10 tests PASS.

### Step 5: Commit

```bash
git add data/narrative_engine.py tests/test_narrative_engine.py
git commit -m "feat(narrative): add narrative_engine.py — three-act MogNarrative arc"
```

---

## Task 2: Create `data/hook_generator.py` — Scroll-stopping hook formula

**Files:**
- Create: `data/hook_generator.py`
- Create: `tests/test_hook_generator.py`

### Step 1: Create `tests/test_hook_generator.py` (failing)

```python
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
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_hook_generator.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'hook_generator'`

### Step 3: Create `data/hook_generator.py`

```python
#!/usr/bin/env python3
"""
Ascension Engine — Hook Generator

Generates the scroll-stopping hook specification for frames 1-15 (first ~0.5s).

Formula:
  1. Select "flaw clip" — highest brightness, softest lighting from victim_contrast/
  2. Zoom-out: 150%→100% over 0.5s
  3. On first bass hit:
     - Shake spike: 30→100→0 over 0.2s
     - Zoom pulse: 100%→115%→100% over 0.3s
     - Grade shift: warm→cold

Returns HookSpec as JSON consumed by BrutalBeatMontage + sequencer.

Usage:
    from data.hook_generator import generate_hook_spec
    spec = generate_hook_spec(clips=clips, beats=beats)
    print(spec.to_json())
"""

from __future__ import annotations
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _ROOT / "library" / "engine.db"
_CLIP_MANIFEST = _ROOT / "clip-manifest.json"

log = logging.getLogger("hook_generator")

SOFT_LIGHTING_TAGS = frozenset({
    "flat_lighting_cope", "soft_light", "natural_indoor",
    "overhead_light", "flat_lighting",
})


@dataclass
class HookSpec:
    """Complete spec for the scroll-stopping hook sequence."""
    flaw_clip_id: str
    flaw_clip_path: str
    flaw_clip_brightness: float

    zoom_start_pct: float = 150.0
    zoom_end_pct: float = 100.0
    zoom_duration_sec: float = 0.5

    bass_hit_time_sec: float = 0.0
    shake_start: float = 30.0
    shake_peak: float = 100.0
    shake_end: float = 0.0
    shake_duration_sec: float = 0.2

    zoom_pulse_start_pct: float = 100.0
    zoom_pulse_peak_pct: float = 115.0
    zoom_pulse_end_pct: float = 100.0
    zoom_pulse_duration_sec: float = 0.3

    grade_start: str = "warm"
    grade_end: str = "cold"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _estimate_brightness(clip: dict) -> float:
    """Invert mog_score: victim clips (low mog) = bright/soft = high brightness."""
    return round(1.0 - float(clip.get("mog_score", 0.5)), 3)


def select_flaw_clip(clips: list[dict]) -> dict | None:
    """
    Pick the best "flaw clip" for the hook.
    - Must be victim_contrast track
    - Highest estimated brightness (lowest mog_score)
    - Bonus for soft-lighting tags
    Falls back to lowest-rank clip if no victim clips exist.
    """
    victim_clips = [
        c for c in clips
        if c.get("mog_track") == "victim_contrast" or c.get("track") == "victim_contrast"
    ]

    if not victim_clips:
        victim_clips = sorted(clips, key=lambda c: float(c.get("rank", 0.5)))[:3]

    if not victim_clips:
        return None

    def _score(c: dict) -> float:
        brightness = _estimate_brightness(c)
        soft_bonus = 0.2 if set(c.get("tags", [])) & SOFT_LIGHTING_TAGS else 0.0
        return brightness + soft_bonus

    return max(victim_clips, key=_score)


def _load_clips_from_manifest() -> list[dict]:
    if not _CLIP_MANIFEST.exists():
        return []
    try:
        return json.loads(_CLIP_MANIFEST.read_text()).get("clips", [])
    except Exception:
        return []


def _load_victim_clips_from_db() -> list[dict]:
    if not _DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT clip_id, file_path, rank, tags FROM clips WHERE track = 'victim_contrast'"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def generate_hook_spec(
    beats: list[float] | None = None,
    clips: list[dict] | None = None,
    dry_run: bool = False,
) -> HookSpec | None:
    """
    Generate a HookSpec for the scroll-stopping hook sequence.

    Parameters
    ----------
    beats:   Detected beat timestamps — first beat = first bass hit
    clips:   Clip list (loaded from clip-manifest.json if None)
    dry_run: Log-only mode, skip DB reads

    Returns
    -------
    HookSpec or None if no suitable flaw clip found
    """
    _clips = clips if clips is not None else _load_clips_from_manifest()

    if not _clips:
        _clips = _load_victim_clips_from_db()

    if not _clips:
        log.warning("hook_generator: no clips available")
        return None

    flaw_clip = select_flaw_clip(_clips)
    if flaw_clip is None:
        log.warning("hook_generator: no victim/flaw clip found")
        return None

    clip_id   = flaw_clip.get("clip_id") or flaw_clip.get("id", "unknown")
    clip_path = flaw_clip.get("file_path") or flaw_clip.get("file", "")
    brightness = _estimate_brightness(flaw_clip)
    bass_hit   = beats[0] if beats else 0.0

    spec = HookSpec(
        flaw_clip_id=clip_id,
        flaw_clip_path=clip_path,
        flaw_clip_brightness=brightness,
        bass_hit_time_sec=bass_hit,
    )

    log.info("HookSpec: flaw_clip=%s brightness=%.2f bass_hit=%.2fs", clip_id, brightness, bass_hit)
    return spec


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    spec = generate_hook_spec()
    if spec:
        print(spec.to_json())
    else:
        print("[error] No flaw clip found — ingest victim_contrast clips first")
        sys.exit(1)
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_hook_generator.py -v
```

Expected: All 7 tests PASS.

### Step 5: Commit

```bash
git add data/hook_generator.py tests/test_hook_generator.py
git commit -m "feat(narrative): add hook_generator.py — scroll-stopping hook formula"
```

---

## Task 3: Update `data/sequencer.py` — Act-aware slot assignment

**Files:**
- Modify: `data/sequencer.py`
- Create: `tests/test_sequencer_narrative.py`

### Step 1: Read current sequencer.py

Read `data/sequencer.py` lines 1–168 before editing.

### Step 2: Create `tests/test_sequencer_narrative.py` (failing)

```python
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
```

### Step 3: Run test to verify it fails

```bash
python -m pytest tests/test_sequencer_narrative.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'build_narrative_sequence'`

### Step 4: Add `NarrativeAwareSlot` and `build_narrative_sequence()` to `data/sequencer.py`

Append to the end of `data/sequencer.py` (after line 168, before `if __name__ == "__main__"`):

```python
# ── Narrative-aware slot (extends Slot with act context) ──────────────────────

from dataclasses import dataclass as _dc
from typing import Optional as _Opt


@_dc
class NarrativeAwareSlot:
    """Slot with full narrative arc metadata for act-aware clip assignment."""
    start: float
    end: float
    duration: float
    section: SectionType
    clip_index: int
    act: str                         # "victim" | "awakening" | "ascension"
    narrative_role: str              # narrative_engine reveal_tier / "victim" / "awakening"
    is_victim_slot: bool
    angle_inversion_required: bool
    pre_drop_silence_sec: float      # 0.04–0.08 or 0.0
    slow_mo: bool
    zoom_pulse: bool
    shake_intensity: float
    preferred_pool: str              # "victim_contrast" | "mid_tier" | "good_parts" | "low_cinematic"


def build_narrative_sequence(
    bpm: float = 114.0,
    total_sec: float = 15.0,
    beats: list | None = None,
) -> list[NarrativeAwareSlot]:
    """
    Build a narrative-aware cut schedule.

    Wraps build_narrative() from narrative_engine and maps NarrativeSlots
    to NarrativeAwareSlots that carry pool hints for clip assignment.

    Act 1 → preferred_pool = "victim_contrast" (or "low_cinematic" fallback)
    Act 2 → preferred_pool = "mid_tier"
    Act 3 → preferred_pool = "good_parts"
    Victim slots before drop → preferred_pool = "victim_contrast"
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from narrative_engine import build_narrative, MogNarrative

    # Map act → sequencer SectionType for backwards compat
    _ACT_SECTION: dict[str, SectionType] = {
        "victim": "verse",
        "awakening": "buildup",
        "ascension": "drop",
    }

    _ACT_POOL: dict[str, str] = {
        "victim": "victim_contrast",
        "awakening": "mid_tier",
        "ascension": "good_parts",
    }

    narrative: MogNarrative = build_narrative(bpm=bpm, total_sec=total_sec, beats=beats or [])

    out: list[NarrativeAwareSlot] = []
    for ns in narrative.slots:
        act = ns.act
        pool = "victim_contrast" if ns.is_victim_slot else _ACT_POOL.get(act, "mid_tier")
        section: SectionType = _ACT_SECTION.get(act, "verse")

        out.append(NarrativeAwareSlot(
            start=ns.start_sec,
            end=ns.end_sec,
            duration=ns.duration_sec,
            section=section,
            clip_index=ns.index,
            act=act,
            narrative_role=ns.reveal_tier or act,
            is_victim_slot=ns.is_victim_slot,
            angle_inversion_required=ns.angle_inversion_required,
            pre_drop_silence_sec=ns.pre_drop_silence_sec,
            slow_mo=ns.slow_mo,
            zoom_pulse=ns.zoom_pulse,
            shake_intensity=ns.shake_intensity,
            preferred_pool=pool,
        ))

    return out
```

### Step 5: Run tests to verify they pass

```bash
python -m pytest tests/test_sequencer_narrative.py -v
```

Expected: All 6 tests PASS.

### Step 6: Verify backwards compat

```bash
python -m pytest tests/test_sequencer_narrative.py::test_backwards_compat_build_sequence -v
```

Expected: PASS.

### Step 7: Commit

```bash
git add data/sequencer.py tests/test_sequencer_narrative.py
git commit -m "feat(sequencer): add NarrativeAwareSlot + build_narrative_sequence() — act-aware assignment"
```

---

## Task 4: Create `data/narrative_tagger.py` — Tag library clips by narrative role

**Files:**
- Create: `data/narrative_tagger.py`
- Create: `tests/test_narrative_tagger.py`

### Step 1: Create `tests/test_narrative_tagger.py` (failing)

```python
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
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_narrative_tagger.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'narrative_tagger'`

### Step 3: Create `data/narrative_tagger.py`

```python
#!/usr/bin/env python3
"""
Ascension Engine — Narrative Tagger

Classifies existing library clips by narrative role using existing metadata:
mog_score, mog_track, tags. No new Claude vision calls needed.

Roles:
  victim_act1           — victim_contrast track + no direct stare
  awakening_transition  — mid_tier + motion tags
  ascension_reveal      — good_parts + stare/dark tags
  jawline_pop           — tight crop + high contrast lower face
  full_walk             — motion toward camera + full body

Writes narrative_role tag into library/tags/index.json.

Usage:
    python data/narrative_tagger.py [--dry-run]
    python data/narrative_tagger.py --report
"""

from __future__ import annotations
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_MANIFEST = _ROOT / "clip-manifest.json"
_TAGS_INDEX    = _ROOT / "library" / "tags" / "index.json"

log = logging.getLogger("narrative_tagger")

NARRATIVE_ROLES = {
    "victim_act1",
    "awakening_transition",
    "ascension_reveal",
    "jawline_pop",
    "full_walk",
    "general",          # fallback
}

# Tag sets used for classification
_DIRECT_STARE_TAGS = frozenset({
    "hunter_eyes", "direct_stare", "talking_head_direct", "eye_contact_direct",
    "mog_face_closeup_push",
})
_DARK_GRADE_TAGS = frozenset({
    "dark_cinema_mood", "dark_grade", "crushed_blacks", "night_car",
})
_MOTION_TAGS = frozenset({
    "high_energy_cut", "motion", "gym_broll_slowmo", "training_montage",
    "street_walk", "fast_cut", "high_energy",
})
_JAWLINE_TAGS = frozenset({
    "jawline_pop", "jaw_pop_side", "bone_structure_reveal", "tight_crop",
    "high_contrast",
})
_FULL_BODY_TAGS = frozenset({
    "full_body", "motion_toward_camera", "full_walk", "outfit_reveal",
    "lifestyle_broll",
})


def classify_narrative_role(clip: dict) -> str:
    """
    Classify a single clip's narrative role from its metadata.

    Priority order (first match wins):
    1. jawline_pop  — tight jawline/high-contrast tags
    2. full_walk    — full body + motion tags
    3. victim_act1  — victim_contrast track + no direct stare
    4. ascension_reveal — good_parts + stare or dark tags
    5. awakening_transition — mid_tier + motion
    6. general      — fallback
    """
    tags = frozenset(clip.get("tags", []))
    track = clip.get("mog_track") or clip.get("track", "unclassified")

    # 1. jawline_pop: tight jawline crop regardless of track
    if tags & _JAWLINE_TAGS:
        return "jawline_pop"

    # 2. full_walk: full body motion regardless of track
    if tags & _FULL_BODY_TAGS:
        return "full_walk"

    # 3. victim_act1: victim track, no direct stare
    if track == "victim_contrast" and not (tags & _DIRECT_STARE_TAGS):
        return "victim_act1"

    # 4. ascension_reveal: good_parts + stare or dark
    if track == "good_parts" and (tags & _DIRECT_STARE_TAGS or tags & _DARK_GRADE_TAGS):
        return "ascension_reveal"

    # 5. general good_parts (no specific stare/dark tag) → still ascension
    if track == "good_parts":
        return "ascension_reveal"

    # 6. awakening_transition: mid_tier + motion
    if track in ("mid_tier", "unclassified") and tags & _MOTION_TAGS:
        return "awakening_transition"

    return "general"


def tag_clips_by_narrative_role(
    clips: list[dict],
    dry_run: bool = False,
) -> dict[str, str]:
    """
    Classify all clips and return mapping clip_id → narrative_role.

    Parameters
    ----------
    clips:   List of clip dicts with clip_id, mog_track, mog_score, tags
    dry_run: If True, just return mapping without writing to index

    Returns
    -------
    dict mapping clip_id → narrative_role
    """
    role_map: dict[str, str] = {}
    for clip in clips:
        cid = clip.get("clip_id") or clip.get("id", "unknown")
        role = classify_narrative_role(clip)
        role_map[cid] = role

    if not dry_run:
        _write_narrative_roles_to_index(role_map)

    return role_map


def _write_narrative_roles_to_index(role_map: dict[str, str]) -> None:
    """Write narrative_role tags to library/tags/index.json."""
    index: dict = {"tags": {}, "last_updated": ""}
    if _TAGS_INDEX.exists():
        try:
            index = json.loads(_TAGS_INDEX.read_text())
        except Exception:
            pass

    # Clear existing narrative role buckets
    for role in NARRATIVE_ROLES:
        tag_key = f"narrative_{role}"
        index["tags"][tag_key] = []

    # Populate
    for clip_id, role in role_map.items():
        tag_key = f"narrative_{role}"
        if tag_key not in index["tags"]:
            index["tags"][tag_key] = []
        if clip_id not in index["tags"][tag_key]:
            index["tags"][tag_key].append(clip_id)

    from datetime import datetime, timezone
    index["last_updated"] = datetime.now(timezone.utc).isoformat()
    _TAGS_INDEX.write_text(json.dumps(index, indent=2))
    log.info("Tags index updated with narrative roles — %d clips tagged", len(role_map))


def _load_manifest_clips() -> list[dict]:
    if not _CLIP_MANIFEST.exists():
        return []
    try:
        return json.loads(_CLIP_MANIFEST.read_text()).get("clips", [])
    except Exception:
        return []


def run_narrative_tagging(dry_run: bool = False) -> dict[str, str]:
    """
    Run narrative tagging against the full library.
    Returns role_map dict.
    """
    clips = _load_manifest_clips()
    if not clips:
        log.warning("No clips found in clip-manifest.json")
        return {}

    log.info("Tagging %d clips by narrative role …", len(clips))
    role_map = tag_clips_by_narrative_role(clips, dry_run=dry_run)

    # Report
    counts = Counter(role_map.values())
    log.info("Narrative role distribution:")
    for role, count in sorted(counts.items()):
        log.info("  %-25s  %d clips", role, count)

    return role_map


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    parser = argparse.ArgumentParser(description="Tag library clips by narrative role")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true", help="Print role distribution only")
    args = parser.parse_args()

    role_map = run_narrative_tagging(dry_run=args.dry_run or args.report)

    if args.report or args.dry_run:
        counts = Counter(role_map.values())
        print("\nNarrative Role Distribution:")
        print("─" * 40)
        for role, count in sorted(counts.items()):
            print(f"  {role:<25}  {count}")
        print(f"\n  Total: {len(role_map)} clips")
    else:
        print(f"\nTagged {len(role_map)} clips. library/tags/index.json updated.")


if __name__ == "__main__":
    main()
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_narrative_tagger.py -v
```

Expected: All 7 tests PASS.

### Step 5: Commit

```bash
git add data/narrative_tagger.py tests/test_narrative_tagger.py
git commit -m "feat(narrative): add narrative_tagger.py — classify library clips by arc role"
```

---

## Task 5: Create `remotion/compositions/BrutalBeatMontage.tsx`

**Files:**
- Create: `remotion/compositions/BrutalBeatMontage.tsx`
- Modify: `remotion/Root.tsx`

### Step 1: Read existing files

Read `remotion/Root.tsx` (26 lines) before editing.

### Step 2: Create `remotion/compositions/BrutalBeatMontage.tsx`

```tsx
/**
 * BrutalBeatMontage.tsx — High-energy three-act phonk montage
 *
 * Visual effects not present in MogEdit.tsx:
 *   - Shake: spring-animated position offset (intensity 0–100)
 *   - Zoom pulse: scale 1.0→1.15→1.0 on bass drop (0.3s)
 *   - Color grade shift: CSS filter warm(5600K)→cold(4200K) across acts
 *   - Pre-drop hold: single-frame freeze 0.04–0.08s before drop
 *   - Act-aware grade: warm Act 1, neutral ramp Act 2, cold Act 3
 *
 * Props:
 *   slots      — NarrativeSlot array from build_narrative_sequence()
 *   hookSpec   — HookSpec JSON from hook_generator.py
 *   bpm        — Song BPM
 *   musicPath  — Audio file path
 *   watermark  — Optional account handle
 */
import React from "react";
import {
  AbsoluteFill,
  Sequence,
  Video,
  Audio,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  staticFile,
  spring,
} from "remotion";

// ── Grade filters per act ──────────────────────────────────────────────────────
const ACT_GRADES: Record<string, string> = {
  victim:    "brightness(1.4) saturate(1.1) sepia(0.15)",
  awakening: "brightness(1.0) saturate(0.85) contrast(1.1)",
  ascension: "brightness(0.7) saturate(0.7) contrast(1.35) hue-rotate(10deg)",
};

// ── Types ──────────────────────────────────────────────────────────────────────
export interface SlotSpec {
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  act: "victim" | "awakening" | "ascension";
  is_victim_slot: boolean;
  zoom_pulse: boolean;
  shake_intensity: number;    // 0–100
  slow_mo: boolean;
  pre_drop_silence_sec: number;
  clip_src?: string;          // filled in by generate_batch
  trim_start_sec?: number;
}

export interface HookSpecJson {
  flaw_clip_id: string;
  flaw_clip_path: string;
  zoom_start_pct: number;
  zoom_end_pct: number;
  zoom_duration_sec: number;
  bass_hit_time_sec: number;
  shake_peak: number;
  shake_duration_sec: number;
  zoom_pulse_peak_pct: number;
  zoom_pulse_duration_sec: number;
}

export interface BrutalBeatMontageProps {
  slots: SlotSpec[];
  hookSpec?: HookSpecJson;
  bpm: number;
  musicPath?: string;
  watermark?: string;
}

// ── Shake effect ──────────────────────────────────────────────────────────────
const ShakeWrapper: React.FC<{
  intensity: number;
  isActive: boolean;
  children: React.ReactNode;
}> = ({ intensity, isActive, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const shakeX = isActive
    ? spring({
        frame,
        fps,
        config: { damping: 8, stiffness: 300, mass: 0.3 },
      }) * (intensity * 0.08)
    : 0;

  const shakeY = isActive
    ? spring({
        frame: frame + 2,
        fps,
        config: { damping: 8, stiffness: 300, mass: 0.3 },
      }) * (intensity * 0.05)
    : 0;

  return (
    <AbsoluteFill
      style={{
        transform: `translate(${shakeX}px, ${shakeY}px)`,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

// ── Zoom pulse on drop ────────────────────────────────────────────────────────
const ZoomPulseWrapper: React.FC<{
  active: boolean;
  children: React.ReactNode;
}> = ({ active, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pulseDurationFrames = Math.round(0.3 * fps);
  const scale = active
    ? interpolate(
        frame,
        [0, Math.round(pulseDurationFrames * 0.4), pulseDurationFrames],
        [1.0, 1.15, 1.0],
        { extrapolateRight: "clamp" },
      )
    : 1.0;

  return (
    <AbsoluteFill style={{ transform: `scale(${scale})`, overflow: "hidden" }}>
      {children}
    </AbsoluteFill>
  );
};

// ── Hook zoom-out (frames 1-15) ───────────────────────────────────────────────
const HookZoomOut: React.FC<{
  src: string;
  trimStart: number;
  spec: HookSpecJson;
}> = ({ src, trimStart, spec }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const zoomDurFrames = Math.round(spec.zoom_duration_sec * fps);
  const scale = interpolate(
    frame,
    [0, zoomDurFrames],
    [spec.zoom_start_pct / 100, spec.zoom_end_pct / 100],
    { extrapolateRight: "clamp" },
  );

  return (
    <AbsoluteFill style={{ transform: `scale(${scale})`, overflow: "hidden" }}>
      <Video
        src={staticFile(src)}
        startFrom={Math.round(trimStart * fps)}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    </AbsoluteFill>
  );
};

// ── Single clip with grade ────────────────────────────────────────────────────
const GradedClip: React.FC<{
  src: string;
  trimStart: number;
  act: string;
  shakeIntensity: number;
  zoomPulse: boolean;
}> = ({ src, trimStart, act, shakeIntensity, zoomPulse }) => {
  const { fps } = useVideoConfig();
  const filter = ACT_GRADES[act] || ACT_GRADES.ascension;
  const isHighEnergy = shakeIntensity > 20;

  return (
    <AbsoluteFill style={{ filter }}>
      <ShakeWrapper intensity={shakeIntensity} isActive={isHighEnergy}>
        <ZoomPulseWrapper active={zoomPulse}>
          <AbsoluteFill>
            <Video
              src={staticFile(src)}
              startFrom={Math.round(trimStart * fps)}
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
          </AbsoluteFill>
        </ZoomPulseWrapper>
      </ShakeWrapper>
    </AbsoluteFill>
  );
};

// ── Watermark ────────────────────────────────────────────────────────────────
const Watermark: React.FC<{ text: string }> = ({ text }) => (
  <div
    style={{
      position: "absolute",
      bottom: 220,
      left: 0,
      right: 0,
      textAlign: "center",
      fontFamily: "Helvetica Neue, Arial, sans-serif",
      fontWeight: 500,
      fontSize: "1.1rem",
      letterSpacing: "0.15em",
      color: "rgba(255,255,255,0.45)",
      zIndex: 20,
    }}
  >
    {text.toUpperCase()}
  </div>
);

// ── Main composition ──────────────────────────────────────────────────────────
export const BrutalBeatMontage: React.FC<BrutalBeatMontageProps> = ({
  slots = [],
  hookSpec,
  bpm,
  musicPath,
  watermark = "",
}) => {
  const { fps, durationInFrames } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* Hook sequence — always first */}
      {hookSpec && slots.length > 0 && slots[0].clip_src && (
        <Sequence from={0} durationInFrames={Math.round(hookSpec.zoom_duration_sec * fps)}>
          <HookZoomOut
            src={slots[0].clip_src}
            trimStart={slots[0].trim_start_sec ?? 0}
            spec={hookSpec}
          />
        </Sequence>
      )}

      {/* Act slots */}
      {slots.map((slot, i) => {
        if (!slot.clip_src) return null;

        const fromFrame = Math.round(slot.start_sec * fps);
        const durFrames = Math.max(1, Math.round(slot.duration_sec * fps));
        if (fromFrame >= durationInFrames) return null;
        const actualDur = Math.min(durFrames, durationInFrames - fromFrame);

        return (
          <Sequence key={i} from={fromFrame} durationInFrames={actualDur}>
            <GradedClip
              src={slot.clip_src}
              trimStart={slot.trim_start_sec ?? 0}
              act={slot.act}
              shakeIntensity={slot.shake_intensity}
              zoomPulse={slot.zoom_pulse}
            />
          </Sequence>
        );
      })}

      {watermark && <Watermark text={watermark} />}

      {musicPath && <Audio src={staticFile(musicPath)} volume={0.85} />}
    </AbsoluteFill>
  );
};
```

### Step 3: Register `BrutalBeatMontage` in `remotion/Root.tsx`

Replace the current `Root.tsx` content:

```tsx
import React from "react";
import { Composition } from "remotion";
import { VideoTemplate } from "./VideoTemplate";
import { BrutalBeatMontage } from "./compositions/BrutalBeatMontage";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="AscensionVideo"
        component={VideoTemplate as React.FC}
        durationInFrames={450}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{
          bodyClips: [],
          colorGrade: "dark_cinema",
          zoomPunch: true,
          showOverlay: false,
          musicVolume: 0.85,
        }}
      />
      <Composition
        id="BrutalBeatMontage"
        component={BrutalBeatMontage as React.FC}
        durationInFrames={450}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{
          slots: [],
          bpm: 114,
          musicPath: undefined,
          watermark: "",
        }}
      />
    </>
  );
};
```

### Step 4: Check TypeScript compiles

```bash
cd /Users/sebastiandysart/Desktop/ascension-engine/.claude/worktrees/magical-hypatia
npx tsc --noEmit 2>&1 | head -30
```

Expected: No new errors (pre-existing errors unrelated to our changes are OK).

### Step 5: Commit

```bash
git add remotion/compositions/BrutalBeatMontage.tsx remotion/Root.tsx
git commit -m "feat(remotion): add BrutalBeatMontage.tsx — shake/zoom/grade-shift effects + act-aware grades"
```

---

## Task 6: Update `data/generate_batch.py` — Wire narrative arc

**Files:**
- Modify: `data/generate_batch.py`

### Step 1: Read current generate_batch.py

Read `data/generate_batch.py` lines 1–360 before editing.

### Step 2: Add narrative imports + `_build_narrative_clips()` helper

Add after the existing imports (after line ~17, before `ROOT = `):

```python
# Narrative engine (optional — graceful fallback if not available)
def _import_narrative_engine():
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from narrative_engine import build_narrative, describe_narrative, GRADE_SPEC
        from hook_generator import generate_hook_spec
        from sequencer import build_narrative_sequence
        return build_narrative, describe_narrative, GRADE_SPEC, generate_hook_spec, build_narrative_sequence
    except ImportError as e:
        return None
```

### Step 3: Add `_assign_clips_to_narrative_slots()` function

Add after `sources_for_template()` (after line ~183):

```python
def _assign_clips_to_narrative_slots(
    slots: list,
    sources: list,
    bpm: float,
    grade: str,
    gf: str,
    tmp_dir: "Path",
    dry_run: bool = False,
) -> list[dict]:
    """
    Assign source clips to narrative slots based on each slot's preferred_pool.
    Returns list of assignment dicts with clip metadata for reporting.

    Pool → source heuristic:
      victim_contrast   → prefer dark/roge-style sources (index 0 or dark_pref)
      mid_tier          → middle sources
      good_parts        → prefer bp-coded sources (index 0 or bp_pref)
      low_cinematic     → fallback to any source
    """
    from pathlib import Path as _Path

    # Split sources by heuristic
    bp_pref    = [f for f in sources if any(k in f.name for k in ["bp","gold","morph","saffyro"])]
    dark_pref  = [f for f in sources if any(k in f.name for k in ["black","roge","pilled","editz"])]

    pool_sources = {
        "good_parts":      bp_pref or sources,
        "victim_contrast": dark_pref or sources,
        "mid_tier":        sources,
        "low_cinematic":   sources,
    }

    assignments: list[dict] = []
    for slot in slots:
        pool   = getattr(slot, "preferred_pool", "mid_tier")
        src_list = pool_sources.get(pool, sources)
        src    = src_list[len(assignments) % len(src_list)]
        dur    = slot.duration

        clip_out = tmp_dir / f"c{slot.clip_index:03d}.mp4"

        if not dry_run:
            start = best_window(src, dur)
            ok    = extract_clip(src, start, dur, clip_out, gf)
            if not ok:
                ok = extract_clip(src, 0.5, dur, clip_out, gf)
        else:
            ok = True

        assignments.append({
            "slot_index":    slot.clip_index,
            "act":           getattr(slot, "act", "unknown"),
            "is_victim":     getattr(slot, "is_victim_slot", False),
            "pre_drop_sil":  getattr(slot, "pre_drop_silence_sec", 0.0),
            "zoom_pulse":    getattr(slot, "zoom_pulse", False),
            "pool":          pool,
            "source":        src.name,
            "duration":      dur,
            "extracted":     ok,
            "path":          str(clip_out) if ok else None,
        })

    return assignments
```

### Step 4: Update `generate_from_template()` to use narrative arc

In `generate_from_template()`, replace the block starting `# Beat detection` through `# Extract clips` with the following (locate via `print(f"  ▶ Beat detection`):

```python
    # ── Narrative arc generation ──────────────────────────────────────────────
    use_narrative = _import_narrative_engine() is not None
    narrative_report = {}

    audio_src = BEAT_AUDIO.get(grade, sources[0])
    if not audio_src.exists():
        audio_src = sources[0]

    print(f"  ▶ Beat detection ({audio_src.name[:40]}) ...", flush=True)
    detected_bpm, beats = detect_beats(audio_src, max_sec=total_dur + 5)
    actual_bpm = detected_bpm if beats else float(bpm)

    if use_narrative:
        _mods = _import_narrative_engine()
        build_narrative, describe_narrative, GRADE_SPEC, generate_hook_spec, build_narrative_sequence = _mods

        print(f"  ▶ Building MogNarrative arc (BPM={actual_bpm:.1f}) ...", flush=True)
        nslots = build_narrative_sequence(bpm=actual_bpm, total_sec=total_dur, beats=beats)

        hook_spec = generate_hook_spec(beats=beats, clips=None)

        # Assign clips to narrative slots
        tmp_dir = TMP / out_name
        tmp_dir.mkdir(exist_ok=True)
        gf = GRADE_FILTER.get(grade, "")

        assignments = _assign_clips_to_narrative_slots(
            nslots, sources, actual_bpm, grade, gf, tmp_dir, dry_run
        )

        extracted = [Path(a["path"]) for a in assignments if a["extracted"] and a["path"]]
        final_durs = [a["duration"] for a in assignments if a["extracted"]]
        used = sum(final_durs)

        # Build narrative report
        act_counts = {"victim": 0, "awakening": 0, "ascension": 0}
        victim_placements = []
        pre_drop_silence_inserted = []
        for a in assignments:
            act_counts[a["act"]] = act_counts.get(a["act"], 0) + 1
            if a["is_victim"]:
                victim_placements.append(a["slot_index"])
            if a["pre_drop_sil"] > 0:
                pre_drop_silence_inserted.append({"slot": a["slot_index"], "sec": a["pre_drop_sil"]})

        narrative_report = {
            "mode": "narrative_arc",
            "act_cuts": act_counts,
            "victim_placements": victim_placements,
            "pre_drop_silence": pre_drop_silence_inserted,
            "total_cuts": len(assignments),
        }

        print(f"  ▶ Narrative: Act1={act_counts['victim']} Act2={act_counts['awakening']} Act3={act_counts['ascension']}  victims@{victim_placements}", flush=True)

        # Print assignment table
        for a in assignments:
            status = "✓" if a["extracted"] else "✗"
            victim_flag = " [VICTIM]" if a["is_victim"] else ""
            predrop_flag = f" [pre-drop-sil={a['pre_drop_sil']:.3f}s]" if a["pre_drop_sil"] > 0 else ""
            print(f"    [{a['slot_index']:2d}] {a['act']:10s}  {a['source'][:35]:38} {a['duration']:.2f}s {status}{victim_flag}{predrop_flag}", flush=True)

    else:
        # Fallback: original flat greedy assignment
        print(f"  ▶ [fallback] narrative_engine not available — using flat beat grid", flush=True)
        beat_interval = 60.0 / actual_bpm
        target_cut = total_dur / n_clips
        beats_per_cut = max(1, round(target_cut / beat_interval))
        cut_durs = []
        prev = 0.0
        if beats:
            for i, bt in enumerate(beats):
                if i % beats_per_cut == 0 and i > 0:
                    d = bt - prev
                    if 0.3 <= d <= 5.0:
                        cut_durs.append(d)
                        prev = bt
        else:
            target_cut_fb = total_dur / n_clips
            cut_durs = [target_cut_fb] * n_clips

        used, final_durs = 0.0, []
        for d in cut_durs:
            if used + d > total_dur:
                break
            final_durs.append(d)
            used += d
        if not final_durs:
            final_durs = [total_dur / n_clips] * n_clips
            used = total_dur

        avg_cut = used / len(final_durs)
        print(f"  ▶ {len(final_durs)} cuts  avg={avg_cut:.2f}s  total={used:.1f}s  BPM={actual_bpm:.1f}", flush=True)

        gf = GRADE_FILTER.get(grade, "")
        tmp_dir = TMP / out_name
        tmp_dir.mkdir(exist_ok=True)
        extracted = []

        for i, dur in enumerate(final_durs):
            src = sources[i % len(sources)]
            clip_out = tmp_dir / f"c{i:03d}.mp4"
            start = best_window(src, dur)
            ok = extract_clip(src, start, dur, clip_out, gf)
            if not ok:
                ok = extract_clip(src, 0.5, dur, clip_out, gf)
            if ok:
                extracted.append(clip_out)
                print(f"    [{i+1:2d}/{len(final_durs)}] {src.name[:35]:38} {dur:.2f}s ✓", flush=True)
            else:
                print(f"    [{i+1:2d}/{len(final_durs)}] {src.name[:35]:38} {dur:.2f}s ✗", flush=True)
```

### Step 5: Add `narrative_report` to the result dict

In `generate_from_template()`, find the `result = {` block and add:

```python
        "narrative": narrative_report,
```

### Step 6: Update `main()` results printout to show narrative data

In `main()`, in the results loop, add after the existing `print` for a successful result:

```python
            nr = r.get("narrative", {})
            if nr.get("mode") == "narrative_arc":
                ac = nr.get("act_cuts", {})
                vp = nr.get("victim_placements", [])
                ps = len(nr.get("pre_drop_silence", []))
                print(f"       Narrative: Act1={ac.get('victim',0)} Act2={ac.get('awakening',0)} Act3={ac.get('ascension',0)}")
                print(f"       Victims@slots={vp}  pre-drop-silence={ps}")
```

### Step 7: Verify no import errors

```bash
python -c "
import sys; sys.path.insert(0, 'data')
import ast, pathlib
src = pathlib.Path('data/generate_batch.py').read_text()
ast.parse(src)
print('Parse OK')
"
```

Expected: `Parse OK`

### Step 8: Commit

```bash
git add data/generate_batch.py
git commit -m "feat(generate_batch): wire narrative arc into clip assignment — act-aware pools"
```

---

## Task 7: Run `narrative_tagger.py` against the library

**Files:**
- No code changes — run tagging against existing library

### Step 1: Run dry-run first

```bash
python data/narrative_tagger.py --report
```

Expected: prints role distribution table for all clips in clip-manifest.json.

### Step 2: Run actual tagging (writes to library/tags/index.json)

```bash
python data/narrative_tagger.py
```

Expected: `Tagged N clips. library/tags/index.json updated.`

### Step 3: Verify tags index has narrative_ keys

```bash
python -c "
import json
idx = json.loads(open('library/tags/index.json').read())
narrative_keys = [k for k in idx['tags'] if k.startswith('narrative_')]
print('Narrative tag keys:', narrative_keys)
total_tagged = sum(len(v) for v in idx['tags'].values() if isinstance(v, list))
print('Total clip-tag entries:', total_tagged)
"
```

Expected: at least 5 `narrative_` keys visible.

### Step 4: Commit updated index

```bash
git add library/tags/index.json
git commit -m "chore(library): tag existing clips with narrative roles via narrative_tagger"
```

---

## Task 8: Run generate_batch.py with narrative system — report output

**Files:**
- No code changes — run generation

### Step 1: Run dry-run

```bash
python data/generate_batch.py --dry-run --top-n 3
```

Expected: Shows narrative arc planning per template. No FFmpeg calls.

### Step 2: Run full generation if `input/gold/` has clips

```bash
ls input/gold/*.mp4 2>/dev/null | head -5
```

If clips exist, run:
```bash
python data/generate_batch.py --top-n 3
```

If `input/gold/` is empty, run with dry-run only and note in commit message.

### Step 3: Report results

After generation, verify `out/generate_batch_manifest.json` contains narrative data:

```bash
python -c "
import json
manifest = json.loads(open('out/generate_batch_manifest.json').read())
for r in manifest:
    print()
    print(f'  {r[\"name\"]}')
    nr = r.get('narrative', {})
    if nr:
        ac = nr.get('act_cuts', {})
        print(f'    Act1(victim)={ac.get(\"victim\",0)}  Act2(awakening)={ac.get(\"awakening\",0)}  Act3(ascension)={ac.get(\"ascension\",0)}')
        print(f'    victim_placements={nr.get(\"victim_placements\", [])}')
        print(f'    pre_drop_silence_count={len(nr.get(\"pre_drop_silence\", []))}')
    else:
        print('    (no narrative data)')
"
```

### Step 4: Commit manifest

```bash
git add out/generate_batch_manifest.json 2>/dev/null; true
git commit -m "chore: run narrative arc generation — 3 edits with act-aware assignment"
```

---

## Task 9: Commit design doc

```bash
git add docs/plans/2026-03-27-narrative-intelligence-layer.md \
        docs/plans/2026-03-27-narrative-intelligence-layer-plan.md
git commit -m "docs: narrative intelligence layer design doc + implementation plan"
```

---

## Full test run — verify all tests pass

```bash
python -m pytest tests/ -v
```

Expected: All tests pass. Record count.

---

## Quick Reference

| Component | File | Entry Point |
|-----------|------|-------------|
| Three-act arc | `data/narrative_engine.py` | `build_narrative(bpm, total_sec, beats)` |
| Hook formula | `data/hook_generator.py` | `generate_hook_spec(beats, clips)` |
| Act-aware slots | `data/sequencer.py` | `build_narrative_sequence(bpm, total_sec, beats)` |
| Library tagger | `data/narrative_tagger.py` | `run_narrative_tagging(dry_run)` |
| Visual effects | `remotion/compositions/BrutalBeatMontage.tsx` | `BrutalBeatMontage` composition |
| Generation | `data/generate_batch.py` | `generate_from_template(t, dry_run)` |

| Constant | Value |
|----------|-------|
| Act 1 window | 0–3s (victim) |
| Act 2 window | 3–8s (awakening) |
| Act 3 window | 8–15s (ascension) |
| Pre-drop silence | 0.06s default (0.04–0.08 range) |
| Hook zoom start | 150% |
| Hook shake peak | 100 |
