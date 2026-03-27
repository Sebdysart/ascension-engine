# Narrative Intelligence Layer — Design Doc

**Date:** 2026-03-27
**Status:** Approved — implementing

---

## Problem

The current Ascension Engine assigns clips via flat greedy selection: clips are picked by BPM and tag matching, then laid out in beat-grid slots with no understanding of psychological arc. Victim clips can appear anywhere. There is no hook formula for the first 15 frames. Visual effects (shake, zoom pulse, grade shift) are absent. The result is technically on-beat but emotionally inert.

---

## Goal

Add a narrative intelligence layer that gives every edit a three-act MogNarrative structure with psychological progression, correct victim clip placement, a scroll-stopping hook, and reactive visual effects.

---

## Architecture

Six components. Three new Python modules, two updates (sequencer + generate_batch), one new Remotion composition.

```
data/narrative_engine.py     ← three-act arc + phonk sync map + pre-drop silence
data/hook_generator.py       ← scroll-stopping hook spec for frames 1-15
data/narrative_tagger.py     ← classify existing library clips by narrative role
data/sequencer.py            ← extend with act-aware slot assignment
data/generate_batch.py       ← wire narrative arc into generation
remotion/compositions/
  BrutalBeatMontage.tsx      ← new composition with shake/zoom/grade-shift effects
library/tags/index.json      ← updated with narrative role tags
```

---

## Three-Act MogNarrative

### Act 1 — "Victim" (0–3s)
- ~1 cut/sec (matches verse section in sequencer)
- Warm color grade (+40% brightness / WarmGold)
- Clip selection: `victim_contrast/` pool, lowest cinematic score clips
- Downward eye contact, low camera angle preferred
- Every victim clip must be followed by an angle inversion (low→high or high→low)

### Act 2 — "Awakening" (3–8s)
- 8–12 cuts/sec (buildup section, rapid half-time beats)
- Desaturation ramps across the act
- Shake effect intensity builds from 0→60
- Color grade shifts warm → neutral
- Clip selection: mid-tier pool, motion clips

### Act 3 — "Ascension" (8–15s)
- Cold grade (-30% brightness / Desaturated/cold)
- Direct stare clips, jawline reveal on the drop
- Reveal hierarchy: eyes → jawline → frame/shoulders → full walk
- Variable pacing with 1-2 strategic slow-mo frames (0.5x duration)
- Clip selection: `good_parts/` S-tier only

### Critical Victim Placement Rules
- Victim clips: Act 1 ONLY + 2 beats before each phonk drop
- NEVER in Act 3
- Next cut after every victim clip = angle inversion

---

## Phonk Sync Map

```
0–3s    → intro/verse   → Act 1 (Victim)
3–8s    → build         → Act 2 (Awakening)
8–11s   → drop          → Act 3 head (Ascension)
11–13s  → drop cont.    → Act 3 peak
13–15s  → outro         → Act 3 close (full walk / frame out)
```

Pre-drop silence: 0.04–0.08s audio gap inserted before every detected drop.

---

## Hook Formula (frames 1–15)

1. Select "flaw clip" — highest brightness + softest lighting from `victim_contrast/`
2. Apply zoom-out: 150%→100% over 0.5s
3. On first bass hit:
   - Shake spike: 30→100→0 over 0.2s
   - Zoom pulse: 100%→115%→100% over 0.3s
   - Grade shift: warm→cold
4. Return as `HookSpec` JSON for sequencer consumption

---

## BrutalBeatMontage.tsx (new composition)

Visual effects absent from MogEdit.tsx that are required:

| Effect | Spec |
|--------|------|
| Shake | Spring-animated position offset, `intensity` param 0–100 |
| Zoom pulse | scale 1.0→1.15→1.0 over 0.3s on bass drop |
| Color grade shift | CSS filter interpolation warm(5600K)→cold(4200K) across acts |
| Pre-drop hold | Single-frame freeze 0.04–0.08s before drop |
| Act-aware grade | WarmGold in Act 1, neutral ramp in Act 2, cold/Desaturated in Act 3 |

Props: `slots` (with act label), `hookSpec`, `beatTimings`, `bpm`.

---

## Narrative Tagger

Classifies existing library clips by narrative role using existing `mog_score`, `mog_track`, tags, and brightness data. No new Claude vision calls needed — pure derived classification from existing metadata.

| Role | Criteria |
|------|----------|
| `victim_act1` | mog_track=victim_contrast + no direct_stare tag |
| `awakening_transition` | mog_track=mid_tier + motion tag |
| `ascension_reveal` | mog_track=good_parts + direct stare/dark grade |
| `jawline_pop` | tight crop + high contrast lower face |
| `full_walk` | motion toward camera + full body |

Writes `narrative_role` tag into `library/tags/index.json`.

---

## Sequencer Changes

Current `Slot` dataclass gains:
- `act: Literal["victim", "awakening", "ascension"]`
- `narrative_role: str`
- `pre_drop_silence: float` (0.0 or 0.04–0.08)
- `is_victim_slot: bool`
- `angle_inversion_required: bool`

`build_narrative_sequence()` replaces flat `build_sequence()` as the primary entry point for narrative-aware generation. Old `build_sequence()` stays for backwards compat.

---

## Generate Batch Changes

`generate_from_template()` gains a `use_narrative=True` flag. When enabled:
1. Calls `build_narrative_sequence()` instead of flat beat detection
2. Assigns clips from act-appropriate pools
3. Inserts pre-drop silence gaps via FFmpeg filter
4. Reports per-act assignment in the result dict

---

## File Map (read before editing)

| File | Role |
|------|------|
| `data/sequencer.py` | Beat-grid builder; extend with act-aware slots |
| `data/generate_batch.py` | Orchestrator; wire narrative arc |
| `data/vision_tagger.py` | Already has mog_score + mog_track |
| `data/engine_db.py` | SQLite; clips.track CHECK constraint = unclassified/good_parts/victim_contrast/archived |
| `library/tags/index.json` | Tags index; add narrative_role tags here |
| `remotion/compositions/MogEdit.tsx` | Soft archetype — do not modify |
| `remotion/Root.tsx` | Register new BrutalBeatMontage composition here |

---

## Key Constants

| Constant | Value |
|----------|-------|
| Act 1 window | 0–3s |
| Act 2 window | 3–8s |
| Act 3 window | 8–15s |
| Pre-drop silence | 0.04–0.08s |
| Hook zoom start | 150% |
| Hook shake peak | 100 |
| Ascension grade | cold (-30% brightness) |
| Victim grade | warm (+40% brightness) |
| Slow-mo frames | 1–2 per edit, Act 3 only |
