#!/usr/bin/env python3
"""
Ascension Engine — sequencer.py
Beat-aligned Edit Decision List (EDL) assembler.

Algorithm
---------
1. Template selection: pick the sequence template whose BPM is within ±5 of
   the detected audio BPM. If multiple match, prefer the one whose section
   distribution (drop/verse ratio) most closely mirrors the audio analysis.

2. Greedy clip assignment across beat-grid slots:
   score(clip, slot) = clip.impact_sum - angle_penalty
   angle_penalty = 0.3 if consecutive face_angle diff < 30° (same direction)

3. Transition selection per slot section type:
   drop    → "glitch"
   buildup → "zoom_in_1.2x"
   verse   → "hard_cut"

Output EDL structure:
  {
    "tempo":      float,
    "beat_grid":  [...],
    "template_id": str,
    "edl": [
      {
        "slot":           int,
        "clip_path":      str,
        "start_frame":    int,
        "duration_frames": int,
        "cut_time":       float,  -- adjusted_time from beat grid
        "section":        str,
        "transition":     str,
        "impact_score":   float,
        "face_angle":     float | None,
      },
      ...
    ]
  }

Usage:
    python3 data/sequencer.py \\
        --clips library/clips/ \\
        --audio input/gold/bp_masterx.mp4 \\
        --templates library/sequence_templates/ \\
        --output out/edl_bpmax5.json \\
        --target-duration 15
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("sequencer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

ANGLE_PENALTY  = 0.30   # applied when consecutive face_angle diff < 30°
ANGLE_DIFF_MIN = 30.0   # degrees — below this triggers penalty
FPS_DEFAULT    = 30.0
TRANSITION_MAP = {
    "drop":    "glitch",
    "buildup": "zoom_in_1.2x",
    "verse":   "hard_cut",
}


# ── Template loading ──────────────────────────────────────────────────────────

def load_templates(templates_dir: str) -> list[dict]:
    """Load all sequence templates from a directory."""
    templates = []
    for f in sorted(Path(templates_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text())
            d["_file"] = str(f)
            d["_name"] = f.stem
            templates.append(d)
        except Exception as e:
            log.warning("Could not load template %s: %s", f.name, e)
    log.info("Loaded %d templates from %s", len(templates), templates_dir)
    return templates


def select_template(templates: list[dict], audio_analysis: dict) -> Optional[dict]:
    """
    Pick best-matching template for audio_analysis.

    Selection criteria (in order):
    1. BPM within ±5 of detected tempo.
    2. Among those, prefer template whose drop_ratio most closely matches
       audio drop ratio. If no exact BPM match, fall back to ±10, then ±20.
    """
    if not templates:
        return None

    detected_bpm = audio_analysis.get("tempo", 0.0)
    audio_drop_ratio = (
        audio_analysis["stats"]["drop_sec"] /
        max(audio_analysis["target_duration"], 1.0)
    )

    def bpm_diff(t: dict) -> float:
        return abs(t.get("bpm", 0) - detected_bpm)

    def drop_diff(t: dict) -> float:
        # Estimate template drop ratio from color_grade proxy if no section data
        grade = t.get("color_grade", "").lower()
        if grade in ("tealorange", "teal_orange"):
            tpl_drop = 0.3
        elif grade in ("warmgold", "warm_gold"):
            tpl_drop = 0.4
        else:
            tpl_drop = 0.15
        return abs(tpl_drop - audio_drop_ratio)

    # Try increasingly wide BPM windows
    for tolerance in (5, 10, 20, 999):
        candidates = [t for t in templates if bpm_diff(t) <= tolerance and t.get("bpm", 0) > 0]
        if candidates:
            best = min(candidates, key=lambda t: (bpm_diff(t), drop_diff(t)))
            log.info("Template selected: %s  (BPM=%s, tolerance=±%d)", best.get("_name", "?"), best.get("bpm"), tolerance)
            return best

    # Last resort: pick highest-BPM template
    valid = [t for t in templates if t.get("bpm", 0) > 0]
    if valid:
        best = max(valid, key=lambda t: t.get("bpm", 0))
        log.info("Template fallback (highest BPM): %s", best.get("_name", "?"))
        return best

    log.warning("No usable template found — using first available")
    return templates[0]


# ── Scored-clip helpers ───────────────────────────────────────────────────────

def load_scored_clips(clip_scores: list[dict]) -> list[dict]:
    """
    Normalise scored-clip dicts from impact_scorer.score_clip_frames().
    Accepts raw score dicts or simple {video_path, summary} entries.
    Filters out clips with errors.
    """
    valid = []
    for sc in clip_scores:
        if "error" in sc:
            log.warning("Skipping errored clip: %s", sc.get("video_path", "?"))
            continue
        summary = sc.get("summary", {})
        valid.append({
            "path":         sc["video_path"],
            "impact_mean":  summary.get("impact_mean", 0.0),
            "impact_max":   summary.get("impact_max", 0.0),
            "impact_p90":   summary.get("impact_p90", 0.0),
            "face_fill":    summary.get("face_fill_mean", 0.0),
            "face_detected":summary.get("face_detected", False),
            # Use first face_angle from the raw list if available
            "face_angle":   (sc.get("face_angles") or [None])[0],
            # Duration: use fps_source and frame_count to reconstruct
            "fps":          sc.get("fps_source", FPS_DEFAULT),
            "frame_count_total": sc.get("frame_count", 0) * round(sc.get("fps_source", FPS_DEFAULT) / sc.get("fps_processed", sc.get("fps_source", FPS_DEFAULT))),
        })

    log.info("Valid scored clips: %d", len(valid))
    return valid


# ── Greedy EDL assembler ──────────────────────────────────────────────────────

def assemble_edit(
    scored_clips: list[dict],
    audio_analysis: dict,
    templates: list[dict],
    target_duration: float = 15.0,
) -> dict:
    """
    Build an EDL from scored clips, audio analysis, and sequence templates.

    Parameters
    ----------
    scored_clips     : output of load_scored_clips()
    audio_analysis   : output of audio_analyzer.analyze_phonk_track()
    templates        : loaded sequence template dicts
    target_duration  : edit target length in seconds

    Returns
    -------
    EDL dict (see module docstring for full schema).
    """
    if not scored_clips:
        raise ValueError("No valid scored clips provided")

    beat_grid = audio_analysis.get("beat_grid", [])
    if not beat_grid:
        raise ValueError("Audio analysis contains no beat grid")

    # Select template
    template = select_template(templates, audio_analysis)
    template_id = template.get("_name", "unknown") if template else "unknown"

    # Derive beats_per_cut from template or audio
    # beats_per_cut: how many beats per clip slot
    template_dur = template.get("total_duration_sec", 0) if template else 0
    template_clips = template.get("total_clips", 0) if template else 0
    tempo = audio_analysis.get("tempo", 100.0)
    beat_interval = 60.0 / max(tempo, 1.0)

    if template_dur > 0 and template_clips > 0:
        avg_template_cut = template_dur / template_clips
        beats_per_cut = max(1, round(avg_template_cut / beat_interval))
    else:
        # Fallback: 2 beats per cut (1.2-1.4s at 90-120 BPM)
        beats_per_cut = 2

    log.info("Assembling EDL: BPM=%.1f  beats_per_cut=%d  template=%s",
             tempo, beats_per_cut, template_id)

    # Build slots from beat grid (every N beats)
    slots: list[dict] = []
    for i in range(0, len(beat_grid) - beats_per_cut, beats_per_cut):
        start_beat = beat_grid[i]
        end_beat   = beat_grid[i + beats_per_cut]
        duration   = end_beat["adjusted_time"] - start_beat["adjusted_time"]
        if duration <= 0.05:
            continue
        slots.append({
            "slot":        len(slots),
            "cut_time":    start_beat["adjusted_time"],
            "duration":    duration,
            "section":     start_beat["section_type"],
            "beat_num":    start_beat["beat_num"],
        })
        if slots[-1]["cut_time"] + duration >= target_duration:
            break

    if not slots:
        raise ValueError("No EDL slots could be built from beat grid")

    log.info("Slots to fill: %d  (%.1fs total)", len(slots), sum(s["duration"] for s in slots))

    # Sort clips by impact_p90 descending for greedy selection
    sorted_clips = sorted(scored_clips, key=lambda c: c["impact_p90"], reverse=True)

    # Greedy assignment: pick best remaining clip for each slot
    # Angle penalty: if last assigned clip's face_angle is within 30° of candidate
    edl_entries: list[dict] = []
    prev_angle: Optional[float] = None

    for slot in slots:
        best_clip    = None
        best_score   = -1.0
        best_start_f = 0

        fps = FPS_DEFAULT
        slot_dur_frames = max(1, round(slot["duration"] * fps))

        for clip in sorted_clips:
            clip_fps    = clip.get("fps", FPS_DEFAULT)
            total_f     = clip.get("frame_count_total", 0)
            available_f = max(0, total_f - slot_dur_frames)

            # Best window start: use middle of clip or 0 if short
            start_f = available_f // 2

            # Base score: impact_p90
            score = clip["impact_p90"]

            # Angle penalty
            if prev_angle is not None and clip["face_angle"] is not None:
                angle_diff = abs(clip["face_angle"] - prev_angle)
                if angle_diff < ANGLE_DIFF_MIN:
                    score -= ANGLE_PENALTY

            if score > best_score:
                best_score  = score
                best_clip   = clip
                best_start_f = start_f
                fps          = clip_fps

        if best_clip is None:
            best_clip    = sorted_clips[slot["slot"] % len(sorted_clips)]
            best_start_f = 0
            fps          = best_clip.get("fps", FPS_DEFAULT)

        prev_angle = best_clip.get("face_angle")
        transition = TRANSITION_MAP.get(slot["section"], "hard_cut")

        edl_entries.append({
            "slot":            slot["slot"],
            "clip_path":       best_clip["path"],
            "start_frame":     best_start_f,
            "duration_frames": round(slot["duration"] * fps),
            "cut_time":        round(slot["cut_time"], 4),
            "section":         slot["section"],
            "transition":      transition,
            "impact_score":    round(best_score, 4),
            "face_angle":      best_clip.get("face_angle"),
        })

    total_edl_dur = sum(e["duration_frames"] / FPS_DEFAULT for e in edl_entries)
    log.info("EDL complete: %d entries  %.2fs  template=%s", len(edl_entries), total_edl_dur, template_id)

    return {
        "tempo":        round(tempo, 2),
        "beat_grid":    audio_analysis["beat_grid"],
        "template_id":  template_id,
        "target_duration": target_duration,
        "total_duration":  round(total_edl_dur, 3),
        "fps":             FPS_DEFAULT,
        "sections":        audio_analysis.get("sections", []),
        "edl":             edl_entries,
        "stats": {
            "slot_count":       len(edl_entries),
            "beats_per_cut":    beats_per_cut,
            "avg_cut_sec":      round(total_edl_dur / max(len(edl_entries), 1), 3),
            "transitions": {
                t: sum(1 for e in edl_entries if e["transition"] == t)
                for t in set(e["transition"] for e in edl_entries)
            },
        },
    }


# ── EDL printer ───────────────────────────────────────────────────────────────

def print_edl(edl: dict) -> None:
    TRANS_ICON = {"glitch": "⚡", "zoom_in_1.2x": "🔍", "hard_cut": "✂"}
    print(f"\n{'═'*62}")
    print(f"  EDL  template={edl['template_id']}  BPM={edl['tempo']}  dur={edl['total_duration']:.2f}s")
    print(f"  {edl['stats']['slot_count']} slots  avg_cut={edl['stats']['avg_cut_sec']:.3f}s  beats_per_cut={edl['stats']['beats_per_cut']}")
    print(f"{'═'*62}")
    for e in edl["edl"]:
        icon  = TRANS_ICON.get(e["transition"], "·")
        angle = f"{e['face_angle']:.0f}°" if e["face_angle"] is not None else "N/A"
        dur_s = e["duration_frames"] / edl["fps"]
        print(f"  {icon} [{e['slot']:2d}] t={e['cut_time']:5.2f}s  {dur_s:.2f}s  {e['section']:8s}  "
              f"impact={e['impact_score']:.3f}  angle={angle:5s}  "
              f"{Path(e['clip_path']).name[:28]}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ascension Engine — sequencer.py")
    parser.add_argument("--clips",     required=True, help="Dir of impact-scored .json files, or a single JSON score file, or a dir of .mp4 clips to score inline")
    parser.add_argument("--audio",     required=True, help="Audio/video source for beat analysis")
    parser.add_argument("--templates", required=True, help="Dir containing sequence template JSONs")
    parser.add_argument("--output",    default="out/edl.json", help="Output EDL JSON path")
    parser.add_argument("--target-duration", "-t", type=float, default=15.0)
    parser.add_argument("--verbose",   "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import inline to allow standalone use
    from data.audio_analyzer import analyze_phonk_track  # type: ignore
    from data.impact_scorer import score_clip_frames, score_clip_batch  # type: ignore

    # Load clips
    clips_path = Path(args.clips)
    if clips_path.is_dir():
        jsons = list(clips_path.glob("*.json"))
        if jsons:
            clip_scores = []
            for j in jsons:
                try:
                    clip_scores.append(json.loads(j.read_text()))
                except Exception:
                    pass
            log.info("Loaded %d pre-scored clip JSONs", len(clip_scores))
        else:
            # Score MP4s inline
            mp4s = sorted(clips_path.glob("*.mp4"))
            log.info("Scoring %d clips inline ...", len(mp4s))
            clip_scores = score_clip_batch([str(m) for m in mp4s])
    elif clips_path.suffix == ".json":
        data = json.loads(clips_path.read_text())
        clip_scores = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"--clips must be a directory or JSON file: {clips_path}")

    # Audio analysis
    log.info("Analysing audio: %s", args.audio)
    audio_analysis = analyze_phonk_track(args.audio, args.target_duration)

    # Templates
    templates = load_templates(args.templates)

    # Scored clips
    scored = load_scored_clips(clip_scores)

    # Assemble EDL
    edl = assemble_edit(scored, audio_analysis, templates, args.target_duration)

    print_edl(edl)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(edl, indent=2))
    log.info("EDL written: %s", out_path)


if __name__ == "__main__":
    main()
