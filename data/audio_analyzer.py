#!/usr/bin/env python3
"""
Ascension Engine — audio_analyzer.py
Phonk/trap audio analysis for beat-aligned edit assembly.

Outputs
-------
tempo           : float   — detected BPM
beat_times      : list[float] — onset-corrected beat timestamps (sec)
onset_times     : list[float] — all onset peak timestamps (sec)
onset_envelope  : list[float] — full onset strength envelope (sampled at hop_length)
sections        : list[dict]  — {start, end, type: "drop"|"buildup"|"verse", micro_offset_ms}
beat_grid       : list[dict]  — {time, beat_num, section_type, micro_offset_ms, adjusted_time}

Beat-grid micro-offsets (spec-exact):
  drop    : +15 to +35ms (cuts land slightly late — feel of hit)
  buildup : -50 to -80ms (pre-empt the drop)
  verse   : 0ms

Section detection heuristic:
  RMS window = 2s.  Spectral centroid window = 2s.
  high RMS (>p75) AND high centroid (>p60) → drop
  rising RMS (diff > 0.5*std) AND mid centroid → buildup
  else → verse

Usage:
    python3 data/audio_analyzer.py input/gold/bp_masterx.mp4
    python3 data/audio_analyzer.py track.wav --target-duration 15 --json out/analysis.json
    python3 data/audio_analyzer.py track.mp4 --plot   # ASCII beat grid
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

log = logging.getLogger("audio_analyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

SR = 22050
HOP_LENGTH = 512   # ~23ms at 22050 Hz


# ── Audio extraction ─────────────────────────────────────────────────────────

def _extract_audio(source_path: Path, target_duration: float) -> tuple[np.ndarray, int]:
    """
    Load audio from an MP4/WAV/MP3 source, truncated to target_duration.
    Returns (y, sr).
    """
    suffix = source_path.suffix.lower()
    if suffix in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        y, sr = librosa.load(str(source_path), sr=SR, duration=target_duration + 5, mono=True)
    else:
        # Video: extract with ffmpeg first
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(source_path),
                 "-ac", "1", "-ar", str(SR), "-vn",
                 "-t", str(target_duration + 5),
                 wav_path],
                capture_output=True, check=True,
            )
            y, sr = librosa.load(wav_path, sr=SR, duration=target_duration + 5, mono=True)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
    return y, sr


# ── Beat tracking ─────────────────────────────────────────────────────────────

def _beat_track(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """
    Detect tempo and beat frames using librosa with tightness=100.
    Returns (tempo_bpm, beat_times_sec).
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP_LENGTH, tightness=100, units="time"
    )
    bpm = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
    return bpm, np.array(beat_frames, dtype=float)


# ── Onset detection ───────────────────────────────────────────────────────────

def _onset_detect(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute onset strength envelope and pick onset peaks.
    Returns (onset_envelope, onset_times_sec).
    peak_pick params: delta=0.5 (spec-exact)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    # Normalise envelope 0-1
    env_max = onset_env.max()
    if env_max > 0:
        onset_env_norm = onset_env / env_max
    else:
        onset_env_norm = onset_env.copy()

    # Spec: librosa.util.peak_pick(delta=0.5)
    peak_frames = librosa.util.peak_pick(
        onset_env_norm,
        pre_max=3, post_max=3,
        pre_avg=3, post_avg=5,
        delta=0.5, wait=10,
    )
    onset_times = librosa.frames_to_time(peak_frames, sr=sr, hop_length=HOP_LENGTH)
    return onset_env_norm, onset_times


# ── Section detection ─────────────────────────────────────────────────────────

def _detect_sections(y: np.ndarray, sr: int, total_duration: float) -> list[dict]:
    """
    Segment audio into drop / buildup / verse sections.

    Heuristic (2s windows):
      high RMS (>p75) AND high centroid (>p60) → drop
      rising RMS (diff > 0.5*std) AND mid centroid → buildup
      else → verse

    Returns list of {start, end, type, micro_offset_ms}.
    """
    window_sec = 2.0
    hop_sec    = 0.5
    window_n   = int(window_sec * sr)
    hop_n      = int(hop_sec * sr)

    times, rms_vals, centroid_vals = [], [], []

    pos = 0
    while pos + window_n <= len(y):
        chunk = y[pos:pos + window_n]
        times.append(pos / sr)
        rms_vals.append(float(np.sqrt(np.mean(chunk ** 2))))
        spec = np.abs(librosa.stft(chunk, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        centroid = float(np.sum(freqs[:, None] * spec, axis=0).mean() / (spec.sum(axis=0).mean() + 1e-8))
        centroid_vals.append(centroid)
        pos += hop_n

    if not rms_vals:
        return [{"start": 0.0, "end": total_duration, "type": "verse", "micro_offset_ms": 0}]

    rms_arr  = np.array(rms_vals)
    cent_arr = np.array(centroid_vals)

    rms_p75  = float(np.percentile(rms_arr, 75))
    cent_p60 = float(np.percentile(cent_arr, 60))
    rms_std  = float(np.std(rms_arr))

    raw_labels = []
    for i, (t, rms, cent) in enumerate(zip(times, rms_arr, cent_arr)):
        if rms > rms_p75 and cent > cent_p60:
            lbl = "drop"
        elif i > 0 and (rms - rms_arr[i - 1]) > 0.5 * rms_std and cent <= cent_p60:
            lbl = "buildup"
        else:
            lbl = "verse"
        raw_labels.append(lbl)

    # Merge consecutive same-type windows into sections
    micro_offsets = {"drop": 25, "buildup": -65, "verse": 0}  # midpoints of spec ranges

    sections: list[dict] = []
    cur_type  = raw_labels[0]
    cur_start = times[0]

    for i in range(1, len(raw_labels)):
        if raw_labels[i] != cur_type:
            sections.append({
                "start": round(cur_start, 3),
                "end":   round(times[i], 3),
                "type":  cur_type,
                "micro_offset_ms": micro_offsets[cur_type],
            })
            cur_type  = raw_labels[i]
            cur_start = times[i]

    sections.append({
        "start": round(cur_start, 3),
        "end":   round(total_duration, 3),
        "type":  cur_type,
        "micro_offset_ms": micro_offsets[cur_type],
    })

    return sections


# ── Beat grid construction ────────────────────────────────────────────────────

def _build_beat_grid(
    beat_times: np.ndarray,
    sections: list[dict],
    target_duration: float,
) -> list[dict]:
    """
    For each beat in [0, target_duration], annotate with section type and
    micro-offset. adjusted_time = time + micro_offset_ms/1000.

    Micro-offset ranges (spec-exact):
      drop    : +15 to +35ms  — interpolated by position within section
      buildup : -50 to -80ms  — interpolated by position within section
      verse   : 0ms
    """
    def section_for_time(t: float) -> dict:
        for s in sections:
            if s["start"] <= t < s["end"]:
                return s
        return sections[-1]

    def micro_offset(t: float, sec: dict) -> float:
        """Returns offset in seconds, interpolated across section."""
        if sec["type"] == "drop":
            # +15ms at start → +35ms at end (anticipation builds)
            sec_len = max(sec["end"] - sec["start"], 0.001)
            alpha   = min((t - sec["start"]) / sec_len, 1.0)
            return (15.0 + alpha * 20.0) / 1000.0
        elif sec["type"] == "buildup":
            # -50ms at start → -80ms at end (increasingly eager)
            sec_len = max(sec["end"] - sec["start"], 0.001)
            alpha   = min((t - sec["start"]) / sec_len, 1.0)
            return -(50.0 + alpha * 30.0) / 1000.0
        return 0.0

    grid = []
    for i, t in enumerate(beat_times):
        if t > target_duration:
            break
        sec = section_for_time(float(t))
        off = micro_offset(float(t), sec)
        grid.append({
            "time":           round(float(t), 4),
            "beat_num":       i + 1,
            "section_type":   sec["type"],
            "micro_offset_ms": round(off * 1000, 1),
            "adjusted_time":  round(float(t) + off, 4),
        })
    return grid


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_phonk_track(audio_path: str, target_duration: float = 15.0) -> dict:
    """
    Full analysis of a phonk/trap track.

    Parameters
    ----------
    audio_path      : path to video or audio file
    target_duration : how many seconds of audio to analyse (default 15s)

    Returns
    -------
    dict with keys:
        source          : str
        target_duration : float
        actual_duration : float
        tempo           : float   — BPM
        beat_times      : list[float]
        onset_times     : list[float]
        onset_envelope  : list[float]  — normalised 0-1, at HOP_LENGTH resolution
        sections        : list[dict]   — {start, end, type, micro_offset_ms}
        beat_grid       : list[dict]   — {time, beat_num, section_type, micro_offset_ms, adjusted_time}
        stats           : dict         — beat_count, onset_count, drop_sec, buildup_sec, verse_sec
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio source not found: {path}")

    log.info("Loading audio: %s  (target %.1fs)", path.name, target_duration)
    y, sr = _extract_audio(path, target_duration)

    actual_duration = float(len(y)) / sr
    log.info("  Loaded %.2fs of audio at %dHz (%d samples)", actual_duration, sr, len(y))

    # Trim to target
    y_target = y[:int(target_duration * sr)]

    log.info("  Beat tracking (tightness=100) ...")
    tempo, beat_times = _beat_track(y_target, sr)
    log.info("  BPM=%.1f  beats=%d", tempo, len(beat_times))

    log.info("  Onset detection (delta=0.5) ...")
    onset_env, onset_times = _onset_detect(y_target, sr)
    log.info("  Onsets detected: %d", len(onset_times))

    log.info("  Section detection ...")
    sections = _detect_sections(y_target, sr, min(actual_duration, target_duration))
    for s in sections:
        log.info("    [%.1f–%.1f]  %-10s  micro=%+.0fms", s["start"], s["end"], s["type"], s["micro_offset_ms"])

    log.info("  Building beat grid ...")
    beat_grid = _build_beat_grid(beat_times, sections, target_duration)

    # Stats
    drop_sec    = sum(s["end"] - s["start"] for s in sections if s["type"] == "drop")
    buildup_sec = sum(s["end"] - s["start"] for s in sections if s["type"] == "buildup")
    verse_sec   = sum(s["end"] - s["start"] for s in sections if s["type"] == "verse")

    return {
        "source":          str(path),
        "target_duration": target_duration,
        "actual_duration": round(actual_duration, 3),
        "tempo":           round(tempo, 2),
        "beat_times":      [round(float(t), 4) for t in beat_times if t <= target_duration],
        "onset_times":     [round(float(t), 4) for t in onset_times if t <= target_duration],
        "onset_envelope":  [round(float(v), 4) for v in onset_env],
        "sections":        sections,
        "beat_grid":       beat_grid,
        "stats": {
            "beat_count":   len([t for t in beat_times if t <= target_duration]),
            "onset_count":  len([t for t in onset_times if t <= target_duration]),
            "drop_sec":     round(drop_sec, 2),
            "buildup_sec":  round(buildup_sec, 2),
            "verse_sec":    round(verse_sec, 2),
            "avg_beat_interval_sec": round(60.0 / tempo, 4) if tempo > 0 else 0.0,
        },
    }


# ── ASCII beat grid visualiser ────────────────────────────────────────────────

def _print_beat_grid(analysis: dict) -> None:
    grid  = analysis["beat_grid"]
    secs  = analysis["sections"]

    print(f"\n  BPM={analysis['tempo']:.1f}  beats={analysis['stats']['beat_count']}  onsets={analysis['stats']['onset_count']}")
    print(f"  Sections: drop={analysis['stats']['drop_sec']:.1f}s  buildup={analysis['stats']['buildup_sec']:.1f}s  verse={analysis['stats']['verse_sec']:.1f}s\n")

    ICONS = {"drop": "█", "buildup": "▲", "verse": "·"}
    cols  = 40
    total = analysis["target_duration"]

    print("  Beat grid (█=drop ▲=buildup ·=verse):")
    line  = [" "] * cols
    ticks = [" "] * cols

    for g in grid:
        col = int(g["time"] / total * cols)
        col = min(col, cols - 1)
        line[col] = ICONS.get(g["section_type"], "?")
        if g["beat_num"] % 4 == 1:
            ticks[col] = "|"

    print("  [" + "".join(line) + "]")
    print("  [" + "".join(ticks) + "]  (| = bar start)")
    print()

    # Section timeline
    print("  Sections:")
    for s in secs:
        bar_start = int(s["start"] / total * cols)
        bar_end   = int(s["end"]   / total * cols)
        bar       = "─" * max(bar_end - bar_start, 1)
        print(f"    {bar_start:2d}-{bar_end:2d}  {s['type']:8s}  {bar}  {s['start']:.1f}–{s['end']:.1f}s  offset={s['micro_offset_ms']:+.0f}ms")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ascension Engine — audio_analyzer.py")
    parser.add_argument("source", help="Audio or video file to analyse")
    parser.add_argument("--target-duration", "-t", type=float, default=15.0)
    parser.add_argument("--json", metavar="FILE", help="Write analysis to JSON")
    parser.add_argument("--plot", action="store_true", help="Print ASCII beat grid")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = analyze_phonk_track(args.source, args.target_duration)

    print(f"\n{'═'*60}")
    print(f"  AUDIO ANALYSIS: {Path(args.source).name}")
    print(f"{'═'*60}")
    print(f"  BPM          : {result['tempo']:.2f}")
    print(f"  Duration     : {result['actual_duration']:.2f}s  (target {result['target_duration']:.0f}s)")
    print(f"  Beats        : {result['stats']['beat_count']}  (interval {result['stats']['avg_beat_interval_sec']:.3f}s)")
    print(f"  Onsets       : {result['stats']['onset_count']}")
    print(f"  Sections     : {len(result['sections'])} total")
    print(f"    drop       : {result['stats']['drop_sec']:.1f}s")
    print(f"    buildup    : {result['stats']['buildup_sec']:.1f}s")
    print(f"    verse      : {result['stats']['verse_sec']:.1f}s")

    print(f"\n  Beat grid (first 16):")
    for g in result["beat_grid"][:16]:
        print(f"    beat {g['beat_num']:3d}  t={g['time']:.3f}s  adj={g['adjusted_time']:.3f}s  "
              f"section={g['section_type']:8s}  offset={g['micro_offset_ms']:+.1f}ms")

    if args.plot:
        _print_beat_grid(result)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        log.info("Written: %s", out)


if __name__ == "__main__":
    main()
