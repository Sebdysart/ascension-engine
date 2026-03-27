#!/usr/bin/env python3
"""
Ascension Engine v2.1 — Gold Ingest Pipeline
Watches ~/Ascension-Engine/input/gold/ for new .mp4 files and runs the full
DNA ingest ritual: scene detection, keyframe extraction, audio analysis,
color profiling, clip segmentation, manifest writing, and git commit.

Usage:
    python data/ingest.py                   # watch mode (continuous)
    python data/ingest.py --dry-run         # dry run, no writes
    python data/ingest.py --setup           # print pip install command
    python data/ingest.py --once <file.mp4> # ingest a single file then exit
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── New pipeline modules ─────────────────────────────────────────────────────
# Imported lazily inside functions so the core pipeline works even if these
# modules have missing optional deps (easyocr, anthropic, etc.).

def _import_vision_tagger():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vision_tagger import tag_clips_for_video
        return tag_clips_for_video
    except Exception as e:
        log.warning("vision_tagger unavailable: %s", e)
        return None

def _import_text_processor():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from text_processor import process_text_for_video
        return process_text_for_video
    except Exception as e:
        log.warning("text_processor unavailable: %s", e)
        return None

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import imagehash
    from PIL import Image
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

try:
    from data.engine_db import EngineDB
    HAS_ENGINE_DB = True
except ImportError:
    try:
        from engine_db import EngineDB
        HAS_ENGINE_DB = True
    except ImportError:
        HAS_ENGINE_DB = False

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SEQUENCE_TEMPLATES_DIR = ROOT / "library" / "sequence_templates"
GOLD_DIR = Path.home() / "Ascension-Engine" / "input" / "gold"
LIBRARY_DIR = ROOT / "library"
CLIPS_DIR = LIBRARY_DIR / "clips"
RAW_DIR = LIBRARY_DIR / "raw"
ASSETS_VIDEO = LIBRARY_DIR / "assets" / "video"
ASSETS_AUDIO = LIBRARY_DIR / "assets" / "audio"
ASSETS_THUMB = LIBRARY_DIR / "assets" / "thumbnails"
STYLE_PROFILES_DIR = ROOT / "style-profiles"
TAGS_INDEX = LIBRARY_DIR / "tags" / "index.json"
CLIP_MANIFEST = ROOT / "clip-manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

SETUP_CMD = (
    "pip install PySceneDetect librosa imagehash watchdog scenedetect Pillow numpy"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found. Install via: brew install ffmpeg")
        sys.exit(1)

def video_id(path: Path) -> str:
    """Deterministic ID: stem + first 8 chars of SHA256."""
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return f"{path.stem}_{h}"

def ffprobe_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())

def run(cmd: list, dry_run: bool = False, **kw) -> subprocess.CompletedProcess:
    log.debug("CMD: %s", " ".join(str(c) for c in cmd))
    if dry_run:
        log.info("[dry-run] %s", " ".join(str(c) for c in cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}

def save_json(path: Path, data: dict, dry_run: bool = False) -> None:
    if dry_run:
        log.info("[dry-run] Would write %s", path)
        return
    path.write_text(json.dumps(data, indent=2))

# ── Scene Detection ───────────────────────────────────────────────────────────

def detect_scenes(video_path: Path, threshold: float = 0.32) -> list[float]:
    """
    Detect scene boundaries using PySceneDetect Python API.
    Tuned for fast TikTok cuts (0.3-1.5s) with low threshold.
    Falls back to beat-aligned cuts if scenedetect is not available.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(
            threshold=threshold,
            min_scene_len=7,  # ~0.23s at 30fps
        ))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if scene_list:
            timestamps = [scene[0].get_seconds() for scene in scene_list]
            log.info("         PySceneDetect: %d scenes (threshold=%.2f)", len(timestamps), threshold)
            return timestamps

    except ImportError:
        log.warning("scenedetect Python API not available.")
    except Exception as e:
        log.warning("Scene detection error: %s", e)

    # Fallback: use FFmpeg scene filter (works without scenedetect)
    log.info("         Using FFmpeg scene filter fallback...")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "frame=pts_time",
             "-of", "csv=p=0", "-f", "lavfi",
             f"movie={video_path},select=gt(scene\\,0.3)"],
            capture_output=True, text=True, timeout=30,
        )
        timestamps = []
        for line in result.stdout.strip().splitlines():
            try:
                timestamps.append(float(line.strip()))
            except ValueError:
                pass
        if timestamps:
            log.info("         FFmpeg scene filter: %d cuts detected", len(timestamps))
            return timestamps
    except Exception:
        pass

    # Final fallback: beat-aligned if we have audio, else fixed intervals
    log.warning("No scene detection available — using 1.0s intervals for BP-pace cuts.")
    duration = ffprobe_duration(video_path)
    return [i * 1.0 for i in range(int(duration // 1.0))]

# ── Frame & Audio Extraction ──────────────────────────────────────────────────

def extract_keyframes(
    video_path: Path, scene_times: list[float], out_dir: Path, dry_run: bool
) -> list[Path]:
    """Extract one frame per scene + one every second using FFmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []

    # Every 1s
    every_sec_out = out_dir / "every1s_%04d.jpg"
    run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", "fps=1", "-q:v", "2", str(every_sec_out)],
        dry_run=dry_run,
    )
    frames += sorted(out_dir.glob("every1s_*.jpg"))

    # One per scene boundary
    for i, t in enumerate(scene_times):
        frame_path = out_dir / f"scene_{i:04d}_{t:.2f}s.jpg"
        run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(frame_path)],
            dry_run=dry_run,
        )
        if not dry_run and frame_path.exists():
            frames.append(frame_path)

    return frames

def extract_audio(video_path: Path, out_dir: Path, dry_run: bool) -> Path:
    """Extract audio track as WAV for librosa analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{video_path.stem}.wav"
    run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
         str(wav_path)],
        dry_run=dry_run,
    )
    return wav_path

# ── Audio Analysis ────────────────────────────────────────────────────────────

def analyze_audio(wav_path: Path) -> dict:
    """Return BPM, beat locations, onset times, and peak moments via librosa."""
    if not HAS_LIBROSA:
        log.warning("librosa not installed — audio analysis skipped.")
        return {"bpm": 0, "beat_times": [], "peak_moments_sec": [], "onset_times": []}

    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Handle tempo being an ndarray (librosa >= 0.10)
    if hasattr(tempo, '__len__'):
        tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_val = float(tempo)

    # Onset detection (strong transients — phonk drops, bass hits)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # RMS energy peaks (high-energy moments)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)
    threshold = float(np.mean(rms) + 1.5 * np.std(rms))
    peak_times = [float(times[i]) for i, v in enumerate(rms) if v > threshold]

    # Beat confidence: ratio of onset-aligned beats (hybrid scoring)
    beat_confidence = 0.0
    if beat_times and onset_times:
        aligned = sum(1 for bt in beat_times if any(abs(bt - ot) < 0.1 for ot in onset_times))
        beat_confidence = round(aligned / max(len(beat_times), 1), 3)

    return {
        "bpm": int(round(tempo_val)),
        "beat_times": [round(t, 3) for t in beat_times],
        "onset_times": [round(t, 3) for t in onset_times],
        "peak_moments_sec": [round(t, 3) for t in peak_times],
        "beat_confidence": beat_confidence,
    }

# ── Color Histogram Analysis ──────────────────────────────────────────────────

def analyze_color_grade(frames: list[Path]) -> dict:
    """
    Classify rough color grade from sampled frames via FFmpeg histogram stats.
    Returns a dict with color_grade label and basic channel averages.
    """
    if not frames:
        return {"color_grade": "unknown", "avg_r": 0, "avg_g": 0, "avg_b": 0}

    sample = frames[::max(1, len(frames) // 8)][:8]
    totals = {"r": 0.0, "g": 0.0, "b": 0.0}
    count = 0

    for frame in sample:
        if not frame.exists():
            continue
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "frame_tags=lavfi.signalstats.YAVG,lavfi.signalstats.UAVG,lavfi.signalstats.VAVG",
             "-f", "lavfi",
             f"movie={frame},signalstats",
             "-print_format", "json"],
            capture_output=True, text=True,
        )
        # Simpler: use Pillow if imagehash is available
        if HAS_IMAGEHASH:
            try:
                img = Image.open(frame).convert("RGB").resize((16, 16))
                r, g, b = zip(*list(img.getdata()))
                totals["r"] += sum(r) / len(r)
                totals["g"] += sum(g) / len(g)
                totals["b"] += sum(b) / len(b)
                count += 1
            except Exception:
                pass

    if count == 0:
        return {"color_grade": "unknown", "avg_r": 0, "avg_g": 0, "avg_b": 0}

    avg_r = totals["r"] / count
    avg_g = totals["g"] / count
    avg_b = totals["b"] / count

    if avg_b > avg_r + 15:
        grade = "ColdBlue"
    elif avg_r > avg_b + 15 and avg_g > avg_b + 10:
        grade = "WarmGold"
    elif avg_r > avg_b + 10 and avg_b > avg_g - 5:
        grade = "TealOrange"
    elif max(avg_r, avg_g, avg_b) - min(avg_r, avg_g, avg_b) < 10:
        grade = "Desaturated"
    else:
        grade = "Neutral"

    return {
        "color_grade": grade,
        "avg_r": round(avg_r, 1),
        "avg_g": round(avg_g, 1),
        "avg_b": round(avg_b, 1),
    }

# ── Clip Segmentation ─────────────────────────────────────────────────────────

def export_clip_segments(
    video_path: Path,
    scene_times: list[float],
    vid_id: str,
    out_dir: Path,
    dry_run: bool,
) -> list[dict]:
    """
    Export clean MP4 per scene. Returns list of clip metadata dicts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = ffprobe_duration(video_path)
    clips = []

    boundaries = scene_times + [duration]
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_dur = round(end - start, 3)
        if seg_dur < 0.5:
            continue

        clip_id = f"{vid_id}_scene{i:03d}"
        out_file = out_dir / f"{clip_id}.mp4"

        run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(video_path),
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                str(out_file),
            ],
            dry_run=dry_run,
        )

        thumb_path = ASSETS_THUMB / f"{clip_id}.jpg"
        run(
            ["ffmpeg", "-y", "-ss", str(start + seg_dur / 2),
             "-i", str(video_path), "-frames:v", "1", "-q:v", "3",
             str(thumb_path)],
            dry_run=dry_run,
        )

        # Perceptual hash for deduplication
        phash = ""
        if HAS_IMAGEHASH and not dry_run and thumb_path.exists():
            try:
                phash = str(imagehash.phash(Image.open(thumb_path)))
            except Exception:
                pass

        clips.append({
            "clip_id": clip_id,
            "source_video_id": vid_id,
            "scene_index": i,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": seg_dur,
            "file": str(out_file.relative_to(ROOT)),
            "thumbnail": str(thumb_path.relative_to(ROOT)),
            "tags": [],
            "rank": 0.5,
            "phash": phash,
            "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
        })

    return clips

# ── Style Profile Builder ─────────────────────────────────────────────────────

def compute_beat_aligned_cuts(
    scene_times: list[float],
    beat_times: list[float],
    tolerance_ms: float = 80.0,
) -> list[dict]:
    """For each scene cut, find nearest beat and compute offset."""
    cut_points = []
    tol_sec = tolerance_ms / 1000.0

    for i, cut_time in enumerate(scene_times):
        nearest_beat = None
        offset_ms = 999.0
        on_beat = False

        if beat_times:
            diffs = [(abs(cut_time - bt), bt) for bt in beat_times]
            min_diff, nearest_beat = min(diffs, key=lambda x: x[0])
            offset_ms = round(min_diff * 1000, 1)
            on_beat = min_diff <= tol_sec

        # Duration to next cut
        if i + 1 < len(scene_times):
            dur = round(scene_times[i + 1] - cut_time, 3)
        else:
            dur = 0.0

        cut_points.append({
            "time_sec": round(cut_time, 3),
            "duration_sec": dur,
            "on_beat": on_beat,
            "beat_offset_ms": offset_ms,
            "tags": [],
        })

    return cut_points


def build_style_profile(
    vid_id: str,
    scene_times: list[float],
    audio_data: dict,
    color_data: dict,
    duration: float,
    source_creator: str = "unknown",
) -> dict:
    """Build brutal BP blueprint from extracted DNA."""
    beat_times = audio_data.get("beat_times", [])
    cut_points = compute_beat_aligned_cuts(scene_times, beat_times)

    cut_lengths = [cp["duration_sec"] for cp in cut_points if cp["duration_sec"] > 0]
    avg_cut = round(statistics.mean(cut_lengths), 3) if cut_lengths else 0.0

    on_beat_count = sum(1 for cp in cut_points if cp["on_beat"])
    on_beat_pct = round(on_beat_count / max(len(cut_points), 1), 3)
    avg_offset = round(
        statistics.mean([cp["beat_offset_ms"] for cp in cut_points]) if cut_points else 0, 1
    )

    return {
        "video_id": vid_id,
        "source_creator": source_creator,
        "total_duration_sec": round(duration, 2),
        "bpm": audio_data.get("bpm", 0),
        "beat_times_sec": beat_times,
        "cut_rhythm": {
            "avg_cut_sec": avg_cut,
            "total_cuts": len(scene_times),
            "cuts_on_beat_pct": on_beat_pct,
            "beat_offset_avg_ms": avg_offset,
            "cut_points": cut_points,
        },
        "visual_grade": {
            "color_grade": color_data.get("color_grade", "unknown"),
            "description": f"avg_r={color_data.get('avg_r', 0)}, avg_g={color_data.get('avg_g', 0)}, avg_b={color_data.get('avg_b', 0)}",
        },
        "audio": {
            "bpm": audio_data.get("bpm", 0),
            "beat_times_sec": beat_times,
            "onset_times_sec": audio_data.get("onset_times", []),
            "peak_moments_sec": audio_data.get("peak_moments_sec", []),
            "energy_profile": "phonk" if audio_data.get("bpm", 0) >= 120 else "atmospheric",
        },
        "motion_fx": {
            "zoom_punch": True,
            "zoom_easing": "cubic-bezier(0.25,0.1,0.25,1)",
            "impact_hold_ms": 120,
        },
        "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ── Library Asset Builders ────────────────────────────────────────────────────

SEQUENCE_TEMPLATES_DIR = LIBRARY_DIR / "sequence_templates"
GRADE_PRESETS_DIR = LIBRARY_DIR / "grade_presets"
BLUEPRINTS_DIR = LIBRARY_DIR / "blueprints"


def build_sequence_template(vid_id, clips, profile, source_creator):
    """Extract reusable timeline template from ingested video."""
    beat_times = profile.get("beat_times_sec", [])
    peak_moments = set(round(p, 1) for p in profile.get("audio", {}).get("peak_moments_sec", []))

    slots = []
    for i, clip in enumerate(clips):
        start = clip["start_sec"]
        dur = clip["duration_sec"]
        on_beat = any(abs(start - bt) < 0.08 for bt in beat_times) if beat_times else False
        is_impact = any(abs(start - p) < 0.2 for p in peak_moments)

        slots.append({
            "index": i, "start_sec": round(start, 3), "duration_sec": round(dur, 3),
            "on_beat": on_beat, "is_impact": is_impact,
            "original_clip_id": clip["clip_id"], "preferred_tags": clip.get("tags", []),
        })

    cut_lengths = [s["duration_sec"] for s in slots if s["duration_sec"] > 0]
    avg_cut = round(statistics.mean(cut_lengths), 3) if cut_lengths else 0
    on_beat_count = sum(1 for s in slots if s["on_beat"])

    return {
        "template_id": f"seq_{vid_id}", "source_video_id": vid_id,
        "source_creator": source_creator, "bpm": profile.get("bpm", 0),
        "total_duration_sec": profile.get("total_duration_sec", 0),
        "total_slots": len(slots), "avg_cut_sec": avg_cut,
        "cuts_on_beat_pct": round(on_beat_count / max(len(slots), 1), 3),
        "color_grade": profile.get("visual_grade", {}).get("color_grade", "unknown"),
        "slots": slots, "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_grade_preset(vid_id, color_data, profile):
    """Generate reusable grade preset from color analysis."""
    grade = color_data.get("color_grade", "unknown")
    css = {"Desaturated": "saturate(0.15) contrast(1.25) brightness(0.95)",
           "ColdBlue": "saturate(0.9) hue-rotate(20deg) brightness(0.85) contrast(1.2)",
           "WarmGold": "saturate(1.2) sepia(0.3) brightness(1.05)",
           "TealOrange": "saturate(1.3) hue-rotate(-10deg) contrast(1.15)",
           "Neutral": "saturate(1.05) contrast(1.08) brightness(1.0)"}
    return {
        "preset_id": f"grade_{vid_id}", "name": f"{grade}_from_{vid_id[:20]}",
        "css_filter": css.get(grade, "none"),
        "avg_r": color_data.get("avg_r", 0), "avg_g": color_data.get("avg_g", 0),
        "avg_b": color_data.get("avg_b", 0), "source_videos": [vid_id],
        "description": f"From {profile.get('source_creator', 'unknown')} — {grade}",
    }


# ── Tags Index Update ─────────────────────────────────────────────────────────

def update_tags_index(clips: list[dict], dry_run: bool) -> None:
    """Append new clip_ids to relevant tag buckets in library/tags/index.json."""
    index = load_json(TAGS_INDEX)
    if "tags" not in index:
        index["tags"] = {}

    for clip in clips:
        for tag in clip.get("tags", []):
            if tag not in index["tags"]:
                index["tags"][tag] = []
            if clip["clip_id"] not in index["tags"][tag]:
                index["tags"][tag].append(clip["clip_id"])

    index["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_json(TAGS_INDEX, index, dry_run)

# ── Manifest ──────────────────────────────────────────────────────────────────

def update_clip_manifest(clips: list[dict], dry_run: bool) -> None:
    """Append new clips to clip-manifest.json."""
    manifest = load_json(CLIP_MANIFEST)
    if "clips" not in manifest:
        manifest["clips"] = []
        manifest["version"] = "1.0"

    existing_ids = {c["clip_id"] for c in manifest["clips"]}
    new_clips = [c for c in clips if c["clip_id"] not in existing_ids]
    manifest["clips"].extend(new_clips)
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    manifest["total_clips"] = len(manifest["clips"])
    save_json(CLIP_MANIFEST, manifest, dry_run)
    log.info("Manifest: %d total clips (+%d new)", manifest["total_clips"], len(new_clips))

# ── Git Commit ────────────────────────────────────────────────────────────────

def git_commit(vid_id: str, dry_run: bool) -> None:
    """Stage all changes and create a git commit."""
    git = shutil.which("git")
    if not git:
        log.warning("git not found — skipping commit.")
        return

    try:
        run([git, "-C", str(ROOT), "add", "-A"], dry_run=dry_run)
        run(
            [git, "-C", str(ROOT), "commit", "-m", f"Gold ingest: {vid_id}"],
            dry_run=dry_run,
        )
        log.info("Git commit: Gold ingest: %s", vid_id)
    except subprocess.CalledProcessError as e:
        log.warning("Git commit failed (nothing to commit?): %s", e.stderr.strip())

# ── Summary Report ────────────────────────────────────────────────────────────

def print_summary(vid_id: str, clips: list[dict], profile: dict, duration: float) -> None:
    cr = profile.get("cut_rhythm", {})
    audio = profile.get("audio", {})
    grade = profile.get("visual_grade", {}).get("color_grade", "unknown")
    print("\n" + "═" * 60)
    print(f"  ✅  INGEST COMPLETE — {vid_id}")
    print("═" * 60)
    print(f"  Duration      : {duration:.1f}s")
    print(f"  Scenes        : {len(clips)}")
    print(f"  BPM           : {audio.get('bpm', 0)}")
    print(f"  Color Grade   : {grade}")
    print(f"  Avg Cut       : {cr.get('avg_cut_sec', 0)}s")
    print(f"  Cuts on Beat  : {cr.get('cuts_on_beat_pct', 0) * 100:.0f}%")
    print(f"  Onsets        : {len(audio.get('onset_times_sec', []))}")
    print("═" * 60 + "\n")

# ── Sequence Template Writer ──────────────────────────────────────────────────

def write_sequence_template(
    vid_id: str,
    clips: list[dict],
    scene_times: list[float],
    audio_data: dict,
    profile: dict,
    dry_run: bool = False,
) -> None:
    """
    Write a structured sequence template to library/sequence_templates/<vid_id>.json.
    Captures clip order, durations, beat alignment points, and transition types.
    """
    beat_times = audio_data.get("beat_times", [])

    def _transition_type(clip: dict) -> str:
        """Classify transition based on clip duration."""
        d = clip.get("duration_sec", 2.0)
        if d < 1.2:
            return "hard_cut"
        if d > 3.5:
            return "hold"
        return "cut"

    def _nearest_beat(t: float) -> float | None:
        if not beat_times:
            return None
        nearest = min(beat_times, key=lambda b: abs(b - t))
        return round(nearest, 3) if abs(nearest - t) < 0.5 else None

    clip_order = []
    for i, clip in enumerate(clips):
        start = clip.get("start_sec", 0.0)
        entry = {
            "index": i,
            "clip_id": clip["clip_id"],
            "start_sec": start,
            "end_sec": clip.get("end_sec", start + clip.get("duration_sec", 0)),
            "duration_sec": clip.get("duration_sec", 0),
            "tags": clip.get("tags", []),
            "beat_anchor_sec": _nearest_beat(start),
            "transition_type": _transition_type(clip),
            "rank": clip.get("rank", 0.5),
        }
        clip_order.append(entry)

    template = {
        "vid_id": vid_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_clips": len(clips),
        "total_duration_sec": clips[-1]["end_sec"] if clips else 0,
        "bpm": audio_data.get("bpm", 0),
        "avg_cut_length_sec": profile.get("cut_rhythm", {}).get("avg_cut_length_sec", 0),
        "beat_alignment": profile.get("audio", {}).get("beat_cut_alignment", "unknown"),
        "color_grade": profile.get("visuals", {}).get("color_grade", "unknown"),
        "hook_pacing_sec": profile.get("cut_rhythm", {}).get("hook_pacing_sec", [0, 3]),
        "clip_order": clip_order,
        "beat_drop_points_sec": audio_data.get("peak_moments_sec", [])[:5],
        "transition_summary": {
            "hard_cut": sum(1 for c in clip_order if c["transition_type"] == "hard_cut"),
            "cut": sum(1 for c in clip_order if c["transition_type"] == "cut"),
            "hold": sum(1 for c in clip_order if c["transition_type"] == "hold"),
        },
    }

    if dry_run:
        log.info("[dry-run] Would write sequence template for %s", vid_id)
        return

    SEQUENCE_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SEQUENCE_TEMPLATES_DIR / f"{vid_id}.json"
    out_path.write_text(json.dumps(template, indent=2))
    log.info("   Sequence template saved → %s", out_path.name)


# ── Core Ingest Ritual ────────────────────────────────────────────────────────

def ingest_video(video_path: Path, dry_run: bool = False) -> None:
    """Run the full ingest ritual for a single .mp4 file. Idempotent via engine DB."""
    log.info("▶  Ingesting: %s", video_path.name)
    require_ffmpeg()

    # Idempotency check via engine DB
    file_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()[:16]
    db = None
    if HAS_ENGINE_DB and not dry_run:
        try:
            db = EngineDB()
            db.init()
            if db.is_ingested(file_hash):
                log.info("   ⏩ Already ingested (hash %s). Skipping.", file_hash)
                db.close()
                return
        except Exception as e:
            log.warning("Engine DB error: %s — continuing without DB tracking", e)
            db = None

    ensure_dirs(CLIPS_DIR, RAW_DIR, ASSETS_VIDEO, ASSETS_AUDIO,
                ASSETS_THUMB, STYLE_PROFILES_DIR, SEQUENCE_TEMPLATES_DIR)

    vid_id = video_id(video_path)
    log.info("   Video ID: %s", vid_id)

    # Register in engine DB
    creator = video_path.stem.split("_")[0] if "_" in video_path.stem else "unknown"
    if db:
        db.start_ingest(vid_id, str(video_path), file_hash, creator)

    # Copy to raw library
    raw_copy = RAW_DIR / video_path.name
    if not dry_run and not raw_copy.exists():
        shutil.copy2(video_path, raw_copy)

    # 1. Scene detection
    log.info("   [1/9] Scene detection …")
    scene_times = detect_scenes(video_path, threshold=0.4)
    log.info("         %d scenes detected", len(scene_times))

    # 2. Keyframe extraction
    log.info("   [2/9] Extracting keyframes …")
    frames_dir = ASSETS_THUMB / vid_id
    frames = extract_keyframes(video_path, scene_times, frames_dir, dry_run)
    log.info("         %d frames extracted", len(frames))

    # 3. Text detection + removal
    log.info("   [3/9] Text detection + OCR template extraction …")
    _text_proc = _import_text_processor()
    if _text_proc and (dry_run or frames_dir.exists()):
        try:
            _text_proc(vid_id, frames_dir, dry_run)
        except Exception as e:
            log.warning("Text processor failed (non-fatal): %s", e)
    else:
        log.info("         Text detection skipped.")

    # 4. Audio extraction + analysis
    log.info("   [4/9] Audio extraction + BPM analysis …")
    wav_path = extract_audio(video_path, ASSETS_AUDIO, dry_run)
    if not dry_run and wav_path.exists():
        audio_data = analyze_audio(wav_path)
    else:
        audio_data = {"bpm": 0, "beat_times": [], "peak_moments_sec": []}
    log.info("         BPM: %d  |  %d beat markers", audio_data["bpm"], len(audio_data.get("beat_times", [])))

    # 5. Color analysis
    log.info("   [5/9] Color histogram analysis …")
    color_data = analyze_color_grade(frames)
    log.info("         Grade: %s  (R%.0f G%.0f B%.0f)",
             color_data["color_grade"], color_data.get("avg_r", 0),
             color_data.get("avg_g", 0), color_data.get("avg_b", 0))

    # 6. Style profile
    log.info("   [6/9] Generating style-profile.json …")
    duration = ffprobe_duration(video_path)
    creator = video_path.stem.split("_")[0] if "_" in video_path.stem else "unknown"
    profile = build_style_profile(vid_id, scene_times, audio_data, color_data, duration, creator)
    profile_path = STYLE_PROFILES_DIR / f"{vid_id}.json"
    save_json(profile_path, profile, dry_run)
    log.info("         Saved → %s", profile_path.name)

    # 7. Clip segmentation
    log.info("   [7/9] Exporting scene clips …")
    clips = export_clip_segments(video_path, scene_times, vid_id, CLIPS_DIR, dry_run)
    log.info("         %d clips exported", len(clips))

    # 8. Vision tagging
    log.info("   [8/9] Vision tagging via Claude …")
    _tagger = _import_vision_tagger()
    if _tagger:
        try:
            clips = _tagger(vid_id, clips, ASSETS_THUMB, dry_run)
        except Exception as e:
            log.warning("Vision tagger failed (non-fatal): %s", e)
    else:
        log.info("         Vision tagging skipped (anthropic not available or no API key).")

    # 9. Sequence template
    log.info("   [9/9] Writing sequence template …")
    try:
        write_sequence_template(vid_id, clips, scene_times, audio_data, profile, dry_run)
    except Exception as e:
        log.warning("Sequence template failed (non-fatal): %s", e)

    # Update manifest + tags index
    update_clip_manifest(clips, dry_run)
    update_tags_index(clips, dry_run)

    # Library assets: sequence template, grade preset, blueprint
    ensure_dirs(SEQUENCE_TEMPLATES_DIR, GRADE_PRESETS_DIR, BLUEPRINTS_DIR)
    seq = build_sequence_template(vid_id, clips, profile, creator)
    save_json(SEQUENCE_TEMPLATES_DIR / f"seq_{vid_id}.json", seq, dry_run)
    log.info("         Sequence template: %d slots → seq_%s.json", len(seq["slots"]), vid_id[:30])
    save_json(GRADE_PRESETS_DIR / f"grade_{vid_id}.json", build_grade_preset(vid_id, color_data, profile), dry_run)
    save_json(BLUEPRINTS_DIR / f"bp_{vid_id}.json", profile, dry_run)

    # Git commit
    git_commit(vid_id, dry_run)

    # Mark complete in engine DB + register clips
    if db:
        try:
            for clip in clips:
                db.add_clip(
                    clip["clip_id"], vid_id, clip["scene_index"],
                    clip["start_sec"], clip["end_sec"], clip["duration_sec"],
                    clip["file"], clip["thumbnail"], clip.get("phash", "")
                )
                if clip.get("mog_score") is not None:
                    db.update_clip_rank(clip["clip_id"], clip["mog_score"])
                if clip.get("mog_track"):
                    db.set_clip_track(clip["clip_id"], clip["mog_track"])
            db.conn.commit()
            db.update_ingest_status(vid_id, "complete",
                bpm=audio_data.get("bpm", 0),
                color_grade=color_data.get("color_grade", ""),
                total_clips=len(clips),
                cuts_on_beat_pct=profile.get("cut_rhythm", {}).get("cuts_on_beat_pct", 0))
            db.close()
        except Exception as e:
            log.warning("Engine DB finalize error: %s", e)

    # Summary
    print_summary(vid_id, clips, profile, duration)

# ── Watchdog Handler ──────────────────────────────────────────────────────────

if HAS_WATCHDOG:
    class GoldFolderHandler(FileSystemEventHandler):
        def __init__(self, dry_run: bool):
            self.dry_run = dry_run
            self._processing: set[str] = set()

        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() != ".mp4":
                return
            if str(path) in self._processing:
                return
            self._processing.add(str(path))
            log.info("🎬 New file detected: %s", path.name)
            # Brief delay to ensure file is fully written
            time.sleep(2)
            try:
                ingest_video(path, self.dry_run)
            except Exception as e:
                log.error("Ingest failed for %s: %s", path.name, e)
            finally:
                self._processing.discard(str(path))

def watch_mode(dry_run: bool = False) -> None:
    """Watch gold folder for new .mp4 files using watchdog."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    if HAS_WATCHDOG:
        log.info("👁  Watching %s (watchdog)", GOLD_DIR)
        handler = GoldFolderHandler(dry_run)
        observer = Observer()
        observer.schedule(handler, str(GOLD_DIR), recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        log.info("👁  Watching %s (polling every 5s)", GOLD_DIR)
        seen: set[str] = set()
        while True:
            try:
                for f in GOLD_DIR.glob("*.mp4"):
                    if str(f) not in seen:
                        seen.add(str(f))
                        log.info("🎬 New file detected: %s", f.name)
                        time.sleep(2)
                        try:
                            ingest_video(f, dry_run)
                        except Exception as e:
                            log.error("Ingest failed for %s: %s", f.name, e)
                time.sleep(5)
            except KeyboardInterrupt:
                break

# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ascension Engine v2.1 — Gold Ingest Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview all operations without writing any files."
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Print the pip install command for all optional dependencies."
    )
    parser.add_argument(
        "--once", metavar="FILE",
        help="Ingest a single .mp4 file and exit."
    )
    args = parser.parse_args()

    if args.setup:
        print(f"\n  One-time setup:\n\n    {SETUP_CMD}\n")
        sys.exit(0)

    missing = []
    if not HAS_LIBROSA:
        missing.append("librosa")
    if not HAS_IMAGEHASH:
        missing.append("imagehash / Pillow")
    if not HAS_WATCHDOG:
        missing.append("watchdog")
    if missing:
        log.warning(
            "Optional deps not installed: %s\n"
            "  Run: %s",
            ", ".join(missing), SETUP_CMD,
        )

    if args.dry_run:
        log.info("🔍 DRY RUN — no files will be written.")

    if args.once:
        target = Path(args.once).expanduser().resolve()
        if not target.exists():
            log.error("File not found: %s", target)
            sys.exit(1)
        ingest_video(target, args.dry_run)
    else:
        watch_mode(args.dry_run)


if __name__ == "__main__":
    main()
