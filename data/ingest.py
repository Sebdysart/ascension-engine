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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
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

def detect_scenes(video_path: Path, threshold: float = 0.4) -> list[float]:
    """
    Run PySceneDetect and return a list of scene-start timestamps in seconds.
    Falls back to fixed 3s intervals if scenedetect is not installed.
    """
    if shutil.which("scenedetect") is None:
        log.warning("scenedetect not found — using 3s fallback intervals.")
        duration = ffprobe_duration(video_path)
        return [i * 3.0 for i in range(int(duration // 3))]

    result = subprocess.run(
        [
            "scenedetect",
            "--input", str(video_path),
            "--output", str(video_path.parent),
            "detect-content",
            "--threshold", str(threshold),
            "list-scenes",
            "--output", str(video_path.parent),
            "--filename", f"{video_path.stem}-scenes.csv",
            "--no-output-file",   # print to stdout
        ],
        capture_output=True, text=True,
    )
    timestamps: list[float] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 3 and parts[0].strip().isdigit():
            try:
                timestamps.append(float(parts[3].strip()))
            except (IndexError, ValueError):
                pass
    if not timestamps:
        log.warning("Scene detection returned no timestamps — using 3s fallback.")
        duration = ffprobe_duration(video_path)
        timestamps = [i * 3.0 for i in range(int(duration // 3))]
    return timestamps

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
    """Return BPM, beat locations, and peak moments via librosa."""
    if not HAS_LIBROSA:
        log.warning("librosa not installed — audio analysis skipped.")
        return {"bpm": 0, "beat_times": [], "peak_moments_sec": []}

    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # RMS energy peaks
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)
    threshold = float(np.mean(rms) + 1.5 * np.std(rms))
    peak_times = [float(times[i]) for i, v in enumerate(rms) if v > threshold]

    return {
        "bpm": int(round(float(tempo))),
        "beat_times": [round(t, 3) for t in beat_times],
        "peak_moments_sec": [round(t, 3) for t in peak_times],
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
            "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
        })

    return clips

# ── Style Profile Builder ─────────────────────────────────────────────────────

def build_style_profile(
    vid_id: str,
    scene_times: list[float],
    audio_data: dict,
    color_data: dict,
    duration: float,
    source_creator: str = "unknown",
) -> dict:
    """Assemble the style-profile.json schema."""
    cut_lengths = []
    if len(scene_times) > 1:
        cut_lengths = [
            round(scene_times[i + 1] - scene_times[i], 3)
            for i in range(len(scene_times) - 1)
        ]

    avg_cut = round(statistics.mean(cut_lengths), 3) if cut_lengths else 0.0
    std_dev = round(statistics.stdev(cut_lengths), 3) if len(cut_lengths) > 1 else 0.0
    cuts_per_15 = int(round(15 / avg_cut)) if avg_cut > 0 else 0

    hook_end = scene_times[2] if len(scene_times) > 2 else min(3.0, duration)

    beat_cut_alignment = "unknown"
    if audio_data.get("beat_times") and cut_lengths:
        aligned = sum(
            1 for ct in scene_times
            if any(abs(ct - bt) < 0.15 for bt in audio_data["beat_times"])
        )
        ratio = aligned / len(scene_times) if scene_times else 0
        beat_cut_alignment = "tight" if ratio > 0.6 else "loose" if ratio > 0.3 else "free"

    return {
        "video_id": vid_id,
        "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_creator": source_creator,
        "cut_rhythm": {
            "avg_cut_length_sec": avg_cut,
            "std_dev_sec": std_dev,
            "cuts_per_15s": cuts_per_15,
            "hook_pacing_sec": [0, round(hook_end, 2)],
            "zoom_points_sec": [],
        },
        "visuals": {
            "color_grade": color_data.get("color_grade", "unknown"),
            "blur_zoom_patterns": "unanalyzed",
            "motion_types_frequency": {},
        },
        "text": {
            "caption_density": 0.0,
            "font_style": "unanalyzed",
            "casing": "unanalyzed",
            "animation": "unanalyzed",
            "tikTok_config": {"position": "bottom-third", "max_chars_per_line": 18},
        },
        "audio": {
            "bpm": audio_data.get("bpm", 0),
            "beat_cut_alignment": beat_cut_alignment,
            "vo_music_balance": {"music": 0.0, "vo": 0.0},
            "peak_moments_sec": audio_data.get("peak_moments_sec", []),
        },
        "hook_style": "unanalyzed",
        "critique": "",
        "user_notes": None,
        "performance_link": None,
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
    print("\n" + "═" * 60)
    print(f"  ✅  INGEST COMPLETE — {vid_id}")
    print("═" * 60)
    print(f"  Duration      : {duration:.1f}s")
    print(f"  Scenes        : {len(clips)}")
    print(f"  BPM           : {profile['audio']['bpm']}")
    print(f"  Color Grade   : {profile['visuals']['color_grade']}")
    print(f"  Avg Cut       : {profile['cut_rhythm']['avg_cut_length_sec']}s")
    print(f"  Beat Alignment: {profile['audio']['beat_cut_alignment']}")
    print(f"  Peak Moments  : {profile['audio']['peak_moments_sec'][:5]}")
    print("═" * 60 + "\n")

# ── Core Ingest Ritual ────────────────────────────────────────────────────────

def ingest_video(video_path: Path, dry_run: bool = False) -> None:
    """Run the full ingest ritual for a single .mp4 file."""
    log.info("▶  Ingesting: %s", video_path.name)
    require_ffmpeg()

    ensure_dirs(CLIPS_DIR, RAW_DIR, ASSETS_VIDEO, ASSETS_AUDIO,
                ASSETS_THUMB, STYLE_PROFILES_DIR)

    vid_id = video_id(video_path)
    log.info("   Video ID: %s", vid_id)

    # Copy to raw library
    raw_copy = RAW_DIR / video_path.name
    if not dry_run and not raw_copy.exists():
        shutil.copy2(video_path, raw_copy)

    # 1. Scene detection
    log.info("   [1/6] Scene detection …")
    scene_times = detect_scenes(video_path, threshold=0.4)
    log.info("         %d scenes detected", len(scene_times))

    # 2. Keyframe extraction
    log.info("   [2/6] Extracting keyframes …")
    frames_dir = ASSETS_THUMB / vid_id
    frames = extract_keyframes(video_path, scene_times, frames_dir, dry_run)
    log.info("         %d frames extracted", len(frames))

    # 3. Audio extraction + analysis
    log.info("   [3/6] Audio extraction + BPM analysis …")
    wav_path = extract_audio(video_path, ASSETS_AUDIO, dry_run)
    if not dry_run and wav_path.exists():
        audio_data = analyze_audio(wav_path)
    else:
        audio_data = {"bpm": 0, "beat_times": [], "peak_moments_sec": []}
    log.info("         BPM: %d  |  %d beat markers", audio_data["bpm"], len(audio_data.get("beat_times", [])))

    # 4. Color analysis
    log.info("   [4/6] Color histogram analysis …")
    color_data = analyze_color_grade(frames)
    log.info("         Grade: %s  (R%.0f G%.0f B%.0f)",
             color_data["color_grade"], color_data.get("avg_r", 0),
             color_data.get("avg_g", 0), color_data.get("avg_b", 0))

    # 5. Style profile
    log.info("   [5/6] Generating style-profile.json …")
    duration = ffprobe_duration(video_path)
    creator = video_path.stem.split("_")[0] if "_" in video_path.stem else "unknown"
    profile = build_style_profile(vid_id, scene_times, audio_data, color_data, duration, creator)
    profile_path = STYLE_PROFILES_DIR / f"{vid_id}.json"
    save_json(profile_path, profile, dry_run)
    log.info("         Saved → %s", profile_path.name)

    # 6. Clip segmentation
    log.info("   [6/6] Exporting scene clips …")
    clips = export_clip_segments(video_path, scene_times, vid_id, CLIPS_DIR, dry_run)
    log.info("         %d clips exported", len(clips))

    # Update manifest + tags index
    update_clip_manifest(clips, dry_run)
    update_tags_index(clips, dry_run)

    # Git commit
    git_commit(vid_id, dry_run)

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
