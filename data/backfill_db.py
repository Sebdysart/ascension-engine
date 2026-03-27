#!/usr/bin/env python3
"""
Ascension Engine — DB Backfill
Reads clip-manifest.json and registers all clips (+ their parent ingest records)
into engine.db. Safe to re-run: INSERT OR REPLACE is idempotent.

Usage:
    python3 data/backfill_db.py
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_db import EngineDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("backfill_db")

MANIFEST = ROOT / "clip-manifest.json"
STYLE_PROFILES_DIR = ROOT / "style-profiles"


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        log.error("clip-manifest.json not found at %s", MANIFEST)
        sys.exit(1)
    data = json.loads(MANIFEST.read_text())
    return data.get("clips", [])


def load_style_profile(video_id: str) -> dict:
    p = STYLE_PROFILES_DIR / f"{video_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def main() -> None:
    db = EngineDB()
    db.init()

    clips = load_manifest()
    log.info("Backfilling %d clips from clip-manifest.json …", len(clips))

    # Group clips by video_id so we register one ingest row per video
    by_video: dict[str, list[dict]] = {}
    for clip in clips:
        vid = clip.get("source_video_id", clip.get("clip_id", "unknown"))
        by_video.setdefault(vid, []).append(clip)

    ingests_added = 0
    clips_added = 0

    for video_id, video_clips in by_video.items():
        profile = load_style_profile(video_id)
        bpm = profile.get("bpm", 0) or profile.get("audio", {}).get("bpm", 0)
        color_grade = (
            profile.get("visual_grade", {}).get("color_grade")
            or profile.get("visuals", {}).get("color_grade", "unknown")
        )
        cuts_on_beat = profile.get("cut_rhythm", {}).get("cuts_on_beat_pct", 0.0)
        creator = profile.get("source_creator", video_id.split("_")[0])

        # Derive a stable file_hash from the video_id (already contains sha256 suffix)
        file_hash = video_id  # video_id = stem + first8 of sha256 — unique enough

        # Register ingest row (idempotent)
        db.conn.execute(
            "INSERT OR REPLACE INTO ingests "
            "(video_id, source_file, file_hash, source_creator, status, "
            " bpm, color_grade, total_clips, cuts_on_beat_pct, completed_at) "
            "VALUES (?, ?, ?, ?, 'complete', ?, ?, ?, ?, datetime('now'))",
            (
                video_id,
                f"library/raw/{video_id}.mp4",
                file_hash,
                creator,
                int(bpm),
                color_grade,
                len(video_clips),
                float(cuts_on_beat),
            ),
        )
        db.conn.commit()
        ingests_added += 1

        for clip in video_clips:
            clip_id     = clip["clip_id"]
            scene_index = clip.get("scene_index", 0)
            start_sec   = clip.get("start_sec", 0.0)
            end_sec     = clip.get("end_sec", 0.0)
            duration    = clip.get("duration_sec", 0.0)
            file_path   = clip.get("file", "")
            thumbnail   = clip.get("thumbnail", "")

            db.conn.execute(
                "INSERT OR REPLACE INTO clips "
                "(clip_id, video_id, scene_index, start_sec, end_sec, "
                " duration_sec, file_path, thumbnail, rank, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    clip_id,
                    video_id,
                    scene_index,
                    float(start_sec),
                    float(end_sec),
                    float(duration),
                    file_path,
                    thumbnail,
                    float(clip.get("rank", 0.5)),
                    json.dumps(clip.get("tags", [])),
                ),
            )
            clips_added += 1

        db.conn.commit()
        log.info("  %-55s  %2d clips  BPM=%-3d  %s",
                 video_id[:55], len(video_clips), int(bpm), color_grade)

    log.info("")
    log.info("Backfill complete: %d ingest records, %d clips registered.", ingests_added, clips_added)
    db.close()


if __name__ == "__main__":
    main()
