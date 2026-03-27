#!/usr/bin/env python3
"""
Ascension Engine — Cinematic Quality Re-Audit
Scores every clip in engine.db using Claude Vision (cinematic quality 0.0–1.0),
classifies into good_parts / mid_tier / victim_contrast, and updates the DB.

Scoring is done per-video (shared keyframes), then applied to all clips
from that video.

Usage:
    python3 data/reaudit_mog.py              # score all unscored clips
    python3 data/reaudit_mog.py --all        # re-score everything
    python3 data/reaudit_mog.py --dry-run    # print plan, no writes
    python3 data/reaudit_mog.py --stats      # show current split without scoring
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision_tagger import _gather_frames, _call_claude_mog_score, classify_mog_track, MOG_S_TIER, MOG_VICTIM
from engine_db import EngineDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("reaudit_mog")

ASSETS_THUMB = ROOT / "library" / "assets" / "thumbnails"

# engine.db track CHECK constraint only allows these values
DB_TRACK_MAP = {
    "good_parts":      "good_parts",
    "victim_contrast": "victim_contrast",
    "mid_tier":        "unclassified",   # mid_tier not in DB enum yet
}


def print_stats(db: EngineDB) -> None:
    rows = db.conn.execute("SELECT track, COUNT(*) FROM clips GROUP BY track").fetchall()
    total = db.conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    counts = {r[0]: r[1] for r in rows}
    print("\n" + "═" * 60)
    print("  CINEMATIC QUALITY SPLIT")
    print("═" * 60)
    print(f"  Total clips               : {total}")
    print(f"  S-tier good_parts (≥{MOG_S_TIER}) : {counts.get('good_parts', 0)}")
    print(f"  mid_tier ({MOG_VICTIM}–{MOG_S_TIER})      : {counts.get('unclassified', 0)}")
    print(f"  victim_contrast (<{MOG_VICTIM})  : {counts.get('victim_contrast', 0)}")
    print(f"  archived                  : {counts.get('archived', 0)}")
    print("═" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",     action="store_true", help="Re-score all clips")
    parser.add_argument("--dry-run", action="store_true", help="No writes")
    parser.add_argument("--stats",   action="store_true", help="Show split and exit")
    args = parser.parse_args()

    db = EngineDB()
    db.init()

    if args.stats:
        print_stats(db)
        db.close()
        return

    # Get all clips from DB
    if args.all:
        rows = db.conn.execute("SELECT clip_id, video_id, rank FROM clips ORDER BY video_id, clip_id").fetchall()
    else:
        # Only score clips at the default rank (0.5 = never scored)
        rows = db.conn.execute(
            "SELECT clip_id, video_id, rank FROM clips WHERE rank = 0.5 ORDER BY video_id, clip_id"
        ).fetchall()

    if not rows:
        log.info("No clips to score. Use --all to force re-score.")
        print_stats(db)
        db.close()
        return

    # Group clips by video_id for efficient scoring (one API call per video)
    from collections import defaultdict
    by_video: dict[str, list[str]] = defaultdict(list)
    for clip_id, video_id, _ in rows:
        by_video[video_id].append(clip_id)

    log.info("Scoring %d video(s) covering %d clips…", len(by_video), len(rows))

    scored_videos: dict[str, dict] = {}

    for vid, clip_ids in by_video.items():
        kf_dir = ASSETS_THUMB / vid
        if not kf_dir.exists():
            # Fall back to individual scene thumbnails
            frame_paths = [ASSETS_THUMB / f"{cid}.jpg" for cid in clip_ids
                           if (ASSETS_THUMB / f"{cid}.jpg").exists()]
            if not frame_paths:
                log.warning("  [SKIP] no keyframes for %s", vid[:55])
                scored_videos[vid] = {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "no frames"}
                continue
        else:
            frame_paths = _gather_frames(kf_dir, max_frames=4)
            if not frame_paths:
                log.warning("  [SKIP] empty keyframe dir for %s", vid[:55])
                scored_videos[vid] = {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "empty dir"}
                continue

        if args.dry_run:
            log.info("  [dry-run] Would score %s (%d frames, %d clips)", vid[:50], len(frame_paths), len(clip_ids))
            scored_videos[vid] = {"mog_score": 0.5, "dominant_trait": "dry_run", "notes": ""}
            continue

        log.info("  Scoring %s … (%d frames, %d clips)", vid[:50], len(frame_paths), len(clip_ids))
        result = _call_claude_mog_score(frame_paths)
        scored_videos[vid] = result
        track = classify_mog_track(result["mog_score"])
        log.info("    → %.3f (%s) — [%s]  %s",
                 result["mog_score"], result.get("dominant_trait", "?"), track,
                 result.get("notes", ""))

    # Update DB
    changed = 0
    for vid, clip_ids in by_video.items():
        result = scored_videos.get(vid)
        if not result:
            continue
        score = result["mog_score"]
        track = DB_TRACK_MAP.get(classify_mog_track(score), "unclassified")
        for clip_id in clip_ids:
            if not args.dry_run:
                db.update_clip_rank(clip_id, score)
                db.set_clip_track(clip_id, track)
            changed += 1

    if not args.dry_run:
        db.conn.commit()
        log.info("Updated %d clips in engine.db", changed)

    db.close()
    if not args.dry_run:
        db2 = EngineDB(); db2.init()
        print_stats(db2)
        db2.close()
    else:
        log.info("[dry-run] No changes written.")


if __name__ == "__main__":
    main()
