#!/usr/bin/env python3
"""
Ascension Engine — Mog Track Re-Audit
Scores every clip in clip-manifest.json using Claude Vision,
classifies into good_parts / mid_tier / victim_contrast,
updates clip-manifest.json, and updates engine.db track column.

Usage:
    python3 data/reaudit_mog.py                    # score all unscored clips
    python3 data/reaudit_mog.py --all              # re-score everything (overwrite)
    python3 data/reaudit_mog.py --dry-run          # print what would happen, no writes
    python3 data/reaudit_mog.py --stats            # show current split without scoring
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision_tagger import _gather_frames, _call_claude_mog_score, classify_mog_track, MOG_S_TIER, MOG_VICTIM
from engine_db import EngineDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("reaudit_mog")

MANIFEST_PATH = ROOT / "clip-manifest.json"
ASSETS_THUMB  = ROOT / "library" / "assets" / "thumbnails"

TRACK_LABELS = {
    "good_parts":      "S-Tier  (≥0.88)",
    "mid_tier":        "Mid     (0.75–0.87)",
    "victim_contrast": "Victim  (<0.75)",
}

# engine.db only allows: unclassified | good_parts | victim_contrast | archived
# mid_tier maps to 'unclassified' in DB until schema is extended
DB_TRACK_MAP = {
    "good_parts":      "good_parts",
    "victim_contrast": "victim_contrast",
    "mid_tier":        "unclassified",
}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        log.error("clip-manifest.json not found")
        sys.exit(1)
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(data: dict) -> None:
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))


def print_stats(clips: list[dict]) -> None:
    counts: dict[str, int] = {"good_parts": 0, "mid_tier": 0, "victim_contrast": 0, "unscored": 0}
    for clip in clips:
        score = clip.get("mog_score")
        if score is None:
            counts["unscored"] += 1
        else:
            counts[classify_mog_track(score)] += 1

    print("\n" + "═" * 60)
    print("  MOG TRACK AUDIT")
    print("═" * 60)
    print(f"  Total clips      : {len(clips)}")
    print(f"  S-Tier  (≥{MOG_S_TIER}) : {counts['good_parts']} clips → good_parts/")
    print(f"  Mid     ({MOG_VICTIM}–{MOG_S_TIER}): {counts['mid_tier']} clips → mid_tier/")
    print(f"  Victim  (<{MOG_VICTIM}) : {counts['victim_contrast']} clips → victim_contrast/")
    print(f"  Unscored         : {counts['unscored']} clips")
    print("═" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mog track re-auditor")
    parser.add_argument("--all",     action="store_true", help="Re-score all clips, not just unscored")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, no writes")
    parser.add_argument("--stats",   action="store_true", help="Show current split and exit")
    args = parser.parse_args()

    manifest = load_manifest()
    clips: list[dict] = manifest.get("clips", [])

    if args.stats:
        print_stats(clips)
        return

    # Deduplicate by source_video_id — score once per video's keyframe set,
    # apply that score to all clips from that video (they share keyframes).
    scored_videos: dict[str, dict] = {}   # video_id → mog_result
    to_score: list[dict] = []

    for clip in clips:
        vid = clip.get("source_video_id", "")
        already_has_score = clip.get("mog_score") is not None
        if not args.all and already_has_score:
            continue
        if vid not in scored_videos:
            to_score.append(clip)   # representative clip per video

    if not to_score:
        log.info("All clips already scored. Use --all to force re-score.")
        print_stats(clips)
        return

    # Identify unique videos that need scoring
    unique_vids = []
    seen_vids: set[str] = set()
    for clip in to_score:
        vid = clip.get("source_video_id", "")
        if vid not in seen_vids:
            seen_vids.add(vid)
            unique_vids.append(vid)

    log.info("Scoring %d video(s) covering %d clips …", len(unique_vids), len(clips))

    db = EngineDB()
    db.init()

    for vid in unique_vids:
        kf_dir = ASSETS_THUMB / vid
        if not kf_dir.exists():
            log.warning("  No keyframes for %s — defaulting to 0.5", vid[:50])
            scored_videos[vid] = {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "no keyframes"}
            continue

        frames = _gather_frames(kf_dir, max_frames=3)
        if not frames:
            log.warning("  No frames found in %s — defaulting to 0.5", kf_dir.name[:50])
            scored_videos[vid] = {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "no frames"}
            continue

        if args.dry_run:
            log.info("  [dry-run] Would score %s (%d frames)", vid[:50], len(frames))
            scored_videos[vid] = {"mog_score": 0.5, "dominant_trait": "dry_run", "notes": ""}
            continue

        log.info("  Scoring %s …", vid[:55])
        result = _call_claude_mog_score(frames)
        scored_videos[vid] = result
        track = classify_mog_track(result["mog_score"])
        log.info("    → %.3f (%s) — %s  [%s]",
                 result["mog_score"], result["dominant_trait"], result.get("notes", ""), track)

    # Apply scores to all clips in manifest
    changed = 0
    for clip in clips:
        vid = clip.get("source_video_id", "")
        if vid not in scored_videos:
            continue
        result = scored_videos[vid]
        old_track = clip.get("mog_track", "unscored")
        new_track = classify_mog_track(result["mog_score"])
        clip["mog_score"] = result["mog_score"]
        clip["mog_track"] = new_track
        clip["mog_notes"] = result.get("notes", "")
        if old_track != new_track:
            changed += 1

        if not args.dry_run:
            # Update engine.db track column
            db.conn.execute(
                "UPDATE clips SET track = ? WHERE clip_id = ?",
                (DB_TRACK_MAP.get(new_track, "unclassified"), clip["clip_id"])
            )

    if not args.dry_run:
        db.conn.commit()
        manifest["clips"] = clips
        save_manifest(manifest)
        log.info("Saved manifest + DB. %d clips changed track.", changed)

    db.close()
    print_stats(clips)


if __name__ == "__main__":
    main()
