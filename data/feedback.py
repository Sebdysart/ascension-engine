#!/usr/bin/env python3
"""
Ascension Engine v2.1 — Analytics Feedback Loop
──────────────────────────────────────────────────────────────────────────────
Reads performance data from analytics.db and writes rank adjustments
back to clip-manifest.json. Clips used in high-performing videos get
promoted; clips in flops get demoted.

Usage:
  python3 data/feedback.py                    # Run full feedback loop
  python3 data/feedback.py --dry-run          # Preview changes only
  python3 data/feedback.py --report           # Print clip performance report
  python3 data/feedback.py --seed-analytics   # Insert sample data for testing
"""

import argparse
import json
import logging
import math
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feedback")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "analytics.db"
MANIFEST_PATH = ROOT / "clip-manifest.json"

# ── Config ────────────────────────────────────────────────────────────────────

PROMOTE_THRESHOLD = 0.55   # avg_watch_pct above this = promote clips
DEMOTE_THRESHOLD = 0.30    # below this = demote
PROMOTE_DELTA = 0.08       # rank boost per positive signal
DEMOTE_DELTA = -0.05       # rank penalty per negative signal
MAX_RANK = 0.98            # ceiling
MIN_RANK = 0.05            # floor
LOOKBACK_DAYS = 30


# ── Manifest I/O ──────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"version": "1.0", "clips": [], "last_updated": "", "total_clips": 0}
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest: dict, dry_run: bool = False) -> None:
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    manifest["total_clips"] = len(manifest["clips"])
    if dry_run:
        log.info("[DRY RUN] Would save manifest with %d clips", len(manifest["clips"]))
        return
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest saved: %d clips", len(manifest["clips"]))


# ── DB helpers ────────────────────────────────────────────────────────────────

def connect_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        log.error("Database not found: %s", DB_PATH)
        log.error("Run: sqlite3 %s < %s/data/schema.sql", DB_PATH, ROOT)
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def fetch_video_performance(conn: sqlite3.Connection, days: int = LOOKBACK_DAYS) -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    cur = conn.execute("""
        SELECT video_id, hook_type, archetype, color_grade, avg_watch_pct,
               ctr_thumbnail, views_7d, sentiment_score, file_path, notes
        FROM videos
        WHERE posted_at >= ?
          AND avg_watch_pct IS NOT NULL
        ORDER BY avg_watch_pct DESC
    """, (cutoff,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── Clip-to-video mapping ────────────────────────────────────────────────────

def map_clips_to_performance(manifest: dict, videos: list[dict]) -> dict[str, list[dict]]:
    """
    Map each clip_id to the videos it was used in.
    Uses source_video_id match — when a video is posted, its video_id
    in analytics should match the source_video_id or contain the clip prefix.
    """
    clip_perf: dict[str, list[dict]] = {}

    for clip in manifest.get("clips", []):
        clip_id = clip["clip_id"]
        source_vid = clip.get("source_video_id", "")

        matched = [v for v in videos if source_vid and source_vid in (v.get("video_id", ""))]

        if matched:
            clip_perf[clip_id] = matched

    return clip_perf


# ── Rank adjustment ───────────────────────────────────────────────────────────

def compute_rank_adjustments(
    manifest: dict,
    clip_perf: dict[str, list[dict]],
) -> list[dict]:
    """
    Compute rank delta for each clip based on video performance.
    Returns list of {clip_id, old_rank, new_rank, delta, reason, videos_used}.
    """
    adjustments = []
    clip_map = {c["clip_id"]: c for c in manifest.get("clips", [])}

    for clip_id, videos in clip_perf.items():
        clip = clip_map.get(clip_id)
        if not clip:
            continue

        old_rank = clip["rank"]
        avg_watch = sum(v["avg_watch_pct"] for v in videos) / len(videos)
        best_watch = max(v["avg_watch_pct"] for v in videos)
        avg_views = sum(v.get("views_7d", 0) for v in videos) / len(videos)

        delta = 0.0
        reasons = []

        if avg_watch >= PROMOTE_THRESHOLD:
            delta += PROMOTE_DELTA
            reasons.append(f"avg_watch={avg_watch:.1%}")
        elif avg_watch < DEMOTE_THRESHOLD:
            delta += DEMOTE_DELTA
            reasons.append(f"avg_watch={avg_watch:.1%} (below {DEMOTE_THRESHOLD:.0%})")

        if best_watch >= 0.70:
            delta += PROMOTE_DELTA * 0.5
            reasons.append(f"peak_watch={best_watch:.1%}")

        if avg_views >= 100_000:
            delta += PROMOTE_DELTA * 0.3
            reasons.append(f"avg_views={avg_views:.0f}")

        if delta == 0:
            continue

        new_rank = max(MIN_RANK, min(MAX_RANK, old_rank + delta))
        delta_actual = new_rank - old_rank

        if abs(delta_actual) < 0.001:
            continue

        adjustments.append({
            "clip_id": clip_id,
            "old_rank": round(old_rank, 4),
            "new_rank": round(new_rank, 4),
            "delta": round(delta_actual, 4),
            "reason": "; ".join(reasons),
            "videos_matched": len(videos),
        })

    return adjustments


def apply_adjustments(manifest: dict, adjustments: list[dict], dry_run: bool = False) -> int:
    """Apply rank adjustments to manifest clips."""
    clip_map = {c["clip_id"]: c for c in manifest.get("clips", [])}
    applied = 0

    for adj in adjustments:
        clip = clip_map.get(adj["clip_id"])
        if not clip:
            continue
        if not dry_run:
            clip["rank"] = adj["new_rank"]
        applied += 1

    return applied


# ── Seed analytics for testing ────────────────────────────────────────────────

def seed_analytics(conn: sqlite3.Connection) -> None:
    """Insert sample video performance data for testing the feedback loop."""
    archetypes = ["glow_up", "frame_maxxing", "skin_maxxing", "style_maxxing"]
    hooks = ["text_punch", "face_reveal", "before_after", "silent_stare"]
    grades = ["teal_orange", "cold_blue", "warm_gold", "desaturated"]

    now = datetime.now()

    samples = []
    for i in range(12):
        vid_id = f"test_vid_{i:03d}"
        posted = (now - timedelta(days=random.randint(0, 13))).isoformat()
        watch_pct = round(random.uniform(0.20, 0.75), 4)
        ctr = round(random.uniform(0.03, 0.12), 4)
        views = random.randint(5000, 500000)
        sentiment = round(random.uniform(-0.3, 0.8), 3)

        samples.append((
            vid_id, f"Test video {i}", hooks[i % len(hooks)],
            archetypes[i % len(archetypes)], round(random.uniform(1.4, 3.0), 1),
            grades[i % len(grades)], random.randint(90, 165), "tiktok",
            posted, views, watch_pct, ctr, sentiment,
            "exploit" if i % 3 != 0 else "explore", "2.1",
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO videos (
            video_id, title, hook_type, archetype, cut_rate, color_grade,
            music_bpm, platform, posted_at, views_7d, avg_watch_pct,
            ctr_thumbnail, sentiment_score, experiment_flag, style_profile_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, samples)
    conn.commit()
    log.info("Seeded %d test videos into analytics.db", len(samples))


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(adjustments: list[dict], videos: list[dict]) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  FEEDBACK LOOP — RANK ADJUSTMENT REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)
    print(f"  Videos with analytics : {len(videos)}")
    print(f"  Clips to adjust       : {len(adjustments)}")

    if adjustments:
        promoted = [a for a in adjustments if a["delta"] > 0]
        demoted = [a for a in adjustments if a["delta"] < 0]
        print(f"  Promoted (↑)          : {len(promoted)}")
        print(f"  Demoted (↓)           : {len(demoted)}")
        print(sep)

        for adj in sorted(adjustments, key=lambda a: a["delta"], reverse=True):
            arrow = "↑" if adj["delta"] > 0 else "↓"
            print(f"  {arrow} {adj['clip_id']}")
            print(f"    rank: {adj['old_rank']:.3f} → {adj['new_rank']:.3f} ({adj['delta']:+.4f})")
            print(f"    reason: {adj['reason']}")
    else:
        print(f"  No rank adjustments needed.")

    print(f"{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ascension Engine v2.1 — Analytics Feedback Loop",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes, don't write")
    parser.add_argument("--report", action="store_true", help="Print report only")
    parser.add_argument("--seed-analytics", action="store_true", help="Insert test data into analytics.db")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="Days to look back")
    args = parser.parse_args()

    conn = connect_db()

    if args.seed_analytics:
        seed_analytics(conn)
        print(f"[OK] Test data seeded. Run without --seed-analytics to process.")
        conn.close()
        return

    videos = fetch_video_performance(conn, args.lookback)
    if not videos:
        print(f"[INFO] No videos with analytics in last {args.lookback} days.")
        conn.close()
        return

    manifest = load_manifest()
    clip_perf = map_clips_to_performance(manifest, videos)
    adjustments = compute_rank_adjustments(manifest, clip_perf)

    print_report(adjustments, videos)

    if args.report or args.dry_run:
        if adjustments and args.dry_run:
            print("[DRY RUN] No changes written.")
    else:
        applied = apply_adjustments(manifest, adjustments)
        save_manifest(manifest)
        log.info("Applied %d rank adjustments", applied)

    conn.close()


if __name__ == "__main__":
    main()
