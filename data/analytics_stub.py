#!/usr/bin/env python3
"""
Ascension Engine — Analytics Stub
Generates synthetic analytics data and inserts it into analytics.db using the
schema defined in data/schema.sql.

This lets the feedback loop and bandit optimizer work without real platform
API connections.

Usage:
    python data/analytics_stub.py                  # seed with 30 synthetic videos
    python data/analytics_stub.py --count 60       # seed N videos
    python data/analytics_stub.py --reset          # drop and recreate DB, then seed
    python data/analytics_stub.py --status         # show current DB row counts
"""

import argparse
import json
import logging
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_SQL = _ROOT / "data" / "schema.sql"
ANALYTICS_DB = _ROOT / "data" / "analytics.db"

log = logging.getLogger("analytics_stub")

# ── Synthetic data distributions ─────────────────────────────────────────────

HOOK_TYPES = [
    ("before_after_zoom", 0.72),
    ("text_punch", 0.63),
    ("mogging_comparison", 0.58),
    ("transformation_reveal", 0.55),
    ("stats_overlay", 0.49),
    ("talking_head", 0.44),
    ("training_montage", 0.41),
]

ARCHETYPES = ["glow_up", "frame_maxxing", "skin_maxxing", "style_maxxing"]

COLOR_GRADES = [
    "teal_orange", "cold_blue", "warm_gold", "desaturated",
    "dark_cinema", "natural_indoor", "night_car",
]

PLATFORMS = ["tiktok", "instagram", "youtube_shorts"]

BPM_RANGES = {
    "glow_up": (148, 160),
    "frame_maxxing": (140, 155),
    "skin_maxxing": (120, 140),
    "style_maxxing": (140, 160),
}

MUSIC_TRACKS = [
    "phonk_drill_01", "hyperpop_chase_02", "trap_bounce_03",
    "ambient_rise_04", "drill_808_05", "phonk_rev_06",
    "trap_melodic_07", "hyperpop_alt_08",
]

FONT_STYLES = ["impact_white", "bebas_yellow", "drukwide_red", "monument_white"]

CUT_RATES = {
    "glow_up": 1.8,
    "frame_maxxing": 2.2,
    "skin_maxxing": 2.8,
    "style_maxxing": 2.0,
}


def _retention_curve(avg_watch_pct: float) -> list[dict]:
    """Generate a realistic retention dropoff curve."""
    points = []
    current = 1.0
    for sec in range(0, 61, 5):
        drop = random.uniform(0.02, 0.08) if sec > 0 else 0.0
        current = max(avg_watch_pct * 0.3, current - drop)
        points.append({"sec": sec, "pct": round(current, 3)})
    return points


def _synthetic_video(seed: int, days_ago_max: int = 60) -> dict:
    """Generate one synthetic video row."""
    rng = random.Random(seed)
    archetype = rng.choice(ARCHETYPES)
    hook_type, base_watch = rng.choice(HOOK_TYPES)

    # Add noise to watch pct
    avg_watch_pct = round(
        max(0.1, min(0.95, base_watch + rng.gauss(0, 0.08))), 4
    )

    bpm_lo, bpm_hi = BPM_RANGES[archetype]
    bpm = rng.randint(bpm_lo, bpm_hi)

    posted_days_ago = rng.randint(0, days_ago_max)
    posted_at = (datetime.now(timezone.utc) - timedelta(days=posted_days_ago)).isoformat()

    views_24h = int(rng.lognormvariate(8, 1.5))
    views_7d = int(views_24h * rng.uniform(3, 8))

    vid_ts = datetime.now(timezone.utc) - timedelta(days=posted_days_ago)
    vid_id = f"{archetype}_{hook_type}_{vid_ts.strftime('%Y%m%d')}_{seed:04d}"

    return {
        "video_id": vid_id,
        "title": f"{archetype.replace('_', ' ').title()} — {hook_type.replace('_', ' ').title()}",
        "hook_type": hook_type,
        "archetype": archetype,
        "cut_rate": CUT_RATES[archetype] + rng.uniform(-0.2, 0.2),
        "font_style": rng.choice(FONT_STYLES),
        "color_grade": rng.choice(COLOR_GRADES),
        "music_bpm": bpm,
        "music_track": rng.choice(MUSIC_TRACKS),
        "platform": rng.choice(PLATFORMS),
        "posted_at": posted_at,
        "views_24h": views_24h,
        "views_7d": views_7d,
        "avg_watch_pct": avg_watch_pct,
        "retention_dropoff": json.dumps(_retention_curve(avg_watch_pct)),
        "ctr_thumbnail": round(rng.uniform(0.03, 0.18), 4),
        "sentiment_score": round(rng.uniform(0.2, 0.95), 3),
        "comment_count": rng.randint(0, 500),
        "experiment_flag": rng.choice(["exploit", "exploit", "exploit", "explore"]),
        "style_profile_version": "2.0",
        "runway_used": 0,
        "render_time_sec": round(rng.uniform(30, 240), 1),
        "file_path": f"out/{vid_id}.mp4",
        "notes": "synthetic — analytics stub",
    }


def _synthetic_weekly_analysis(week_offset: int) -> dict:
    """Generate one weekly analysis row."""
    rng = random.Random(week_offset + 1000)
    week_start_dt = datetime.now(timezone.utc) - timedelta(weeks=week_offset + 1)
    week_end_dt = week_start_dt + timedelta(days=6)

    top_hook, base_watch = HOOK_TYPES[rng.randint(0, 2)]
    top_grade = rng.choice(COLOR_GRADES[:3])
    top_arch = rng.choice(ARCHETYPES)
    avg_watch = round(base_watch + rng.gauss(0, 0.05), 4)

    winning_patterns = [
        {"hook_type": top_hook, "color_grade": top_grade, "avg_watch_pct": avg_watch},
        {"hook_type": HOOK_TYPES[rng.randint(0, 4)][0], "avg_watch_pct": round(avg_watch - 0.05, 4)},
    ]
    recommendations = [
        f"Increase {top_hook} frequency to 4+ videos/week",
        f"Test {rng.choice(COLOR_GRADES[3:])} color grade",
        "Add stats overlay in first 2 seconds",
    ]

    return {
        "week_start": week_start_dt.date().isoformat(),
        "week_end": week_end_dt.date().isoformat(),
        "total_videos": rng.randint(5, 12),
        "avg_watch_pct": avg_watch,
        "top_hook_type": top_hook,
        "top_color_grade": top_grade,
        "top_archetype": top_arch,
        "top_bpm_range": f"{rng.randint(140, 155)}–{rng.randint(155, 165)}",
        "winning_patterns": json.dumps(winning_patterns),
        "recommendations": json.dumps(recommendations),
        "style_profile_delta": json.dumps({}),
    }


def _synthetic_mutation_log_entry(entry_id: int) -> dict:
    """Generate one mutation log row."""
    rng = random.Random(entry_id + 2000)
    mutation_types = ["promote", "demote", "retire", "new_experiment"]
    targets = ["before_after_zoom|teal_orange", "text_punch|cold_blue", "stats_overlay|warm_gold"]

    mutation_date = (
        datetime.now(timezone.utc) - timedelta(days=rng.randint(0, 30))
    ).date().isoformat()

    old_val = round(rng.uniform(0.3, 0.7), 3)
    new_val = round(old_val + rng.uniform(-0.15, 0.25), 3)

    return {
        "mutation_date": mutation_date,
        "type": rng.choice(mutation_types),
        "target": rng.choice(targets),
        "old_value": str(old_val),
        "new_value": str(new_val),
        "reason": rng.choice([
            "avg_watch_pct below 0.35 threshold",
            "top performer — promote for exploit",
            "track used >5 times in 14 days — retire",
            "new experiment batch",
        ]),
        "avg_watch_pct_before": old_val,
        "avg_watch_pct_after": new_val,
    }


# ── DB operations ─────────────────────────────────────────────────────────────

def ensure_db(db_path: Path = ANALYTICS_DB) -> sqlite3.Connection:
    """Create DB and apply schema if needed. Returns open connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if not SCHEMA_SQL.exists():
        log.error("schema.sql not found at %s", SCHEMA_SQL)
        sys.exit(1)

    schema = SCHEMA_SQL.read_text()
    try:
        conn.executescript(schema)
        conn.commit()
    except sqlite3.OperationalError as e:
        # Views/indexes already exist — fine
        if "already exists" not in str(e):
            raise

    return conn


def reset_db(db_path: Path = ANALYTICS_DB) -> sqlite3.Connection:
    """Delete and recreate the DB from schema."""
    if db_path.exists():
        db_path.unlink()
        log.info("Deleted existing DB: %s", db_path)
    return ensure_db(db_path)


def seed_videos(conn: sqlite3.Connection, count: int, start_seed: int = 1) -> int:
    """Insert synthetic video rows. Skips existing video_ids. Returns rows inserted."""
    inserted = 0
    for i in range(start_seed, start_seed + count):
        row = _synthetic_video(i)
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO videos (
                    video_id, title, hook_type, archetype, cut_rate, font_style,
                    color_grade, music_bpm, music_track, platform, posted_at,
                    views_24h, views_7d, avg_watch_pct, retention_dropoff,
                    ctr_thumbnail, sentiment_score, comment_count,
                    experiment_flag, style_profile_version, runway_used,
                    render_time_sec, file_path, notes
                ) VALUES (
                    :video_id, :title, :hook_type, :archetype, :cut_rate, :font_style,
                    :color_grade, :music_bpm, :music_track, :platform, :posted_at,
                    :views_24h, :views_7d, :avg_watch_pct, :retention_dropoff,
                    :ctr_thumbnail, :sentiment_score, :comment_count,
                    :experiment_flag, :style_profile_version, :runway_used,
                    :render_time_sec, :file_path, :notes
                )
                """,
                row,
            )
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except sqlite3.Error as e:
            log.warning("Insert failed for seed %d: %s", i, e)
    conn.commit()
    return inserted


def seed_weekly_analysis(conn: sqlite3.Connection, weeks: int = 8) -> int:
    """Insert synthetic weekly analysis rows."""
    inserted = 0
    for w in range(weeks):
        row = _synthetic_weekly_analysis(w)
        try:
            conn.execute(
                """
                INSERT INTO weekly_analysis (
                    week_start, week_end, total_videos, avg_watch_pct,
                    top_hook_type, top_color_grade, top_archetype, top_bpm_range,
                    winning_patterns, recommendations, style_profile_delta
                ) VALUES (
                    :week_start, :week_end, :total_videos, :avg_watch_pct,
                    :top_hook_type, :top_color_grade, :top_archetype, :top_bpm_range,
                    :winning_patterns, :recommendations, :style_profile_delta
                )
                """,
                row,
            )
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except sqlite3.Error as e:
            log.warning("Weekly analysis insert failed for week %d: %s", w, e)
    conn.commit()
    return inserted


def seed_mutation_log(conn: sqlite3.Connection, entries: int = 12) -> int:
    """Insert synthetic mutation log rows."""
    inserted = 0
    for i in range(entries):
        row = _synthetic_mutation_log_entry(i)
        try:
            conn.execute(
                """
                INSERT INTO mutation_log (
                    mutation_date, type, target, old_value, new_value,
                    reason, avg_watch_pct_before, avg_watch_pct_after
                ) VALUES (
                    :mutation_date, :type, :target, :old_value, :new_value,
                    :reason, :avg_watch_pct_before, :avg_watch_pct_after
                )
                """,
                row,
            )
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except sqlite3.Error as e:
            log.warning("Mutation log insert failed for entry %d: %s", i, e)
    conn.commit()
    return inserted


def print_status(conn: sqlite3.Connection) -> None:
    """Print row counts and top performers."""
    tables = ["videos", "weekly_analysis", "mutation_log", "content_queue", "trend_research"]
    print("\n── Analytics DB Status ──────────────────────────────")
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<20} {count:>6} rows")
        except sqlite3.Error:
            print(f"  {table:<20}  (table missing)")

    try:
        print("\n── Top 5 hook types (avg_watch_pct) ─────────────────")
        rows = conn.execute(
            "SELECT hook_type, ROUND(AVG(avg_watch_pct),4) as awp, COUNT(*) as n "
            "FROM videos WHERE avg_watch_pct IS NOT NULL "
            "GROUP BY hook_type ORDER BY awp DESC LIMIT 5"
        ).fetchall()
        for r in rows:
            print(f"  {r['hook_type']:<28} awp={r['awp']}  n={r['n']}")
    except sqlite3.Error as e:
        log.warning("Status query failed: %s", e)
    print("─────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    parser = argparse.ArgumentParser(description="Analytics stub — Ascension Engine")
    parser.add_argument("--count", type=int, default=30, help="Synthetic video rows to seed")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate DB before seeding")
    parser.add_argument("--status", action="store_true", help="Show DB row counts and exit")
    parser.add_argument("--db", default=str(ANALYTICS_DB), help="Path to analytics.db")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.reset:
        conn = reset_db(db_path)
        log.info("DB reset.")
    else:
        conn = ensure_db(db_path)

    if args.status:
        print_status(conn)
        conn.close()
        return

    log.info("Seeding %d synthetic video rows …", args.count)
    # Offset seed to avoid collisions with existing rows
    existing = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    v_inserted = seed_videos(conn, args.count, start_seed=existing + 1)
    w_inserted = seed_weekly_analysis(conn)
    m_inserted = seed_mutation_log(conn)

    log.info(
        "Seeded: %d videos, %d weekly_analysis, %d mutation_log",
        v_inserted, w_inserted, m_inserted,
    )
    print_status(conn)
    conn.close()


if __name__ == "__main__":
    main()
