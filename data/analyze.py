#!/usr/bin/env python3
"""
Ascension Engine v2.0 — Weekly Bandit Optimizer
Usage:
  python3 data/analyze.py               # Run full analysis + update style-profile.json
  python3 data/analyze.py --dry-run     # Print recommendations, don't write files
  python3 data/analyze.py --report-only # Print report only, no updates
"""

import sqlite3
import json
import os
import sys
import math
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "analytics.db"
STYLE_PROFILE_PATH = BASE_DIR / "style-profile.json"
MUTATION_LOG_PATH = BASE_DIR / "data" / "mutation_log.json"

# ── Config ─────────────────────────────────────────────────────────────────
EPSILON = 0.3           # explore fraction for epsilon-greedy
TOP_PERCENTILE = 0.30   # top 30% = "winner"
MIN_SAMPLES = 3         # min videos per arm to trust stats
LOOKBACK_DAYS = 14      # rolling window for analysis
WATCH_PCT_DROP_ALERT = 0.15  # 15% week-over-week drop triggers reversion warning


def connect_db():
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at {DB_PATH}")
        print(f"[INFO]  Run: sqlite3 {DB_PATH} < {BASE_DIR}/data/schema.sql")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def fetch_recent_videos(conn, days=LOOKBACK_DAYS):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    cur = conn.execute("""
        SELECT video_id, hook_type, archetype, cut_rate, color_grade,
               music_bpm, avg_watch_pct, ctr_thumbnail, experiment_flag,
               views_7d, sentiment_score
        FROM videos
        WHERE posted_at >= ?
          AND avg_watch_pct IS NOT NULL
        ORDER BY posted_at DESC
    """, (cutoff,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def compute_arm_stats(videos):
    """Aggregate performance by (hook_type, color_grade) arm."""
    arms = {}
    for v in videos:
        key = (v["hook_type"], v["color_grade"])
        if key not in arms:
            arms[key] = {"pulls": 0, "total_watch": 0.0, "max_watch": 0.0, "videos": []}
        arms[key]["pulls"] += 1
        wp = v["avg_watch_pct"] or 0
        arms[key]["total_watch"] += wp
        arms[key]["max_watch"] = max(arms[key]["max_watch"], wp)
        arms[key]["videos"].append(v)

    for key, arm in arms.items():
        arm["mean_watch"] = arm["total_watch"] / arm["pulls"] if arm["pulls"] > 0 else 0
        arm["hook_type"] = key[0]
        arm["color_grade"] = key[1]
    return arms


def ucb1_score(mean, pulls, total_pulls):
    """Upper Confidence Bound 1 score for bandit arm."""
    if pulls == 0:
        return float("inf")
    return mean + math.sqrt(2 * math.log(max(total_pulls, 1)) / pulls)


def epsilon_greedy_allocation(arms, epsilon=EPSILON):
    """
    Epsilon-greedy: exploit best arm (1-epsilon) of the time,
    explore uniformly among all arms epsilon of the time.
    Returns dict of arm -> recommended weight.
    """
    qualified = {k: v for k, v in arms.items() if v["pulls"] >= MIN_SAMPLES}
    if not qualified:
        return {}

    best_arm = max(qualified.items(), key=lambda x: x[1]["mean_watch"])
    total_arms = len(qualified)

    weights = {}
    for key in qualified:
        if key == best_arm[0]:
            weights[key] = (1 - epsilon) + (epsilon / total_arms)
        else:
            weights[key] = epsilon / total_arms

    # Normalize
    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in weights.items()}


def find_winners(videos):
    """Top 30% by avg_watch_pct."""
    sorted_vids = sorted(videos, key=lambda v: v["avg_watch_pct"] or 0, reverse=True)
    cutoff = max(1, int(len(sorted_vids) * TOP_PERCENTILE))
    return sorted_vids[:cutoff]


def extract_winning_patterns(winners):
    """Extract most common attribute values from top performers."""
    def mode(lst):
        if not lst:
            return None
        return max(set(lst), key=lst.count)

    return {
        "hook_type": mode([v["hook_type"] for v in winners]),
        "color_grade": mode([v["color_grade"] for v in winners if v.get("color_grade")]),
        "archetype": mode([v["archetype"] for v in winners]),
        "music_bpm_avg": round(
            sum(v["music_bpm"] for v in winners if v.get("music_bpm")) /
            max(1, len([v for v in winners if v.get("music_bpm")])), 0
        ) if any(v.get("music_bpm") for v in winners) else None,
        "avg_watch_pct": round(
            sum(v["avg_watch_pct"] for v in winners) / len(winners), 4
        ),
        "sample_count": len(winners)
    }


def check_week_over_week(conn):
    """Compare this week's avg_watch_pct vs last week."""
    cur = conn.execute("""
        SELECT
          strftime('%Y-W%W', posted_at) as week,
          AVG(avg_watch_pct) as avg_wp
        FROM videos
        WHERE avg_watch_pct IS NOT NULL
          AND posted_at >= date('now', '-14 days')
        GROUP BY week
        ORDER BY week DESC
        LIMIT 2
    """)
    rows = cur.fetchall()
    if len(rows) < 2:
        return None, None, None
    this_week_wp = rows[0][1]
    last_week_wp = rows[1][1]
    delta = (this_week_wp - last_week_wp) / max(last_week_wp, 0.001)
    return this_week_wp, last_week_wp, delta


def update_style_profile(patterns, weights, dry_run=False):
    """Write winning patterns back to style-profile.json."""
    if not STYLE_PROFILE_PATH.exists():
        print(f"[WARN] style-profile.json not found at {STYLE_PROFILE_PATH}, skipping update")
        return None

    with open(STYLE_PROFILE_PATH) as f:
        profile = json.load(f)

    old_hook = profile.get("active_hook_type")
    old_grade = profile.get("color_grade", {}).get("active")

    profile["active_hook_type"] = patterns["hook_type"] or old_hook
    if patterns.get("color_grade"):
        profile.setdefault("color_grade", {})["active"] = patterns["color_grade"]

    # Update bandit weights
    if weights:
        best_arm = max(weights.items(), key=lambda x: x[1])
        profile["bandit"]["exploit_weight"] = round(min(best_arm[1], 0.85), 4)
        profile["bandit"]["explore_weight"] = round(1 - profile["bandit"]["exploit_weight"], 4)

    profile["last_updated"] = datetime.now().isoformat()

    delta = {
        "hook_type": {"old": old_hook, "new": profile["active_hook_type"]},
        "color_grade": {"old": old_grade, "new": profile["color_grade"]["active"]},
    }

    if not dry_run:
        with open(STYLE_PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"[OK] style-profile.json updated")

    return delta


def log_mutation(delta, patterns, dry_run=False):
    """Append mutation record to mutation_log.json."""
    entry = {
        "date": datetime.now().isoformat(),
        "type": "weekly_optimization",
        "delta": delta,
        "winning_patterns": patterns,
        "dry_run": dry_run
    }
    if not dry_run:
        log = []
        if MUTATION_LOG_PATH.exists():
            with open(MUTATION_LOG_PATH) as f:
                try:
                    log = json.load(f)
                except json.JSONDecodeError:
                    log = []
        log.append(entry)
        with open(MUTATION_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)


def print_report(videos, winners, patterns, weights, wow_data):
    """Print ASCII summary report."""
    this_wp, last_wp, delta = wow_data
    sep = "─" * 60

    print(f"\n{sep}")
    print(f"  ASCENSION ENGINE — WEEKLY ANALYSIS REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)
    print(f"  Videos analyzed ({LOOKBACK_DAYS}d):  {len(videos)}")
    print(f"  Winners (top {int(TOP_PERCENTILE*100)}%):          {len(winners)}")
    if this_wp:
        print(f"  This week avg watch%:  {this_wp:.1%}")
    if last_wp:
        direction = "▲" if (delta or 0) >= 0 else "▼"
        print(f"  vs last week:          {direction} {abs(delta or 0):.1%}")
        if (delta or 0) < -WATCH_PCT_DROP_ALERT:
            print(f"  ⚠️  REVERSION WARNING: Drop exceeds {WATCH_PCT_DROP_ALERT:.0%} threshold")
    print(sep)
    print(f"  WINNING PATTERNS:")
    print(f"    hook_type:    {patterns.get('hook_type', 'n/a')}")
    print(f"    color_grade:  {patterns.get('color_grade', 'n/a')}")
    print(f"    archetype:    {patterns.get('archetype', 'n/a')}")
    if patterns.get("music_bpm_avg"):
        print(f"    music_bpm:    ~{int(patterns['music_bpm_avg'])} BPM")
    print(f"    avg_watch%:   {patterns.get('avg_watch_pct', 0):.1%}")
    print(sep)
    if weights:
        print(f"  BANDIT ALLOCATIONS (top 5):")
        top_arms = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for arm, w in top_arms:
            print(f"    {arm[0]:20s} | {arm[1]:14s} | {w:.1%}")
    print(sep)
    print(f"  NEXT BATCH RECOMMENDATION:")
    print(f"    Exploit: {1-EPSILON:.0%} → {patterns.get('hook_type')} + {patterns.get('color_grade')}")
    print(f"    Explore: {EPSILON:.0%} → rotate untested combinations")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ascension Engine Weekly Optimizer")
    parser.add_argument("--dry-run", action="store_true", help="Print only, don't write files")
    parser.add_argument("--report-only", action="store_true", help="Report only, no style-profile update")
    args = parser.parse_args()

    dry_run = args.dry_run or args.report_only

    conn = connect_db()

    videos = fetch_recent_videos(conn)
    if not videos:
        print(f"[INFO] No videos with analytics in last {LOOKBACK_DAYS} days. Nothing to optimize.")
        return

    arms = compute_arm_stats(videos)
    weights = epsilon_greedy_allocation(arms)
    winners = find_winners(videos)
    patterns = extract_winning_patterns(winners)
    wow_data = check_week_over_week(conn)

    print_report(videos, winners, patterns, weights, wow_data)

    if not args.report_only:
        delta = update_style_profile(patterns, weights, dry_run=dry_run)
        if delta:
            log_mutation(delta, patterns, dry_run=dry_run)
            if dry_run:
                print("[DRY RUN] Would update style-profile.json with:")
                print(json.dumps(delta, indent=2))

    conn.close()


if __name__ == "__main__":
    main()
