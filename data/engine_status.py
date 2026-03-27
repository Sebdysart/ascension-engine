#!/usr/bin/env python3
"""
Ascension Engine v4.2 — Engine Health Dashboard
Single source of truth for "is the engine healthy?"

Usage:
    python3 data/engine_status.py
    python3 data/engine_status.py --json
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.engine_db import EngineDB


def get_disk_usage(path: Path) -> str:
    total, used, free = shutil.disk_usage(str(path))
    return f"{used / (1024**3):.1f}GB used / {total / (1024**3):.1f}GB total ({free / (1024**3):.1f}GB free)"


def get_library_size() -> str:
    lib = ROOT / "library"
    if not lib.exists():
        return "0 MB"
    total = sum(f.stat().st_size for f in lib.rglob("*") if f.is_file())
    return f"{total / (1024**2):.1f} MB"


def count_files(directory: Path, pattern: str = "*") -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def main():
    parser = argparse.ArgumentParser(description="Ascension Engine Health Dashboard")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    db = EngineDB()
    db.init()
    health = db.get_health()

    # File counts
    file_stats = {
        "clips_on_disk": count_files(ROOT / "library" / "clips", "*.mp4"),
        "sequence_templates": count_files(ROOT / "library" / "sequence_templates", "*.json"),
        "grade_presets": count_files(ROOT / "library" / "grade_presets", "*.json"),
        "blueprints": count_files(ROOT / "library" / "blueprints", "*.json"),
        "style_profiles": count_files(ROOT / "style-profiles", "*.json"),
        "gold_videos": count_files(ROOT / "input" / "gold", "*.mp4"),
    }

    report = {**health, **file_stats, "library_size": get_library_size(), "disk": get_disk_usage(ROOT)}

    if args.json:
        print(json.dumps(report, indent=2))
        db.close()
        return

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  ASCENSION ENGINE v4.2 — HEALTH DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)

    print(f"\n  LIBRARY")
    print(f"    Total clips (DB)     : {health['total_clips']}")
    print(f"    Clips on disk        : {file_stats['clips_on_disk']}")
    print(f"    Good parts (S-tier)  : {health['good_parts']}")
    print(f"    Victim contrast      : {health['victim_contrast']}")
    print(f"    Unclassified         : {health['unclassified']}")
    print(f"    Sequence templates   : {file_stats['sequence_templates']}")
    print(f"    Grade presets        : {file_stats['grade_presets']}")
    print(f"    Blueprints           : {file_stats['blueprints']}")
    print(f"    Library size         : {file_stats.get('library_size', get_library_size())}")

    print(f"\n  INGESTS")
    print(f"    Completed            : {health['total_ingests']}")
    print(f"    Failed               : {health['failed_ingests']}")
    print(f"    Gold videos waiting  : {file_stats['gold_videos']}")

    print(f"\n  GENERATION")
    print(f"    Avg fidelity (7d)    : {health['avg_fidelity_7d']:.4f}")
    print(f"    Pending approvals    : {health['pending_approvals']}")

    print(f"\n  SYSTEM")
    print(f"    Disk                 : {report['disk']}")
    print(f"    Style profiles       : {file_stats['style_profiles']}")

    # Warnings
    warnings = []
    if health["total_clips"] == 0:
        warnings.append("No clips in library — run ingest first")
    if health["failed_ingests"] > 0:
        warnings.append(f"{health['failed_ingests']} failed ingests — check logs")
    if health["unclassified"] > health["total_clips"] * 0.5 and health["total_clips"] > 10:
        warnings.append(f"{health['unclassified']} unclassified clips — run Claude Vision tagging")
    if health["pending_approvals"] > 10:
        warnings.append(f"{health['pending_approvals']} edits awaiting approval")

    if warnings:
        print(f"\n  ⚠️  WARNINGS")
        for w in warnings:
            print(f"    • {w}")

    print(f"\n{sep}\n")

    db.log_health_snapshot()
    db.close()


if __name__ == "__main__":
    main()
