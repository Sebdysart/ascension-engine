#!/usr/bin/env python3
"""
Ascension Engine — Narrative Tagger

Classifies existing library clips by narrative role using existing metadata:
mog_score, mog_track, tags. No new Claude vision calls needed.

Roles:
  victim_act1           — victim_contrast track + no direct stare
  awakening_transition  — mid_tier + motion tags
  ascension_reveal      — good_parts + stare/dark tags (or just good_parts)
  jawline_pop           — tight crop + high contrast lower face (any track)
  full_walk             — motion toward camera + full body (any track)
  general               — fallback

Writes narrative_role tag into library/tags/index.json.

Usage:
    python data/narrative_tagger.py [--dry-run]
    python data/narrative_tagger.py --report
"""

from __future__ import annotations
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_MANIFEST = _ROOT / "clip-manifest.json"
_TAGS_INDEX    = _ROOT / "library" / "tags" / "index.json"

log = logging.getLogger("narrative_tagger")

NARRATIVE_ROLES = {
    "victim_act1",
    "awakening_transition",
    "ascension_reveal",
    "jawline_pop",
    "full_walk",
    "general",
}

_DIRECT_STARE_TAGS = frozenset({
    "hunter_eyes", "direct_stare", "talking_head_direct", "eye_contact_direct",
    "mog_face_closeup_push",
})
_DARK_GRADE_TAGS = frozenset({
    "dark_cinema_mood", "dark_grade", "crushed_blacks", "night_car",
})
_MOTION_TAGS = frozenset({
    "high_energy_cut", "motion", "gym_broll_slowmo", "training_montage",
    "street_walk", "fast_cut", "high_energy",
})
_JAWLINE_TAGS = frozenset({
    "jawline_pop", "jaw_pop_side", "bone_structure_reveal", "tight_crop",
    "high_contrast",
})
_FULL_BODY_TAGS = frozenset({
    "full_body", "motion_toward_camera", "full_walk", "outfit_reveal",
    "lifestyle_broll",
})


def classify_narrative_role(clip: dict) -> str:
    """
    Classify a single clip's narrative role from its metadata.

    Priority order (first match wins):
    1. jawline_pop  — tight jawline/high-contrast tags (any track)
    2. full_walk    — full body + motion tags (any track)
    3. victim_act1  — victim_contrast track + no direct stare
    4. ascension_reveal — good_parts + stare or dark tags
    5. ascension_reveal — good_parts (no specific stare tag)
    6. awakening_transition — mid_tier + motion
    7. general      — fallback
    """
    tags = frozenset(clip.get("tags", []))
    track = clip.get("mog_track") or clip.get("track", "unclassified")

    if tags & _JAWLINE_TAGS:
        return "jawline_pop"

    if tags & _FULL_BODY_TAGS:
        return "full_walk"

    if track == "victim_contrast" and not (tags & _DIRECT_STARE_TAGS):
        return "victim_act1"

    if track == "good_parts":
        return "ascension_reveal"

    if track in ("mid_tier", "unclassified") and tags & _MOTION_TAGS:
        return "awakening_transition"

    return "general"


def tag_clips_by_narrative_role(
    clips: list[dict],
    dry_run: bool = False,
) -> dict[str, str]:
    """
    Classify all clips and return mapping clip_id → narrative_role.

    Parameters
    ----------
    clips:   List of clip dicts with clip_id, mog_track, mog_score, tags
    dry_run: If True, just return mapping without writing to index

    Returns
    -------
    dict mapping clip_id → narrative_role
    """
    role_map: dict[str, str] = {}
    for clip in clips:
        cid = clip.get("clip_id") or clip.get("id", "unknown")
        role = classify_narrative_role(clip)
        role_map[cid] = role

    if not dry_run:
        _write_narrative_roles_to_index(role_map)

    return role_map


def _write_narrative_roles_to_index(role_map: dict[str, str]) -> None:
    """Write narrative_role tags to library/tags/index.json."""
    index: dict = {"tags": {}, "last_updated": ""}
    if _TAGS_INDEX.exists():
        try:
            index = json.loads(_TAGS_INDEX.read_text())
        except Exception:
            pass

    for role in NARRATIVE_ROLES:
        tag_key = f"narrative_{role}"
        index["tags"][tag_key] = []

    for clip_id, role in role_map.items():
        tag_key = f"narrative_{role}"
        if tag_key not in index["tags"]:
            index["tags"][tag_key] = []
        if clip_id not in index["tags"][tag_key]:
            index["tags"][tag_key].append(clip_id)

    from datetime import datetime, timezone
    index["last_updated"] = datetime.now(timezone.utc).isoformat()
    _TAGS_INDEX.write_text(json.dumps(index, indent=2))
    log.info("Tags index updated with narrative roles — %d clips tagged", len(role_map))


def _load_manifest_clips() -> list[dict]:
    if not _CLIP_MANIFEST.exists():
        return []
    try:
        return json.loads(_CLIP_MANIFEST.read_text()).get("clips", [])
    except Exception:
        return []


def run_narrative_tagging(dry_run: bool = False) -> dict[str, str]:
    """Run narrative tagging against the full library. Returns role_map dict."""
    clips = _load_manifest_clips()
    if not clips:
        log.warning("No clips found in clip-manifest.json")
        return {}

    log.info("Tagging %d clips by narrative role …", len(clips))
    role_map = tag_clips_by_narrative_role(clips, dry_run=dry_run)

    counts = Counter(role_map.values())
    log.info("Narrative role distribution:")
    for role, count in sorted(counts.items()):
        log.info("  %-25s  %d clips", role, count)

    return role_map


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    parser = argparse.ArgumentParser(description="Tag library clips by narrative role")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true", help="Print role distribution only")
    args = parser.parse_args()

    role_map = run_narrative_tagging(dry_run=args.dry_run or args.report)

    if args.report or args.dry_run:
        counts = Counter(role_map.values())
        print("\nNarrative Role Distribution:")
        print("─" * 40)
        for role, count in sorted(counts.items()):
            print(f"  {role:<25}  {count}")
        print(f"\n  Total: {len(role_map)} clips")
    else:
        print(f"\nTagged {len(role_map)} clips. library/tags/index.json updated.")


if __name__ == "__main__":
    main()
