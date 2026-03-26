#!/usr/bin/env python3
"""
Ascension Engine — Vision Clip Tagger
Uses Claude Vision (claude-haiku-4-5) to read keyframes and write semantic
tags into library/tags/index.json.

Tags are also written back onto each clip's 'tags' list in-place so the
ingest pipeline can persist them to clip-manifest.json.

Usage (standalone):
    python data/vision_tagger.py <vid_id> --keyframes-dir library/assets/thumbnails/<vid_id>
    python data/vision_tagger.py --help

Called programmatically from ingest.py:
    from data.vision_tagger import tag_clips_for_video
    updated_clips = tag_clips_for_video(vid_id, clips, keyframes_dir, dry_run)
"""

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path

# Load .env from repo root before importing anthropic
_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _ROOT / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE)
    except ImportError:
        # Manual fallback: parse KEY=VALUE lines
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

log = logging.getLogger("vision_tagger")

# ── Config ────────────────────────────────────────────────────────────────────

VISION_MODEL = os.environ.get("VISION_MODEL", "claude-haiku-4-5-20251001")
BATCH_SIZE = 3          # keyframes per API call
MAX_FRAMES_PER_CLIP = 6 # cap frames sent per clip to control cost
TAGS_INDEX = _ROOT / "library" / "tags" / "index.json"

# Prompt injected with each batch
_SYSTEM_PROMPT = """\
You are a video content analyst for a short-form fitness/lifestyle video engine.
Given keyframes from a video clip, output ONLY a comma-separated list of semantic tags.

Tag vocabulary examples (use these styles — invent similar ones as needed):
  jaw_pop_side, face_closeup_push, gym_broll_slowmo, phonk_drop_impact,
  transformation_reveal, outfit_reveal, skincare_routine, before_after_frame,
  lifestyle_broll, mirror_selfie, street_walk, supplement_shot, physique_flex,
  lighting_golden, lighting_cold, text_overlay_heavy, talking_head_direct,
  crowd_reaction, training_montage, posture_demo, hair_styling, grooming_ritual,
  aesthetic_broll, locker_room, outdoor_natural, dark_cinema_mood, high_energy_cut.

Rules:
- Output ONLY tags, comma-separated, no explanations.
- 3–8 tags per batch of keyframes.
- Prefer specific over generic (jaw_pop_side > face).
- Do NOT include forbidden content tags (slurs, self-harm, etc.).
"""

_USER_TEMPLATE = "Here are {n} keyframes from a clip. Tag them:"


# ── Core ─────────────────────────────────────────────────────────────────────

def _encode_image(path: Path) -> str:
    """Return base64-encoded JPEG data."""
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _call_vision_api(client: "anthropic.Anthropic", frame_paths: list[Path]) -> list[str]:
    """Send up to BATCH_SIZE frames to Claude Vision, return list of tags."""
    content = [{"type": "text", "text": _USER_TEMPLATE.format(n=len(frame_paths))}]
    for fp in frame_paths:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _encode_image(fp),
            },
        })

    response = client.messages.create(
        model=VISION_MODEL,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    tags = [t.strip().lower().replace(" ", "_") for t in raw.split(",") if t.strip()]
    return tags


def _gather_frames(keyframes_dir: Path, max_frames: int = MAX_FRAMES_PER_CLIP) -> list[Path]:
    """Return up to max_frames keyframe paths, preferring scene boundaries."""
    all_frames = sorted(keyframes_dir.glob("*.jpg"))
    if not all_frames:
        all_frames = sorted(keyframes_dir.glob("*.png"))
    if not all_frames:
        return []

    # Prefer scene_* frames; fill remainder with every1s_*
    scene_frames = [f for f in all_frames if f.name.startswith("scene_")]
    other_frames = [f for f in all_frames if not f.name.startswith("scene_")]

    selected = scene_frames[:max_frames]
    remaining = max_frames - len(selected)
    if remaining > 0:
        step = max(1, len(other_frames) // remaining)
        selected += other_frames[::step][:remaining]

    return selected[:max_frames]


def tag_clip_keyframes(
    client: "anthropic.Anthropic",
    keyframes_dir: Path,
    dry_run: bool = False,
) -> list[str]:
    """Tag all keyframes in a single clip's keyframes dir. Returns merged tag list."""
    frames = _gather_frames(keyframes_dir)
    if not frames:
        log.warning("No keyframes found in %s", keyframes_dir)
        return []

    if dry_run:
        log.info("[dry-run] Would tag %d frames from %s", len(frames), keyframes_dir.name)
        return ["dry_run_tag_a", "dry_run_tag_b"]

    all_tags: list[str] = []
    # Send in batches of BATCH_SIZE
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i : i + BATCH_SIZE]
        try:
            tags = _call_vision_api(client, batch)
            all_tags.extend(tags)
            log.debug("  Batch %d/%d → %s", i // BATCH_SIZE + 1,
                      -(-len(frames) // BATCH_SIZE), tags)
        except Exception as e:
            log.warning("Vision API error for batch %s: %s", i, e)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tags: list[str] = []
    for t in all_tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)
    return unique_tags


def tag_clips_for_video(
    vid_id: str,
    clips: list[dict],
    thumbnails_root: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Tag all clips for a video. Called from ingest.py.

    Updates each clip dict's 'tags' list in-place and returns the updated list.
    Also writes tags into library/tags/index.json.
    """
    if not HAS_ANTHROPIC:
        log.warning("anthropic not installed — vision tagging skipped. pip install anthropic")
        return clips

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — vision tagging skipped.")
        return clips

    client = anthropic.Anthropic(api_key=api_key)
    tags_index = _load_tags_index()

    for clip in clips:
        clip_id = clip["clip_id"]
        # Keyframes live under thumbnails_root/<vid_id>/ (shared dir)
        keyframes_dir = thumbnails_root / vid_id
        if not keyframes_dir.exists():
            log.warning("Keyframes dir not found: %s", keyframes_dir)
            continue

        log.info("   Tagging clip %s …", clip_id)
        tags = tag_clip_keyframes(client, keyframes_dir, dry_run)
        clip["tags"] = tags
        log.info("   Tags: %s", tags)

        # Update tags index
        if not dry_run:
            for tag in tags:
                if tag not in tags_index["tags"]:
                    tags_index["tags"][tag] = []
                if clip_id not in tags_index["tags"][tag]:
                    tags_index["tags"][tag].append(clip_id)

    if not dry_run:
        _save_tags_index(tags_index)
        log.info("Tags index updated: %d tag buckets", len(tags_index["tags"]))

    return clips


# ── Tags Index I/O ────────────────────────────────────────────────────────────

def _load_tags_index() -> dict:
    if TAGS_INDEX.exists():
        try:
            return json.loads(TAGS_INDEX.read_text())
        except Exception:
            pass
    return {"tags": {}, "last_updated": ""}


def _save_tags_index(index: dict) -> None:
    from datetime import datetime, timezone
    index["last_updated"] = datetime.now(timezone.utc).isoformat()
    TAGS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    TAGS_INDEX.write_text(json.dumps(index, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="Vision clip tagger — Ascension Engine")
    parser.add_argument("vid_id", help="Video ID (e.g. myvideo_ab12cd34)")
    parser.add_argument(
        "--keyframes-dir",
        help="Path to keyframes directory (default: library/assets/thumbnails/<vid_id>)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keyframes_dir = (
        Path(args.keyframes_dir)
        if args.keyframes_dir
        else _ROOT / "library" / "assets" / "thumbnails" / args.vid_id
    )

    if not HAS_ANTHROPIC:
        log.error("anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    tags = tag_clip_keyframes(client, keyframes_dir, dry_run=args.dry_run)
    print(f"\nTags for {args.vid_id}:")
    for t in tags:
        print(f"  • {t}")


if __name__ == "__main__":
    main()
