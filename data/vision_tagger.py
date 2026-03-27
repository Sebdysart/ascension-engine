#!/usr/bin/env python3
"""
Ascension Engine — Vision Clip Tagger
Uses the authenticated `claude` CLI (Claude Code) to analyse keyframes and
write semantic tags into library/tags/index.json.

No API key or .env configuration required — the tagger delegates to the
`claude` binary which is already authenticated on the machine.

Usage (standalone):
    python data/vision_tagger.py <vid_id> [--keyframes-dir PATH] [--dry-run]

Called programmatically from ingest.py:
    from vision_tagger import tag_clips_for_video
    updated_clips = tag_clips_for_video(vid_id, clips, thumbnails_root, dry_run)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("vision_tagger")

# ── Config ────────────────────────────────────────────────────────────────────

# How many keyframes to pass per claude CLI call (keep ≤ 4 to control context)
BATCH_SIZE = 3
# Hard cap on frames sent per clip
MAX_FRAMES_PER_CLIP = 6
# Model passed to the claude CLI — haiku is fast and cheap for tagging
VISION_MODEL = os.environ.get("VISION_MODEL", "haiku")

TAGS_INDEX = _ROOT / "library" / "tags" / "index.json"

_PROMPT_TEMPLATE = """\
Use the Read tool to read each of these image files:
{paths}

Then output ONLY a JSON array of 3-8 semantic tags describing the video content.

Tag vocabulary examples (use these styles — invent similar ones as needed):
  "jaw_pop_side", "face_closeup_push", "gym_broll_slowmo", "phonk_drop_impact",
  "transformation_reveal", "outfit_reveal", "skincare_routine", "before_after_frame",
  "lifestyle_broll", "mirror_selfie", "street_walk", "supplement_shot",
  "physique_flex", "lighting_golden", "lighting_cold", "text_overlay_heavy",
  "talking_head_direct", "crowd_reaction", "training_montage", "posture_demo",
  "hair_styling", "grooming_ritual", "aesthetic_broll", "locker_room",
  "outdoor_natural", "dark_cinema_mood", "high_energy_cut".
  "mog_face_closeup_push", "jawline_pop", "hunter_eyes", "bone_structure_reveal",
  "victim_mogged_by", "pre_glowup_sideprofile", "average_guy_stare",
  "recessed_chin_reveal", "flat_lighting_cope",

Rules:
- Output ONLY a valid JSON array, nothing else. No markdown, no explanation.
- 3-8 tags maximum.
- Prefer specific over generic ("jaw_pop_side" > "face").
- Use lowercase_underscore format.

Example output: ["face_closeup_push","gym_broll_slowmo","dark_cinema_mood"]
"""

_MOG_SCORE_PROMPT = """\
Use the Read tool to read each of these image files:
{paths}

You are scoring video clip frames for CINEMATIC QUALITY on a 0.0–1.0 scale.
Evaluate the technical and aesthetic quality of the footage itself.

Scoring criteria:
- Lighting quality: dramatic, contrast-rich, cinematic lighting = high; flat, blown-out, harsh overhead = low (0–0.25)
- Motion and sharpness: crisp motion blur or sharp freeze = high; blurry, unfocused, shaky = low (0–0.20)
- Composition: tight face fill, rule of thirds, dynamic framing = high; awkward crop, excessive headroom, dead space = low (0–0.20)
- Visual punch: crushed blacks, teal/orange palette, strong depth/contrast = high; washed out, grey, low contrast = low (0–0.20)
- Energy and impact: high-motion, dramatic cut point, impact frame = high; static, no visual interest = low (0–0.15)

Output ONLY a valid JSON object, nothing else:
{{"mog_score": <float 0.0-1.0>, "dominant_trait": "<one word>", "notes": "<10 words max>"}}

Calibration notes — typical phone/selfie footage scores 0.30–0.75. Reserve 0.75+ for genuinely
cinematic clips with real lighting and composition. Use the full range; do not cluster near extremes.

Examples:
{{"mog_score": 0.78, "dominant_trait": "lighting", "notes": "dramatic rim light, crushed blacks, teal grade"}}
{{"mog_score": 0.35, "dominant_trait": "flat", "notes": "blown out overhead light, no contrast, dead composition"}}
{{"mog_score": 0.58, "dominant_trait": "composition", "notes": "decent framing, moderate contrast, average lighting"}}
"""


# ── Core ─────────────────────────────────────────────────────────────────────

def _check_claude_available() -> bool:
    """Return True if the `claude` CLI is on PATH."""
    return shutil.which("claude") is not None


def _gather_frames(keyframes_dir: Path, max_frames: int = MAX_FRAMES_PER_CLIP) -> list[Path]:
    """Return up to max_frames keyframe paths, preferring scene boundaries."""
    all_frames = sorted(keyframes_dir.glob("*.jpg"))
    if not all_frames:
        all_frames = sorted(keyframes_dir.glob("*.png"))
    if not all_frames:
        return []

    # Prefer scene_* frames; fill remainder with evenly-spaced every1s_*
    scene_frames = [f for f in all_frames if f.name.startswith("scene_")]
    other_frames = [f for f in all_frames if not f.name.startswith("scene_")]

    selected = scene_frames[:max_frames]
    remaining = max_frames - len(selected)
    if remaining > 0:
        step = max(1, len(other_frames) // remaining)
        selected += other_frames[::step][:remaining]

    return selected[:max_frames]


def _call_claude_cli(frame_paths: list[Path]) -> list[str]:
    """
    Invoke the `claude` CLI to tag a batch of keyframes.
    Uses --allowedTools Read so Claude can open the image files.
    Returns a list of tag strings.
    """
    paths_str = "\n".join(f"  {p.resolve()}" for p in frame_paths)
    prompt = _PROMPT_TEMPLATE.format(paths=paths_str)

    # Unset CLAUDECODE so nested invocation is allowed (when called from
    # within an active Claude Code session, e.g. during development).
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    cmd = [
        "claude", "-p", prompt,
        "--allowedTools", "Read",
        "--output-format", "json",
        "--no-session-persistence",
        "--model", VISION_MODEL,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except subprocess.TimeoutExpired:
        log.warning("claude CLI timed out for batch of %d frames", len(frame_paths))
        return []
    except FileNotFoundError:
        log.error("claude CLI not found — install Claude Code or add it to PATH")
        return []

    if result.returncode != 0:
        log.warning("claude CLI exited %d: %s", result.returncode,
                    result.stderr.strip()[-300:] if result.stderr else "(no stderr)")
        return []

    # The --output-format json wrapper: {"type":"result","result":"...","..."}
    try:
        wrapper = json.loads(result.stdout)
        raw = wrapper.get("result", "").strip()
    except (json.JSONDecodeError, AttributeError):
        # Fallback: try parsing stdout directly as a JSON array
        raw = result.stdout.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.startswith("```")
        ).strip()

    try:
        tags = json.loads(raw)
        if isinstance(tags, list):
            return [str(t).strip().lower().replace(" ", "_") for t in tags if t]
    except json.JSONDecodeError:
        # Best-effort: split on commas after stripping brackets
        raw_clean = raw.strip("[]").replace('"', "").replace("'", "")
        return [t.strip().lower().replace(" ", "_") for t in raw_clean.split(",") if t.strip()]

    return []


def _call_claude_mog_score(frame_paths: list[Path]) -> dict:
    """
    Score face clips 0.0–1.0 for mog potential via Claude vision.
    Returns dict with mog_score, dominant_trait, notes.
    """
    paths_str = "\n".join(f"  {p.resolve()}" for p in frame_paths)
    prompt = _MOG_SCORE_PROMPT.format(paths=paths_str)

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = [
        "claude", "-p", prompt,
        "--allowedTools", "Read",
        "--output-format", "json",
        "--no-session-persistence",
        "--model", VISION_MODEL,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "scoring failed"}

    if result.returncode != 0:
        return {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "scoring failed"}

    try:
        wrapper = json.loads(result.stdout)
        raw = wrapper.get("result", "").strip()
    except (json.JSONDecodeError, AttributeError):
        raw = result.stdout.strip()

    # Strip all markdown fences and collect every JSON object in the response.
    # The model may return one object per frame — we average the scores.
    import re as _re
    clean = _re.sub(r"```[a-z]*", "", raw).strip()

    candidates = []
    for m in _re.finditer(r'\{[^{}]+\}', clean):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "mog_score" in obj:
                candidates.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass

    if candidates:
        avg_score = sum(float(c["mog_score"]) for c in candidates) / len(candidates)
        best = max(candidates, key=lambda c: float(c["mog_score"]))
        return {
            "mog_score": round(max(0.0, min(1.0, avg_score)), 3),
            "dominant_trait": str(best.get("dominant_trait", "unknown")),
            "notes": str(best.get("notes", "")),
        }

    return {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "parse failed"}


# ── Mog Track Constants ────────────────────────────────────────────────────────
MOG_S_TIER = 0.65    # → good_parts/  (realistic ceiling for phone footage)
MOG_VICTIM = 0.45    # below → victim_contrast/
# 0.45–0.65 → mid_tier/


def classify_mog_track(mog_score: float) -> str:
    """Return 'good_parts', 'victim_contrast', or 'mid_tier'."""
    if mog_score >= MOG_S_TIER:
        return "good_parts"
    if mog_score < MOG_VICTIM:
        return "victim_contrast"
    return "mid_tier"


def tag_clip_keyframes(
    keyframes_dir: Path,
    dry_run: bool = False,
) -> list[str]:
    """
    Tag all keyframes in a single clip's keyframes directory.
    Returns deduplicated list of tags.
    """
    frames = _gather_frames(keyframes_dir)
    if not frames:
        log.warning("No keyframes found in %s", keyframes_dir)
        return []

    if dry_run:
        log.info("[dry-run] Would tag %d frames from %s", len(frames), keyframes_dir.name)
        return ["dry_run_tag_a", "dry_run_tag_b"]

    all_tags: list[str] = []
    n_batches = -(-len(frames) // BATCH_SIZE)  # ceiling division
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i: i + BATCH_SIZE]
        log.debug("  Vision batch %d/%d (%d frames) …",
                  i // BATCH_SIZE + 1, n_batches, len(batch))
        tags = _call_claude_cli(batch)
        all_tags.extend(tags)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in all_tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def tag_clips_for_video(
    vid_id: str,
    clips: list[dict],
    thumbnails_root: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Tag all clips for a video. Called from ingest.py.
    Updates each clip dict's 'tags' list in-place and also writes to
    library/tags/index.json. Returns the updated clips list.
    """
    if not _check_claude_available():
        log.warning("claude CLI not found — vision tagging skipped.")
        return clips

    tags_index = _load_tags_index()
    keyframes_dir = thumbnails_root / vid_id

    if not keyframes_dir.exists():
        log.warning("Keyframes dir not found: %s — vision tagging skipped.", keyframes_dir)
        return clips

    log.info("   Vision tagging: %d clips, keyframes from %s …",
             len(clips), keyframes_dir.name)

    # Tag all clips from the shared keyframes directory once
    tags = tag_clip_keyframes(keyframes_dir, dry_run)
    log.info("   Tags returned: %s", tags)

    # Mog scoring — use up to 3 scene frames
    mog_result = {"mog_score": 0.5, "dominant_trait": "unknown", "notes": ""}
    if not dry_run:
        scene_frames = _gather_frames(keyframes_dir, max_frames=3)
        if scene_frames:
            mog_result = _call_claude_mog_score(scene_frames)
            log.info("   Mog score: %.3f (%s) — track: %s",
                     mog_result["mog_score"],
                     mog_result["dominant_trait"],
                     classify_mog_track(mog_result["mog_score"]))

    # Apply to every clip from this video
    for clip in clips:
        clip["tags"] = tags
        clip["mog_score"] = mog_result["mog_score"]
        clip["mog_track"] = classify_mog_track(mog_result["mog_score"])
        clip["mog_notes"] = mog_result.get("notes", "")
        if not dry_run:
            clip_id = clip["clip_id"]
            for tag in tags:
                if tag not in tags_index["tags"]:
                    tags_index["tags"][tag] = []
                if clip_id not in tags_index["tags"][tag]:
                    tags_index["tags"][tag].append(clip_id)

    if not dry_run:
        _save_tags_index(tags_index)
        log.info("   Tags index updated — %d buckets", len(tags_index["tags"]))

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Vision clip tagger — Ascension Engine (uses `claude` CLI)",
    )
    parser.add_argument("vid_id", help="Video ID")
    parser.add_argument(
        "--keyframes-dir",
        help="Path to keyframes directory (default: library/assets/thumbnails/<vid_id>)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not _check_claude_available():
        log.error("`claude` CLI not found. Install Claude Code and ensure it is on PATH.")
        sys.exit(1)

    kf_dir = (
        Path(args.keyframes_dir)
        if args.keyframes_dir
        else _ROOT / "library" / "assets" / "thumbnails" / args.vid_id
    )

    tags = tag_clip_keyframes(kf_dir, dry_run=args.dry_run)
    print(f"\nTags for {args.vid_id}:")
    for t in tags:
        print(f"  • {t}")


if __name__ == "__main__":
    main()
