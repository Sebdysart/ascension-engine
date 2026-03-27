#!/usr/bin/env python3
"""
Ascension Engine — Batch Edit Generator

Generates N edits using the beat-grid sequencer and FFmpeg concat.
Remotion is not used.  LUT post-processing is applied via data/lut_processor.py.

Clip source priority
────────────────────
1. Real clips listed in clip-manifest.json (library/clips/*.mp4).
2. Synthetic lavfi color clips generated on the fly (used when real clips are
   absent, e.g. before the library is populated).

Usage
─────
  python3 data/generate_batch.py              # 3 edits, default BPM ladder
  python3 data/generate_batch.py --count 5    # 5 edits
  python3 data/generate_batch.py --bpm 120    # single BPM for all edits
  python3 data/generate_batch.py --dry-run    # print plan, no render
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure repo root is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.sequencer import build_sequence, describe  # noqa: E402
from data.lut_processor import process_single, ensure_sample_luts  # noqa: E402

OUT_DIR = _ROOT / "out"
TMP_ROOT = _ROOT / "tmp_batch"
MANIFEST = _ROOT / "clip-manifest.json"
LUTS_DIR = _ROOT / "luts"

log = logging.getLogger("generate_batch")

# BPM ladder used when generating multiple edits
BPM_LADDER = [108.0, 114.0, 120.0]

# One LUT per edit (cycles through available LUTs)
LUT_ROTATION = ["teal_orange.cube", "warm_gold.cube", "cold_blue.cube", "neutral.cube"]

# Synthetic clip palette — one colour per section type (H,S,V style via ffmpeg)
SECTION_COLOURS: dict[str, str] = {
    "verse":   "0x1a1a2e",   # deep navy
    "buildup": "0x16213e",   # dark blue
    "drop":    "0x0f3460",   # cobalt
}
FALLBACK_COLOUR = "0x111111"

# 1080×1920 portrait (TikTok / Reels)
VIDEO_W, VIDEO_H = 1080, 1920
VIDEO_FPS = 30


# ── Clip resolution ───────────────────────────────────────────────────────────

def _load_manifest() -> list[dict]:
    """Return clip dicts from clip-manifest.json, or [] if file is missing."""
    if not MANIFEST.exists():
        return []
    with open(MANIFEST) as f:
        return json.load(f).get("clips", [])


def _real_clips(manifest: list[dict]) -> list[Path]:
    """Return Path objects for clips that actually exist on disk."""
    found = []
    for c in manifest:
        p = _ROOT / c["file"]
        if p.exists():
            found.append(p)
    return found


def _make_synthetic_clip(
    out_path: Path,
    duration: float,
    colour: str,
    label: str,
    dry_run: bool = False,
) -> bool:
    """Render a solid-colour lavfi clip with a text label burned in."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use drawtext only if a font is available; fall back to plain color
    vf_parts = [
        f"scale={VIDEO_W}:{VIDEO_H}",
        (
            f"drawtext=text='{label}':fontsize=48:fontcolor=white"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
        ),
    ]
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={colour}:s={VIDEO_W}x{VIDEO_H}:r={VIDEO_FPS}:d={duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
        "-an",
        str(out_path),
    ]

    if dry_run:
        log.info("[dry-run] %s", " ".join(cmd))
        return True

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Retry without drawtext (no font available)
        cmd_simple = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={colour}:s={VIDEO_W}x{VIDEO_H}:r={VIDEO_FPS}:d={duration:.3f}",
            "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
            "-an",
            str(out_path),
        ]
        result = subprocess.run(cmd_simple, capture_output=True)

    return result.returncode == 0


def _trim_real_clip(
    src: Path,
    out_path: Path,
    duration: float,
    src_offset: float = 0.0,
    dry_run: bool = False,
) -> bool:
    """Trim `duration` seconds from `src` starting at `src_offset`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(src_offset),
        "-t", str(duration),
        "-i", str(src),
        "-vf", f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
               f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an",
        str(out_path),
    ]
    if dry_run:
        log.info("[dry-run] %s", " ".join(cmd))
        return True
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ── Edit assembly ─────────────────────────────────────────────────────────────

def generate_edit(
    edit_index: int,
    bpm: float,
    lut_name: str,
    real_clips: list[Path],
    tmp_dir: Path,
    dry_run: bool = False,
) -> dict | None:
    """
    Build one edit.  Returns a summary dict on success, None on failure.
    """
    log.info("── Edit %d  BPM=%.0f  LUT=%s ──", edit_index + 1, bpm, lut_name)

    slots = build_sequence(bpm=bpm)
    log.info(describe(slots, bpm=bpm))

    use_real = len(real_clips) >= 2
    segments: list[Path] = []

    for i, slot in enumerate(slots):
        seg_path = tmp_dir / f"edit{edit_index:02d}_seg{i:02d}.mp4"

        if use_real:
            src = real_clips[i % len(real_clips)]
            ok = _trim_real_clip(src, seg_path, slot.duration, src_offset=slot.start % 3.0, dry_run=dry_run)
        else:
            colour = SECTION_COLOURS.get(slot.section, FALLBACK_COLOUR)
            label = f"s{i+1} {slot.section[:3]} {slot.duration:.2f}s"
            ok = _make_synthetic_clip(seg_path, slot.duration, colour, label, dry_run=dry_run)

        if ok:
            segments.append(seg_path)
        else:
            log.warning("  Segment %d failed — skipping", i + 1)

    if not segments:
        log.error("  No segments rendered for edit %d", edit_index + 1)
        return None

    # Build concat list
    concat_list = tmp_dir / f"edit{edit_index:02d}_concat.txt"
    concat_list.write_text("\n".join(f"file '{s}'" for s in segments) + "\n")

    raw_out = OUT_DIR / f"edit_{edit_index+1:02d}_raw.mp4"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(raw_out),
    ]

    if dry_run:
        log.info("[dry-run] concat → %s", raw_out)
        graded_path = raw_out.parent / f"{raw_out.stem}_graded.mp4"
        return _build_summary(edit_index, bpm, lut_name, slots, raw_out, graded_path)

    result = subprocess.run(concat_cmd, capture_output=True)
    if result.returncode != 0:
        log.error("  Concat failed: %s", result.stderr[-300:])
        return None

    # LUT post-process
    lut_path = LUTS_DIR / lut_name
    graded_path = process_single(raw_out, lut_arg=str(lut_path))
    if graded_path is None:
        log.warning("  LUT grading failed — raw file kept")
        graded_path = raw_out

    return _build_summary(edit_index, bpm, lut_name, slots, raw_out, graded_path)


def _build_summary(
    edit_index: int,
    bpm: float,
    lut_name: str,
    slots,
    raw_path: Path,
    graded_path: Path,
) -> dict:
    total = slots[-1].end if slots else 0.0
    n_cuts = len(slots)
    cut_rate = n_cuts / total if total > 0 else 0

    raw_size = raw_path.stat().st_size if raw_path.exists() else 0
    graded_size = graded_path.stat().st_size if graded_path.exists() else 0

    return {
        "edit": edit_index + 1,
        "bpm": bpm,
        "lut": lut_name,
        "n_cuts": n_cuts,
        "duration_sec": round(total, 2),
        "cut_rate": round(cut_rate, 2),
        "raw_path": str(raw_path),
        "graded_path": str(graded_path),
        "raw_size_kb": raw_size // 1024,
        "graded_size_kb": graded_size // 1024,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Ascension Engine — batch edit generator")
    parser.add_argument("--count", type=int, default=3, help="Number of edits to generate")
    parser.add_argument("--bpm", type=float, default=None, help="Fixed BPM (default: ladder 108/114/120)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_sample_luts()

    manifest = _load_manifest()
    real_clips = _real_clips(manifest)
    log.info("Clips in manifest: %d  |  Clips on disk: %d", len(manifest), len(real_clips))
    if not real_clips:
        log.info("No real clips found — using synthetic lavfi colour clips")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ascension_batch_", dir=TMP_ROOT if TMP_ROOT.exists() else None))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(args.count):
        bpm = args.bpm if args.bpm else BPM_LADDER[i % len(BPM_LADDER)]
        lut = LUT_ROTATION[i % len(LUT_ROTATION)]
        summary = generate_edit(i, bpm, lut, real_clips, tmp_dir, dry_run=args.dry_run)
        if summary:
            results.append(summary)

    # ── Report ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  BATCH COMPLETE — {len(results)}/{args.count} edits rendered")
    print("=" * 60)
    for r in results:
        raw_kb = r["raw_size_kb"]
        graded_kb = r["graded_size_kb"]
        print(
            f"\n  Edit {r['edit']}  BPM={r['bpm']:.0f}  LUT={r['lut']}\n"
            f"    Cuts      : {r['n_cuts']}  ({r['cut_rate']:.2f} cuts/s over {r['duration_sec']}s)\n"
            f"    Raw       : {Path(r['raw_path']).name}  ({raw_kb} KB)\n"
            f"    Graded    : {Path(r['graded_path']).name}  ({graded_kb} KB)\n"
            f"    Location  : {r['graded_path']}"
        )
    print()


if __name__ == "__main__":
    main()
