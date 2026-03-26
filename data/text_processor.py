#!/usr/bin/env python3
"""
Ascension Engine — Text Detection & Removal Processor
Scans clip keyframes for text overlays using easyocr, then:
  1. Writes FFmpeg delogo filter commands to library/text_templates/delogo_commands.json
  2. Writes clean OCR caption text to library/text_templates/ocr_templates.json

Called from ingest.py after keyframe extraction.

Usage (standalone):
    python data/text_processor.py <vid_id> [--keyframes-dir PATH] [--dry-run]
    python data/text_processor.py --help

Called programmatically:
    from data.text_processor import process_text_for_video
    process_text_for_video(vid_id, keyframes_dir, dry_run)
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = _ROOT / "library" / "text_templates"
DELOGO_FILE = TEMPLATES_DIR / "delogo_commands.json"
OCR_FILE = TEMPLATES_DIR / "ocr_templates.json"

log = logging.getLogger("text_processor")

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Config ────────────────────────────────────────────────────────────────────

# EasyOCR confidence threshold — detections below this are discarded
OCR_CONFIDENCE = 0.4

# Bounding-box area threshold (fraction of frame) below which we ignore hits
MIN_BOX_AREA_FRACTION = 0.001

# Text region types heuristics
# Watermark: small area, high in frame or bottom-right corner
# Caption: bottom third, wider region
# Overlay: large central text (usually good — reuse as caption template)


def _classify_text_region(bbox, text: str, frame_w: int, frame_h: int) -> str:
    """
    Classify an OCR hit as 'watermark', 'caption', or 'overlay'.
    bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (easyocr format).
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w = x2 - x1
    h = y2 - y1
    area_frac = (w * h) / max(frame_w * frame_h, 1)
    center_y_frac = (y1 + y2) / 2 / max(frame_h, 1)
    center_x_frac = (x1 + x2) / 2 / max(frame_w, 1)

    # Small + top/corner → watermark
    if area_frac < 0.02 and (center_y_frac < 0.15 or center_y_frac > 0.85):
        return "watermark"
    if area_frac < 0.015 and (center_x_frac < 0.15 or center_x_frac > 0.85):
        return "watermark"

    # Bottom third, wider → caption
    if center_y_frac > 0.65 and w > frame_w * 0.3:
        return "caption"

    # Large, upper/center → overlay (hook text)
    if area_frac > 0.05:
        return "overlay"

    return "caption"


def _bbox_to_delogo(bbox, frame_w: int, frame_h: int, margin: int = 8) -> str:
    """
    Convert easyocr bbox to FFmpeg delogo filter string.
    delogo=x=N:y=N:w=N:h=N
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1 = max(0, int(min(xs)) - margin)
    y1 = max(0, int(min(ys)) - margin)
    x2 = min(frame_w, int(max(xs)) + margin)
    y2 = min(frame_h, int(max(ys)) + margin)
    return f"delogo=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}"


def _get_frame_dimensions(frame_path: Path) -> tuple[int, int]:
    """Return (width, height) of a frame image."""
    if HAS_PIL:
        try:
            with Image.open(frame_path) as img:
                return img.size  # (width, height)
        except Exception:
            pass
    return (1080, 1920)  # TikTok default fallback


# ── Core ─────────────────────────────────────────────────────────────────────

_reader = None  # lazy singleton

def _get_reader():
    global _reader
    if _reader is None:
        log.info("Initializing easyocr reader (first call may be slow) …")
        _reader = easyocr.Reader(["en"], verbose=False)
    return _reader


def process_frame(frame_path: Path) -> dict:
    """
    Run OCR on a single frame. Returns:
    {
      "frame": str,
      "hits": [{"text": str, "type": str, "confidence": float, "delogo": str}]
    }
    """
    reader = _get_reader()
    frame_w, frame_h = _get_frame_dimensions(frame_path)

    try:
        results = reader.readtext(str(frame_path), detail=1)
    except Exception as e:
        log.warning("OCR failed for %s: %s", frame_path.name, e)
        return {"frame": frame_path.name, "hits": []}

    hits = []
    for (bbox, text, conf) in results:
        if conf < OCR_CONFIDENCE:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        area_frac = ((max(xs) - min(xs)) * (max(ys) - min(ys))) / max(frame_w * frame_h, 1)
        if area_frac < MIN_BOX_AREA_FRACTION:
            continue

        text_clean = text.strip()
        if not text_clean:
            continue

        region_type = _classify_text_region(bbox, text_clean, frame_w, frame_h)
        delogo = _bbox_to_delogo(bbox, frame_w, frame_h)

        hits.append({
            "text": text_clean,
            "type": region_type,
            "confidence": round(float(conf), 3),
            "delogo": delogo,
        })

    return {"frame": frame_path.name, "hits": hits}


def process_text_for_video(
    vid_id: str,
    keyframes_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Process all keyframes for a video. Writes delogo + OCR template files.
    Returns summary dict.

    Called from ingest.py after keyframe extraction.
    """
    if not HAS_EASYOCR:
        log.warning("easyocr not installed — text detection skipped. pip install easyocr")
        return {"vid_id": vid_id, "skipped": True}

    frames = sorted(keyframes_dir.glob("*.jpg")) + sorted(keyframes_dir.glob("*.png"))
    if not frames:
        log.warning("No keyframes in %s — text detection skipped.", keyframes_dir)
        return {"vid_id": vid_id, "frames_processed": 0}

    # Sample frames to control processing time (max 12 per video)
    if len(frames) > 12:
        step = len(frames) // 12
        frames = frames[::step][:12]

    log.info("   Text detection: %d frames for %s …", len(frames), vid_id)

    frame_results = []
    watermark_delogs: list[str] = []
    caption_texts: list[str] = []
    overlay_texts: list[str] = []

    for fp in frames:
        if dry_run:
            frame_results.append({"frame": fp.name, "hits": []})
            continue
        result = process_frame(fp)
        frame_results.append(result)
        for hit in result["hits"]:
            if hit["type"] == "watermark":
                watermark_delogs.append(hit["delogo"])
            elif hit["type"] == "caption":
                caption_texts.append(hit["text"])
            elif hit["type"] == "overlay":
                overlay_texts.append(hit["text"])

    summary = {
        "vid_id": vid_id,
        "frames_processed": len(frames),
        "watermarks_found": len(watermark_delogs),
        "captions_found": len(caption_texts),
        "overlays_found": len(overlay_texts),
    }

    if dry_run:
        log.info("[dry-run] Would write delogo + OCR template files")
        return summary

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    _append_delogo_commands(vid_id, watermark_delogs, frame_results)
    _append_ocr_templates(vid_id, caption_texts, overlay_texts)

    log.info(
        "   Text: %d watermarks, %d captions, %d overlays",
        len(watermark_delogs), len(caption_texts), len(overlay_texts),
    )
    return summary


# ── File I/O ─────────────────────────────────────────────────────────────────

def _append_delogo_commands(vid_id: str, delogs: list[str], frame_results: list[dict]) -> None:
    data = {}
    if DELOGO_FILE.exists():
        try:
            data = json.loads(DELOGO_FILE.read_text())
        except Exception:
            data = {}

    # Deduplicate delogo strings
    unique_delogs = list(dict.fromkeys(delogs))

    data[vid_id] = {
        "delogo_filters": unique_delogs,
        # Combined ffmpeg filter chain (multiple delogos joined with comma)
        "ffmpeg_filter_chain": ",".join(unique_delogs) if unique_delogs else None,
        "frame_details": frame_results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    DELOGO_FILE.write_text(json.dumps(data, indent=2))


def _append_ocr_templates(vid_id: str, captions: list[str], overlays: list[str]) -> None:
    data = {}
    if OCR_FILE.exists():
        try:
            data = json.loads(OCR_FILE.read_text())
        except Exception:
            data = {}

    # Deduplicate and cap at 20 each
    data[vid_id] = {
        "caption_templates": list(dict.fromkeys(captions))[:20],
        "overlay_templates": list(dict.fromkeys(overlays))[:10],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    OCR_FILE.write_text(json.dumps(data, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    parser = argparse.ArgumentParser(description="Text detection + removal — Ascension Engine")
    parser.add_argument("vid_id", help="Video ID")
    parser.add_argument("--keyframes-dir", help="Path to keyframes dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not HAS_EASYOCR:
        log.error("easyocr not installed. Run: pip install easyocr")
        sys.exit(1)

    kf_dir = (
        Path(args.keyframes_dir)
        if args.keyframes_dir
        else _ROOT / "library" / "assets" / "thumbnails" / args.vid_id
    )

    result = process_text_for_video(args.vid_id, kf_dir, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
