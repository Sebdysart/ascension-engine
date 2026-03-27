#!/usr/bin/env python3
"""
Ascension Engine — impact_scorer.py
Per-frame visual impact analysis for beat-synced edit assembly.

Metrics:
  - motion_magnitude   : Farneback dense optical flow mean magnitude
  - face_fill          : fraction of frame area occupied by detected face bbox
  - contrast           : RMS of Sobel edge magnitude (normalised 0-1)
  - shadow_ratio       : fraction of pixels with luminance < 64 (lifted blacks = lower shadow)
  - face_angle         : 3-point landmark yaw (None if MediaPipe unavailable)

Composite impact score:
  impact = 0.35 * motion + 0.30 * face_fill + 0.20 * contrast + 0.15 * shadow_ratio

Farneback params (spec-exact):
  pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2

Processing target: ≤10 ms/frame on M2 at 24fps equivalent.
60fps sources are downsampled to every Nth frame so effective rate = 24fps.

Usage:
    python3 data/impact_scorer.py library/clips/clip.mp4
    python3 data/impact_scorer.py library/clips/ --batch
    python3 data/impact_scorer.py clip.mp4 --json out/scores.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("impact_scorer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

# ── Optional: MediaPipe face mesh for yaw angle ───────────────────────────────
try:
    import mediapipe as mp  # type: ignore
    _MP_FACE = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.4,
    )
    HAS_MEDIAPIPE = True
except Exception:
    _MP_FACE = None
    HAS_MEDIAPIPE = False

# ── OpenCV face detector (fallback, always available) ────────────────────────
_CASCADE_PATH = str(cv2.data.haarcascades) + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
    return _FACE_CASCADE


# ── Constants ────────────────────────────────────────────────────────────────
TARGET_FPS = 24.0
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)
# Impact weights (must sum to 1.0)
W_MOTION = 0.35
W_FACE   = 0.30
W_CONTRAST = 0.20
W_SHADOW = 0.15

# Processing resolutions (split: face needs size, flow needs speed)
# Face detection — larger for accuracy
FACE_WIDTH  = 360
FACE_HEIGHT = 640
# Optical flow — 240×427 benchmarks at ~8.8ms on M2 (target ≤10ms total)
FLOW_WIDTH  = 240
FLOW_HEIGHT = 427


# ── Per-frame metrics ────────────────────────────────────────────────────────

def _compute_contrast(gray: np.ndarray) -> float:
    """Normalised RMS Sobel edge magnitude (0–1)."""
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Normalise by theoretical max (255 * sqrt(2) * ~kernel_gain)
    return float(np.sqrt(np.mean(mag ** 2)) / 362.0)


def _compute_shadow_ratio(gray: np.ndarray) -> float:
    """Fraction of pixels with luma < 64 (deep shadow coverage)."""
    return float(np.mean(gray < 64))


def _detect_face_bbox(gray: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return (x, y, w, h) of largest detected face, or None."""
    faces = _face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4,
        minSize=(24, 24), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    # Pick largest by area
    return tuple(max(faces, key=lambda f: f[2] * f[3]))  # type: ignore[return-value]


def _face_fill(bbox: Optional[tuple], frame_h: int, frame_w: int) -> float:
    """Fraction 0–1 of frame area covered by face bbox."""
    if bbox is None:
        return 0.0
    _, _, fw, fh = bbox
    return float(fw * fh) / float(frame_h * frame_w)


def _face_yaw_mediapipe(bgr: np.ndarray) -> Optional[float]:
    """3-point yaw estimate via MediaPipe (nose tip, left/right eye corner)."""
    if not HAS_MEDIAPIPE or _MP_FACE is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _MP_FACE.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    h, w = bgr.shape[:2]
    # Indices: nose=1, left-eye-corner=33, right-eye-corner=263
    nose  = np.array([lm[1].x * w, lm[1].y * h])
    leye  = np.array([lm[33].x * w, lm[33].y * h])
    reye  = np.array([lm[263].x * w, lm[263].y * h])
    mid   = (leye + reye) / 2.0
    dx    = nose[0] - mid[0]
    eye_w = float(np.linalg.norm(reye - leye))
    if eye_w < 1.0:
        return None
    # Yaw in degrees: positive = looking right
    return float(np.degrees(np.arctan2(dx, eye_w * 0.5)))


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_clip_frames(video_path: str) -> dict:
    """
    Score all (downsampled-to-24fps) frames of a video clip.

    Returns
    -------
    dict with keys:
        video_path       : str
        fps_source       : float
        fps_processed    : float
        frame_count      : int
        processing_ms    : float
        ms_per_frame     : float
        impact_scores    : list[float]   — composite 0-1 per frame
        face_fills       : list[float]
        motion_magnitudes: list[float]   — normalised 0-1
        contrasts        : list[float]
        shadow_ratios    : list[float]
        face_angles      : list[float|None]
        summary          : dict          — mean/max/p90
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downsample stride: skip frames so effective rate ≈ TARGET_FPS
    stride = max(1, round(src_fps / TARGET_FPS))
    log.info("  %s  src_fps=%.1f  stride=%d  frames=%d", path.name, src_fps, stride, total_frames)

    t0 = time.perf_counter()

    # --- collections ---
    impact_scores: list[float]    = []
    face_fills:    list[float]    = []
    motions:       list[float]    = []
    contrasts:     list[float]    = []
    shadows:       list[float]    = []
    angles:        list           = []

    prev_gray_flow: Optional[np.ndarray] = None    # low-res for flow
    face_bbox_cache: Optional[tuple]     = None   # from first frame — consistent framing
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # Two working resolutions: flow (small/fast) and face (larger/accurate)
        small_flow = cv2.resize(frame, (FLOW_WIDTH, FLOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
        gray_flow  = cv2.cvtColor(small_flow, cv2.COLOR_BGR2GRAY)

        # ── Face bbox (detected once on first frame; consistent framing) ────
        if face_bbox_cache is None and frame_idx == 0:
            small_face = cv2.resize(frame, (FACE_WIDTH, FACE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            gray_face  = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
            face_bbox_cache = _detect_face_bbox(gray_face)
            # Scale bbox from face-res to flow-res for consistent fill calc
            if face_bbox_cache is not None:
                sx = FLOW_WIDTH  / FACE_WIDTH
                sy = FLOW_HEIGHT / FACE_HEIGHT
                x, y, w, h = face_bbox_cache
                face_bbox_cache = (int(x*sx), int(y*sy), int(w*sx), int(h*sy))

        ff = _face_fill(face_bbox_cache, FLOW_HEIGHT, FLOW_WIDTH)

        # ── Optical flow ────────────────────────────────────────────────────
        if prev_gray_flow is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray_flow, gray_flow, None, **FARNEBACK_PARAMS
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            # Normalise: max expected motion ~20px at FLOW resolution
            motion_norm = float(np.mean(mag)) / 20.0
            motion_norm = min(motion_norm, 1.0)
        else:
            motion_norm = 0.0

        # ── Contrast ────────────────────────────────────────────────────────
        contrast = _compute_contrast(gray_flow)

        # ── Shadow ratio ────────────────────────────────────────────────────
        shadow = _compute_shadow_ratio(gray_flow)

        # ── Face angle (MediaPipe optional) ─────────────────────────────────
        # Only run on first frame for speed; propagate through clip
        if len(angles) == 0:
            angle = _face_yaw_mediapipe(small_flow)
        else:
            angle = angles[-1]  # carry forward (consistent framing assumption)

        # ── Composite impact ─────────────────────────────────────────────────
        impact = (
            W_MOTION   * motion_norm +
            W_FACE     * ff +
            W_CONTRAST * contrast +
            W_SHADOW   * shadow
        )
        impact = float(np.clip(impact, 0.0, 1.0))

        impact_scores.append(impact)
        face_fills.append(ff)
        motions.append(motion_norm)
        contrasts.append(contrast)
        shadows.append(shadow)
        angles.append(angle)

        prev_gray_flow = gray_flow
        frame_idx += 1

    cap.release()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n = max(len(impact_scores), 1)
    ms_per_frame = elapsed_ms / n

    log.info(
        "  → %d frames processed in %.1fms (%.2fms/frame)  impact_mean=%.3f  impact_max=%.3f",
        n, elapsed_ms, ms_per_frame,
        float(np.mean(impact_scores)) if impact_scores else 0.0,
        float(np.max(impact_scores)) if impact_scores else 0.0,
    )

    arr = np.array(impact_scores) if impact_scores else np.zeros(1)
    summary = {
        "impact_mean": round(float(np.mean(arr)), 4),
        "impact_max":  round(float(np.max(arr)), 4),
        "impact_p90":  round(float(np.percentile(arr, 90)), 4),
        "motion_mean": round(float(np.mean(motions)), 4) if motions else 0.0,
        "face_fill_mean": round(float(np.mean(face_fills)), 4) if face_fills else 0.0,
        "contrast_mean": round(float(np.mean(contrasts)), 4) if contrasts else 0.0,
        "shadow_mean": round(float(np.mean(shadows)), 4) if shadows else 0.0,
        "ms_per_frame": round(ms_per_frame, 2),
        "face_detected": face_bbox_cache is not None,
        "mediapipe_available": HAS_MEDIAPIPE,
    }

    return {
        "video_path":        str(path),
        "fps_source":        round(src_fps, 2),
        "fps_processed":     round(src_fps / stride, 2),
        "frame_count":       n,
        "processing_ms":     round(elapsed_ms, 1),
        "ms_per_frame":      round(ms_per_frame, 2),
        "impact_scores":     [round(v, 4) for v in impact_scores],
        "face_fills":        [round(v, 4) for v in face_fills],
        "motion_magnitudes": [round(v, 4) for v in motions],
        "contrasts":         [round(v, 4) for v in contrasts],
        "shadow_ratios":     [round(v, 4) for v in shadows],
        "face_angles":       angles,
        "summary":           summary,
    }


def score_clip_batch(video_paths: list[str]) -> list[dict]:
    results = []
    for p in video_paths:
        try:
            results.append(score_clip_frames(p))
        except Exception as e:
            log.error("Failed %s: %s", p, e)
            results.append({"video_path": p, "error": str(e)})
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_result(r: dict) -> None:
    if "error" in r:
        print(f"  ✗  {Path(r['video_path']).name}: {r['error']}")
        return
    s = r["summary"]
    name = Path(r["video_path"]).name
    print(f"\n  ── {name}")
    print(f"     Frames    : {r['frame_count']}  @{r['fps_processed']:.1f}fps effective  ({r['processing_ms']:.0f}ms total, {s['ms_per_frame']:.2f}ms/frame)")
    print(f"     Impact    : mean={s['impact_mean']:.4f}  max={s['impact_max']:.4f}  p90={s['impact_p90']:.4f}")
    print(f"     Motion    : {s['motion_mean']:.4f}")
    print(f"     Face fill : {s['face_fill_mean']:.4f}  (detected: {s['face_detected']})")
    print(f"     Contrast  : {s['contrast_mean']:.4f}")
    print(f"     Shadow    : {s['shadow_mean']:.4f}")
    print(f"     MediaPipe : {'yes' if s['mediapipe_available'] else 'no (angle=None)'}")
    if r["impact_scores"]:
        peak_idx = int(np.argmax(r["impact_scores"]))
        print(f"     Peak frame: {peak_idx}  score={r['impact_scores'][peak_idx]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ascension Engine — impact_scorer.py")
    parser.add_argument("input", nargs="?", help="Video file or directory")
    parser.add_argument("--batch", action="store_true", help="Process all .mp4 in directory")
    parser.add_argument("--json", metavar="FILE", help="Write results to JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input:
        parser.print_help()
        return

    inp = Path(args.input)
    if inp.is_dir() or args.batch:
        videos = sorted(inp.glob("*.mp4")) if inp.is_dir() else [Path(args.input)]
        results = score_clip_batch([str(v) for v in videos])
    else:
        results = [score_clip_frames(args.input)]

    print(f"\n{'═'*60}")
    print("  IMPACT SCORER RESULTS")
    print(f"{'═'*60}")
    for r in results:
        _print_result(r)
    print()

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        log.info("Written: %s", out)


if __name__ == "__main__":
    main()
