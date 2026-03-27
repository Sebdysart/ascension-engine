#!/usr/bin/env python3
"""
MogNet — Feature Extractor
Extracts visual, audio, and text features from any video for viral scoring.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import cv2
import librosa
import numpy as np

log = logging.getLogger("mognet.feature_extractor")

_ROOT = Path(__file__).resolve().parent.parent.parent  # project root

# Act time boundaries (seconds)
_ACT1_END = 3.0
_ACT2_END = 8.0
_ACT3_END = 15.0


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def _extract_frame_sequence(cap: cv2.VideoCapture, max_frames: int = 300) -> list[dict]:
    """
    Return list of {frame_idx, time_sec, frame_bgr} dicts sampled evenly.
    Capped at max_frames to bound memory.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step  = max(1, total // max_frames)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % step == 0:
            frames.append({
                "frame_idx": fi,
                "time_sec":  fi / fps,
                "frame_bgr": frame,
            })
        fi += 1
    return frames


def _detect_cuts(frames: list[dict]) -> list[float]:
    """
    Scene cut timestamps via histogram difference between adjacent frames.
    Returns list of cut times in seconds.
    """
    cut_times: list[float] = []
    prev_hist = None
    for f in frames:
        gray = cv2.cvtColor(f["frame_bgr"], cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        if prev_hist is not None:
            diff = float(np.sum(np.abs(hist - prev_hist)))
            if diff > 0.35:
                cut_times.append(f["time_sec"])
        prev_hist = hist
    return cut_times


def _cuts_per_sec_in_window(cut_times: list[float], start: float, end: float) -> float:
    window = end - start
    if window <= 0:
        return 0.0
    count = sum(1 for t in cut_times if start <= t < end)
    return round(count / window, 3)


def _brightness_per_clip(frames: list[dict], cut_times: list[float]) -> list[float]:
    """Mean luminance (Y channel) for each clip segment."""
    if not frames:
        return []
    total_dur = frames[-1]["time_sec"] if frames else 15.0
    # Build clip boundaries from cut_times
    boundaries = [0.0] + cut_times + [total_dur + 0.1]
    brightnesses: list[float] = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        seg_frames = [f["frame_bgr"] for f in frames
                      if seg_start <= f["time_sec"] < seg_end]
        if not seg_frames:
            brightnesses.append(0.0)
            continue
        ys = []
        for fr in seg_frames:
            yuv = cv2.cvtColor(fr, cv2.COLOR_BGR2YUV)
            ys.append(float(yuv[:, :, 0].mean()))
        brightnesses.append(round(float(np.mean(ys)), 2))
    return brightnesses


def _color_temp_per_clip(frames: list[dict], cut_times: list[float]) -> list[float]:
    """
    Color temperature proxy per clip: R/B ratio.
    > 1.2 → warm (5600K), < 0.9 → cold (4200K).
    Returns list of R/B ratios.
    """
    if not frames:
        return []
    total_dur = frames[-1]["time_sec"] if frames else 15.0
    boundaries = [0.0] + cut_times + [total_dur + 0.1]
    temps: list[float] = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        seg_frames = [f["frame_bgr"] for f in frames
                      if seg_start <= f["time_sec"] < seg_end]
        if not seg_frames:
            temps.append(1.0)
            continue
        rb_ratios = []
        for fr in seg_frames:
            b_ch = float(fr[:, :, 0].mean()) + 1e-6
            r_ch = float(fr[:, :, 2].mean()) + 1e-6
            rb_ratios.append(r_ch / b_ch)
        temps.append(round(float(np.mean(rb_ratios)), 3))
    return temps


def _detect_shake_events(frames: list[dict]) -> int:
    """Count frames with large optical-flow displacement (shake spike)."""
    if len(frames) < 2:
        return 0
    shake_count = 0
    for i in range(1, len(frames)):
        f1 = cv2.cvtColor(frames[i - 1]["frame_bgr"], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames[i]["frame_bgr"], cv2.COLOR_BGR2GRAY)
        # Resize to small for speed
        f1s = cv2.resize(f1, (160, 90))
        f2s = cv2.resize(f2, (160, 90))
        flow = cv2.calcOpticalFlowFarneback(
            f1s, f2s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
        if mag > 8.0:
            shake_count += 1
    return shake_count


def _detect_zoom_pulses(frames: list[dict]) -> int:
    """
    Detect zoom pulses via center-crop variance spike.
    A zoom pulse = sudden increase in center-crop pixel variance.
    """
    if len(frames) < 3:
        return 0
    variances = []
    for f in frames:
        h, w = f["frame_bgr"].shape[:2]
        cx, cy = w // 2, h // 2
        crop = f["frame_bgr"][cy - h//6:cy + h//6, cx - w//6:cx + w//6]
        variances.append(float(crop.var()))
    pulses = 0
    for i in range(1, len(variances)):
        if variances[i] > variances[i - 1] * 2.5:
            pulses += 1
    return pulses


def _detect_slow_mo(path: str) -> int:
    """
    Count frames that appear to be slow-motion via ffprobe stream r_frame_rate vs avg_frame_rate.
    Heuristic: if reported FPS > 59, count half those frames as slow-mo.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, timeout=10)
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                fps_str = s.get("avg_frame_rate", "30/1")
                num, den = fps_str.split("/")
                fps = float(num) / max(float(den), 1)
                if fps > 59:
                    nb = int(s.get("nb_frames", 0))
                    return nb // 2
    except Exception:
        pass
    return 0


def _detect_direct_stare(frames: list[dict]) -> int:
    """
    Heuristic for direct eye contact: detect face in center third of frame,
    facing roughly forward. Uses cv2 Haar cascade (no mediapipe dep needed here).
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    stare_count = 0
    for f in frames[::3]:  # sample every 3rd frame
        gray = cv2.cvtColor(f["frame_bgr"], cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        for (fx, fy, fw, fh) in faces:
            # Face center must be in middle third horizontally
            cx = fx + fw // 2
            if w // 3 < cx < 2 * w // 3:
                stare_count += 1
    return stare_count


def _camera_angles(frames: list[dict], cut_times: list[float]) -> list[float]:
    """
    Heuristic yaw angle per clip via face detection horizontal position.
    Returns list of yaw values in degrees (0=front, ±45=side).
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if not frames:
        return []
    total_dur = frames[-1]["time_sec"] if frames else 15.0
    boundaries = [0.0] + cut_times + [total_dur + 0.1]
    angles: list[float] = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        seg_frames = [f for f in frames if seg_start <= f["time_sec"] < seg_end]
        if not seg_frames:
            angles.append(0.0)
            continue
        # Use middle frame of segment
        mid = seg_frames[len(seg_frames) // 2]
        gray = cv2.cvtColor(mid["frame_bgr"], cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            angles.append(0.0)
        else:
            # Largest face
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx = fx + fw // 2
            # Map from [0, w] to [-45, 45]
            yaw = ((cx / w) - 0.5) * 90.0
            angles.append(round(yaw, 1))
    return angles


def _angle_inversion_count(angles: list[float]) -> int:
    """Count low→high yaw inversions (victim looking away → mogger looking at camera)."""
    inversions = 0
    for i in range(1, len(angles)):
        if abs(angles[i - 1]) > 20 and abs(angles[i]) < 15:
            inversions += 1
    return inversions


def _brightness_contrast_ratio(brightnesses: list[float]) -> float:
    """
    Ratio of max/min brightness across clips.
    High = victim is washed out vs mogger dark/contrasty.
    """
    if not brightnesses or min(brightnesses) < 1:
        return 1.0
    return round(max(brightnesses) / (min(brightnesses) + 1e-6), 3)


def _color_temp_shift(temps: list[float]) -> bool:
    """True if there's a warm→cold transition across clips."""
    if len(temps) < 2:
        return False
    warm_first = any(t > 1.15 for t in temps[:max(1, len(temps)//2)])
    cold_second = any(t < 0.95 for t in temps[len(temps)//2:])
    return warm_first and cold_second


def _audio_features(path: str, max_sec: float = 22.0) -> dict:
    """Extract BPM, beat drops, silence gaps, build detection, drop intensity."""
    try:
        import librosa
    except ImportError:
        return {
            "bpm": 108.0, "drop_timestamps": [], "silence_gaps_before_drop": [],
            "avg_silence_gap_ms": 0.0, "build_detected": False, "drop_intensity_db": -20.0,
        }

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "22050", "-vn", wav_path],
            capture_output=True, timeout=30)
        y, sr = librosa.load(wav_path, sr=22050, duration=max_sec)
    except Exception:
        return {
            "bpm": 108.0, "drop_timestamps": [], "silence_gaps_before_drop": [],
            "avg_silence_gap_ms": 0.0, "build_detected": False, "drop_intensity_db": -20.0,
        }
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
    beat_times = [float(t) for t in beat_frames]

    # Drop detection: highest onset strength regions after 7s
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    drop_mask = onset_times >= 7.0
    if drop_mask.any():
        drop_env = onset_env[drop_mask]
        drop_t   = onset_times[drop_mask]
        threshold = drop_env.mean() + 1.5 * drop_env.std()
        drop_timestamps = [float(drop_t[i]) for i in range(len(drop_env))
                           if drop_env[i] > threshold]
        # Cluster: keep only first in each 0.5s window
        clustered: list[float] = []
        for dt in sorted(drop_timestamps):
            if not clustered or dt - clustered[-1] > 0.5:
                clustered.append(dt)
        drop_timestamps = clustered[:4]
    else:
        drop_timestamps = []

    # Silence gaps before drops: check 0.5s window before each drop
    silence_gaps: list[float] = []
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    for dt in drop_timestamps:
        window_mask = (rms_times >= dt - 0.5) & (rms_times < dt)
        if window_mask.any():
            rms_window = rms[window_mask]
            silence_frames = rms_window[rms_window < rms_window.mean() * 0.2]
            gap_ms = (len(silence_frames) / sr) * hop * 1000
            silence_gaps.append(round(gap_ms, 1))

    avg_gap = round(float(np.mean(silence_gaps)), 1) if silence_gaps else 0.0

    # Build detection: increasing RMS over 3–8s window
    build_mask = (rms_times >= 3.0) & (rms_times < 8.0)
    build_detected = False
    if build_mask.sum() >= 4:
        build_rms = rms[build_mask]
        build_detected = bool(np.polyfit(np.arange(len(build_rms)), build_rms, 1)[0] > 0)

    # Drop intensity: peak dB in drop region
    drop_intensity_db = -60.0
    if drop_timestamps:
        drop_start = drop_timestamps[0]
        drop_mask2 = (rms_times >= drop_start) & (rms_times < drop_start + 1.0)
        if drop_mask2.any():
            peak_rms = float(rms[drop_mask2].max())
            drop_intensity_db = round(20 * np.log10(peak_rms + 1e-9), 1)

    return {
        "bpm":                      round(bpm, 1),
        "drop_timestamps":          drop_timestamps,
        "silence_gaps_before_drop": silence_gaps,
        "avg_silence_gap_ms":       avg_gap,
        "build_detected":           build_detected,
        "drop_intensity_db":        round(drop_intensity_db, 1),
    }


def _ocr_hook_text(path: str) -> str:
    """Extract text from first 15 frames via EasyOCR."""
    try:
        import easyocr
    except ImportError:
        return ""
    try:
        cap = cv2.VideoCapture(path)
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        texts: list[str] = []
        for i in range(15):
            ret, frame = cap.read()
            if not ret:
                break
            results = reader.readtext(frame, detail=0)
            texts.extend(results)
        cap.release()
        return " ".join(texts).strip()
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return ""


def _score_aggression(text: str) -> float:
    """
    Score hook text aggression 0–10 via claude CLI.
    Returns 5.0 on any failure (neutral default).
    """
    if not text.strip():
        return 5.0
    prompt = (
        f'Rate the aggression/provocation level of this TikTok hook text on a scale '
        f'0-10 (0=passive/informative, 10=maximally aggressive/confrontational). '
        f'Output ONLY a single float number.\n\nText: "{text}"'
    )
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=15)
        val = float(r.stdout.strip().split()[0])
        return round(max(0.0, min(10.0, val)), 1)
    except Exception:
        # Fallback heuristic: count aggressive words
        agg_words = {"you", "ugly", "weak", "loser", "cope", "recessed",
                     "mogged", "below", "average", "worst", "failed", "never"}
        words = set(text.lower().split())
        score = min(10.0, len(words & agg_words) * 1.5 + 3.0)
        return round(score, 1)


def extract_video_features(video_path: str) -> dict:
    """
    Extract all features needed for viral score model from a video file.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file.

    Returns
    -------
    dict with keys: visual, audio, text
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = _open_video(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = total_frames / fps

    frames = _extract_frame_sequence(cap, max_frames=200)
    cap.release()

    cut_times = _detect_cuts(frames)

    # ── Visual features ────────────────────────────────────────────────────────
    cps_overall = _cuts_per_sec_in_window(cut_times, 0.0, total_dur)
    cps_act1    = _cuts_per_sec_in_window(cut_times, 0.0, _ACT1_END)
    cps_act2    = _cuts_per_sec_in_window(cut_times, _ACT1_END, _ACT2_END)
    cps_act3    = _cuts_per_sec_in_window(cut_times, _ACT2_END, min(_ACT3_END, total_dur))

    brightnesses = _brightness_per_clip(frames, cut_times)
    temps        = _color_temp_per_clip(frames, cut_times)
    angles       = _camera_angles(frames, cut_times)

    visual = {
        "cuts_per_second":          cps_overall,
        "cuts_per_second_act1":     cps_act1,
        "cuts_per_second_act2":     cps_act2,
        "cuts_per_second_act3":     cps_act3,
        "camera_angles":            angles,
        "angle_inversion_count":    _angle_inversion_count(angles),
        "brightness_per_clip":      brightnesses,
        "brightness_contrast_ratio": _brightness_contrast_ratio(brightnesses),
        "color_temp_per_clip":      temps,
        "color_temp_shift":         _color_temp_shift(temps),
        "zoom_pulse_count":         _detect_zoom_pulses(frames),
        "shake_events":             _detect_shake_events(frames),
        "slow_mo_frames":           _detect_slow_mo(str(p)),
        "direct_stare_clips":       _detect_direct_stare(frames),
    }

    # ── Audio features ─────────────────────────────────────────────────────────
    audio = _audio_features(str(p))

    # ── Text features ─────────────────────────────────────────────────────────
    hook_text = _ocr_hook_text(str(p))
    aggression = _score_aggression(hook_text)
    has_second_person = any(w in hook_text.lower().split()
                             for w in ["you", "your", "you're", "you've", "you'll"])
    text_density = round(len(hook_text) / max(15.0, 1.0), 2)

    text = {
        "hook_text":             hook_text,
        "hook_aggression_score": aggression,
        "has_second_person":     has_second_person,
        "text_density":          text_density,
    }

    return {"visual": visual, "audio": audio, "text": text}
