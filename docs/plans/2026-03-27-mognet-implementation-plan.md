# MogNet Self-Validating Edit Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 5-module intelligence layer (`data/mognet/`) that extracts viral features from video, trains an XGBoost viral scorer on gold references, validates generated edits with rule-based critiques, and wires a reject-and-retry loop into `generate_batch.py`.

**Architecture:** Feature extractor feeds a trained XGBoost regressor for viral score prediction (0–100). The validator combines model score + 5 rule-based critiques to APPROVE/REJECT edits. `generate_batch.py` retries rejected edits up to 3× with param jitter. A feedback loop table (`mognet_performance` in `engine.db`) collects actuals for future retraining.

**Tech Stack:** Python 3.11, OpenCV (`cv2`), librosa, EasyOCR, XGBoost, LightGBM, joblib, SQLite via existing `EngineDB`, `claude` CLI subprocess for aggression scoring.

---

## Pre-flight

The worktree is at `/Users/sebastiandysart/Desktop/ascension-engine/.claude/worktrees/affectionate-hodgkin`.
All commands run from ROOT = that directory.
Gold videos are at `input/gold/*.mp4` (10 files exist).
The existing `EngineDB` class is in `data/engine_db.py`.

Pull latest from main first:
```bash
cd /Users/sebastiandysart/Desktop/ascension-engine/.claude/worktrees/affectionate-hodgkin
git fetch origin main && git merge origin/main
```

Add new deps to `requirements.txt`:
```
xgboost
lightgbm
joblib
```

Install:
```bash
pip install xgboost lightgbm joblib
```

---

## Task 1: `data/mognet/__init__.py` — Package scaffold

**Files:**
- Create: `data/mognet/__init__.py`

**Step 1: Create the package init**

```python
"""MogNet — self-validating edit intelligence layer."""
```

**Step 2: Verify import works**

```bash
python -c "import sys; sys.path.insert(0,'data'); from mognet import __doc__; print(__doc__)"
```
Expected: `MogNet — self-validating edit intelligence layer.`

**Step 3: Commit**

```bash
git add data/mognet/__init__.py requirements.txt
git commit -m "feat(mognet): scaffold package + add xgboost/lightgbm/joblib deps"
```

---

## Task 2: `data/mognet/feature_extractor.py` — Video feature extraction

**Files:**
- Create: `data/mognet/feature_extractor.py`
- Test: `tests/test_feature_extractor.py`

### Step 1: Write the failing test

```python
# tests/test_feature_extractor.py
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.feature_extractor import extract_video_features

GOLD_DIR = Path(__file__).resolve().parent.parent / "input" / "gold"
GOLD_VIDEOS = list(GOLD_DIR.glob("*.mp4"))


def test_extract_returns_required_keys():
    """Feature extractor must return all three top-level keys."""
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    assert "visual" in feats
    assert "audio" in feats
    assert "text" in feats


def test_visual_keys_present():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    v = feats["visual"]
    for key in ["cuts_per_second", "cuts_per_second_act1", "cuts_per_second_act2",
                "cuts_per_second_act3", "camera_angles", "angle_inversion_count",
                "brightness_per_clip", "brightness_contrast_ratio",
                "color_temp_per_clip", "color_temp_shift",
                "zoom_pulse_count", "shake_events", "slow_mo_frames",
                "direct_stare_clips"]:
        assert key in v, f"Missing visual key: {key}"


def test_audio_keys_present():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    a = feats["audio"]
    for key in ["bpm", "drop_timestamps", "silence_gaps_before_drop",
                "avg_silence_gap_ms", "build_detected", "drop_intensity_db"]:
        assert key in a, f"Missing audio key: {key}"


def test_text_keys_present():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    t = feats["text"]
    for key in ["hook_text", "hook_aggression_score", "has_second_person", "text_density"]:
        assert key in t, f"Missing text key: {key}"


def test_numeric_values_are_floats_or_ints():
    if not GOLD_VIDEOS:
        pytest.skip("No gold videos available")
    feats = extract_video_features(str(GOLD_VIDEOS[0]))
    assert isinstance(feats["visual"]["cuts_per_second"], float)
    assert isinstance(feats["audio"]["bpm"], float)
    assert isinstance(feats["text"]["hook_aggression_score"], float)


def test_invalid_path_raises():
    with pytest.raises(Exception):
        extract_video_features("/nonexistent/path.mp4")
```

### Step 2: Run to verify it fails

```bash
pytest tests/test_feature_extractor.py -v 2>&1 | head -30
```
Expected: `ImportError` or `ModuleNotFoundError` for `mognet.feature_extractor`.

### Step 3: Write the implementation

```python
# data/mognet/feature_extractor.py
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
```

### Step 4: Run tests

```bash
pytest tests/test_feature_extractor.py -v
```
Expected: all 5 tests PASS (or skip if no gold videos).

### Step 5: Commit

```bash
git add data/mognet/feature_extractor.py tests/test_feature_extractor.py
git commit -m "feat(mognet): feature_extractor — visual/audio/text feature extraction"
```

---

## Task 3: `data/mognet/viral_scorer.py` — XGBoost viral score model

**Files:**
- Create: `data/mognet/viral_scorer.py`
- Test: `tests/test_viral_scorer.py`

### Step 1: Write the failing test

```python
# tests/test_viral_scorer.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.viral_scorer import ViralScorer


def _make_dummy_features(n: int = 10) -> list[dict]:
    return [
        {
            "visual": {
                "cuts_per_second": 2.0 + i * 0.3,
                "cuts_per_second_act1": 1.0,
                "cuts_per_second_act2": 8.0 + i,
                "cuts_per_second_act3": 3.0,
                "camera_angles": [0.0, 15.0, -10.0],
                "angle_inversion_count": i % 3,
                "brightness_per_clip": [120.0, 80.0, 150.0],
                "brightness_contrast_ratio": 1.5 + i * 0.1,
                "color_temp_per_clip": [1.2, 0.9, 0.8],
                "color_temp_shift": bool(i % 2),
                "zoom_pulse_count": i % 4,
                "shake_events": i % 5,
                "slow_mo_frames": i % 3,
                "direct_stare_clips": i % 4,
            },
            "audio": {
                "bpm": 108.0 + i,
                "drop_timestamps": [8.0, 11.0],
                "silence_gaps_before_drop": [50.0 + i * 5],
                "avg_silence_gap_ms": 50.0 + i * 5,
                "build_detected": bool(i % 2),
                "drop_intensity_db": -15.0 + i,
            },
            "text": {
                "hook_text": "you are below average",
                "hook_aggression_score": 5.0 + i * 0.3,
                "has_second_person": True,
                "text_density": 1.5,
            },
        }
        for i in range(n)
    ]


def _make_dummy_metrics(n: int = 10) -> list[dict]:
    return [
        {"views": 10000 * (i + 1), "watch_pct": 0.4 + i * 0.05,
         "shares": 100 * i, "saves": 50 * i}
        for i in range(n)
    ]


def test_train_and_predict():
    scorer = ViralScorer()
    features = _make_dummy_features(10)
    metrics  = _make_dummy_metrics(10)
    scorer.train(features, metrics)
    result = scorer.predict(features[0])
    assert "score" in result
    assert "confidence" in result
    assert "breakdown" in result
    assert 0.0 <= result["score"] <= 100.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_without_train_raises():
    scorer = ViralScorer()
    with pytest.raises(RuntimeError, match="not trained"):
        scorer.predict(_make_dummy_features(1)[0])


def test_save_and_load(tmp_path):
    scorer = ViralScorer()
    scorer.train(_make_dummy_features(10), _make_dummy_metrics(10))
    pkl_path = str(tmp_path / "scorer.pkl")
    scorer.save(pkl_path)
    scorer2 = ViralScorer()
    scorer2.load(pkl_path)
    result = scorer2.predict(_make_dummy_features(1)[0])
    assert 0.0 <= result["score"] <= 100.0


def test_breakdown_has_expected_keys():
    scorer = ViralScorer()
    scorer.train(_make_dummy_features(10), _make_dummy_metrics(10))
    result = scorer.predict(_make_dummy_features(1)[0])
    bd = result["breakdown"]
    assert "cut_rate_contribution" in bd
    assert "audio_contribution" in bd
    assert "hook_contribution" in bd
```

### Step 2: Run to verify it fails

```bash
pytest tests/test_viral_scorer.py -v 2>&1 | head -20
```
Expected: `ImportError` for `mognet.viral_scorer`.

### Step 3: Write the implementation

```python
# data/mognet/viral_scorer.py
#!/usr/bin/env python3
"""
MogNet — Viral Score Model
XGBoost regressor trained on reference video features + engagement metrics.
Ensemble with LightGBM for robustness with small training sets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np

log = logging.getLogger("mognet.viral_scorer")

_FEATURE_NAMES = [
    "cuts_per_second",
    "cuts_per_second_act1",
    "cuts_per_second_act2",
    "cuts_per_second_act3",
    "angle_inversion_count",
    "brightness_contrast_ratio",
    "color_temp_shift",
    "zoom_pulse_count",
    "shake_events",
    "slow_mo_frames",
    "direct_stare_clips",
    "bpm",
    "avg_silence_gap_ms",
    "build_detected",
    "drop_intensity_db",
    "hook_aggression_score",
    "has_second_person",
    "text_density",
]


def _engagement_rate(m: dict) -> float:
    views     = float(m.get("views", 0))
    watch_pct = float(m.get("watch_pct", 0.0))
    shares    = float(m.get("shares", 0))
    saves     = float(m.get("saves", 0))
    raw = views * watch_pct + shares * 10 + saves * 5
    # Normalise to 0–100 (log scale, capped at 1M engagement)
    import math
    return round(min(100.0, max(0.0, math.log1p(raw) / math.log1p(1_000_000) * 100)), 2)


def _features_to_vector(f: dict) -> np.ndarray:
    v = f.get("visual", {})
    a = f.get("audio", {})
    t = f.get("text", {})
    row = [
        float(v.get("cuts_per_second", 0)),
        float(v.get("cuts_per_second_act1", 0)),
        float(v.get("cuts_per_second_act2", 0)),
        float(v.get("cuts_per_second_act3", 0)),
        float(v.get("angle_inversion_count", 0)),
        float(v.get("brightness_contrast_ratio", 1.0)),
        float(v.get("color_temp_shift", False)),
        float(v.get("zoom_pulse_count", 0)),
        float(v.get("shake_events", 0)),
        float(v.get("slow_mo_frames", 0)),
        float(v.get("direct_stare_clips", 0)),
        float(a.get("bpm", 108)),
        float(a.get("avg_silence_gap_ms", 0)),
        float(a.get("build_detected", False)),
        float(a.get("drop_intensity_db", -60)),
        float(t.get("hook_aggression_score", 5.0)),
        float(t.get("has_second_person", False)),
        float(t.get("text_density", 0)),
    ]
    return np.array(row, dtype=np.float32)


class ViralScorer:
    """
    Viral score predictor.

    Usage
    -----
        scorer = ViralScorer()
        scorer.train(features_list, metrics_list)
        result = scorer.predict(feature_dict)
        # {"score": 82.3, "confidence": 0.74, "breakdown": {...}}

        scorer.save("data/mognet/viral_scorer.pkl")
        scorer.load("data/mognet/viral_scorer.pkl")
    """

    def __init__(self):
        self._model = None          # XGBRegressor (primary)
        self._lgbm  = None          # LGBMRegressor (ensemble)
        self._trained = False

    def train(self, reference_features: list[dict], performance_metrics: list[dict]):
        """
        Train on reference data.

        Parameters
        ----------
        reference_features : list of feature dicts from extract_video_features()
        performance_metrics : list of dicts with keys: views, watch_pct, shares, saves
        """
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        if len(reference_features) != len(performance_metrics):
            raise ValueError("features and metrics must have the same length")
        if len(reference_features) < 2:
            raise ValueError("Need at least 2 training samples")

        X = np.array([_features_to_vector(f) for f in reference_features])
        y = np.array([_engagement_rate(m) for m in performance_metrics])

        n_estimators = min(100, max(10, len(X) * 3))

        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X, y)

        # Optional LightGBM ensemble
        try:
            from lightgbm import LGBMRegressor
            self._lgbm = LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=-1,
            )
            self._lgbm.fit(X, y)
        except Exception as e:
            log.warning("LightGBM ensemble skipped: %s", e)
            self._lgbm = None

        self._trained = True
        log.info("ViralScorer trained on %d samples (y range %.1f–%.1f)",
                 len(X), y.min(), y.max())

    def predict(self, edit_features: dict) -> dict:
        """
        Predict viral score for a single edit.

        Returns
        -------
        {
            "score": float (0–100),
            "confidence": float (0–1),
            "breakdown": {
                "cut_rate_contribution": float,
                "audio_contribution":    float,
                "hook_contribution":     float,
            }
        }
        """
        if not self._trained or self._model is None:
            raise RuntimeError("ViralScorer is not trained — call train() or load() first")

        x = _features_to_vector(edit_features).reshape(1, -1)

        xgb_score = float(self._model.predict(x)[0])

        if self._lgbm is not None:
            lgbm_score = float(self._lgbm.predict(x)[0])
            raw_score = (xgb_score * 0.6 + lgbm_score * 0.4)
            confidence = 0.85
        else:
            raw_score = xgb_score
            confidence = 0.70

        score = round(max(0.0, min(100.0, raw_score)), 1)

        # Feature importance breakdown (approximate contribution groups)
        try:
            importances = self._model.feature_importances_
            cut_idx   = [0, 1, 2, 3]
            audio_idx = [11, 12, 13, 14]
            hook_idx  = [15, 16, 17]
            total_imp = importances.sum() + 1e-9
            cut_contrib   = round(float(importances[cut_idx].sum() / total_imp), 3)
            audio_contrib = round(float(importances[audio_idx].sum() / total_imp), 3)
            hook_contrib  = round(float(importances[hook_idx].sum() / total_imp), 3)
        except Exception:
            cut_contrib = audio_contrib = hook_contrib = 0.333

        return {
            "score":      score,
            "confidence": round(confidence, 2),
            "breakdown": {
                "cut_rate_contribution": cut_contrib,
                "audio_contribution":    audio_contrib,
                "hook_contribution":     hook_contrib,
            },
        }

    def save(self, path: str):
        if not self._trained:
            raise RuntimeError("Cannot save untrained scorer")
        payload = {"model": self._model, "lgbm": self._lgbm, "trained": True}
        joblib.dump(payload, path)
        log.info("ViralScorer saved to %s", path)

    def load(self, path: str):
        payload = joblib.load(path)
        self._model   = payload["model"]
        self._lgbm    = payload.get("lgbm")
        self._trained = payload.get("trained", True)
        log.info("ViralScorer loaded from %s", path)
```

### Step 4: Run tests

```bash
pytest tests/test_viral_scorer.py -v
```
Expected: all 4 tests PASS.

### Step 5: Commit

```bash
git add data/mognet/viral_scorer.py tests/test_viral_scorer.py
git commit -m "feat(mognet): viral_scorer — XGBoost+LightGBM engagement predictor"
```

---

## Task 4: `data/mognet/validator.py` — Self-critique loop

**Files:**
- Create: `data/mognet/validator.py`
- Test: `tests/test_validator.py`

### Step 1: Write the failing test

```python
# tests/test_validator.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.validator import validate_edit, _apply_rule_critiques
from mognet.viral_scorer import ViralScorer


def _make_scorer(score_override: float = 80.0) -> ViralScorer:
    """Return a trained scorer that always predicts score_override."""
    class _MockScorer(ViralScorer):
        def predict(self, feats):
            return {
                "score": score_override,
                "confidence": 0.8,
                "breakdown": {
                    "cut_rate_contribution": 0.4,
                    "audio_contribution": 0.35,
                    "hook_contribution": 0.25,
                },
            }
    s = _MockScorer()
    s._trained = True
    return s


def _make_good_features() -> dict:
    return {
        "visual": {
            "cuts_per_second": 4.0, "cuts_per_second_act1": 1.2,
            "cuts_per_second_act2": 8.5, "cuts_per_second_act3": 3.0,
            "camera_angles": [30.0, 5.0, -5.0], "angle_inversion_count": 2,
            "brightness_per_clip": [130.0, 70.0, 160.0],
            "brightness_contrast_ratio": 1.6, "color_temp_per_clip": [1.2, 0.85],
            "color_temp_shift": True, "zoom_pulse_count": 2,
            "shake_events": 3, "slow_mo_frames": 10, "direct_stare_clips": 4,
        },
        "audio": {
            "bpm": 114.0, "drop_timestamps": [8.0, 11.0],
            "silence_gaps_before_drop": [55.0], "avg_silence_gap_ms": 55.0,
            "build_detected": True, "drop_intensity_db": -10.0,
        },
        "text": {
            "hook_text": "you are below average", "hook_aggression_score": 7.5,
            "has_second_person": True, "text_density": 1.8,
        },
    }


def _make_bad_features() -> dict:
    f = _make_good_features()
    f["audio"]["avg_silence_gap_ms"] = 20.0    # too short
    f["visual"]["angle_inversion_count"] = 0    # missing inversion
    f["text"]["hook_aggression_score"] = 4.0    # too passive
    f["visual"]["brightness_contrast_ratio"] = 1.1   # not washed out
    f["visual"]["cuts_per_second_act2"] = 3.0   # too slow
    return f


def test_approve_on_good_features(tmp_path):
    """High scorer + good features → APPROVE."""
    # Create dummy mp4 placeholder (validator reads path, we mock extract)
    from unittest.mock import patch
    scorer = _make_scorer(82.0)
    good = _make_good_features()
    with patch("mognet.validator.extract_video_features", return_value=good):
        result = validate_edit(str(tmp_path / "edit.mp4"), scorer)
    assert result["decision"] == "APPROVE"
    assert result["viral_score"] == 82.0


def test_reject_on_low_score(tmp_path):
    """Score < 75 → auto-REJECT regardless of rules."""
    from unittest.mock import patch
    scorer = _make_scorer(60.0)
    good = _make_good_features()
    with patch("mognet.validator.extract_video_features", return_value=good):
        result = validate_edit(str(tmp_path / "edit.mp4"), scorer)
    assert result["decision"] == "REJECT"


def test_critiques_on_bad_features():
    """All 5 rule critiques fire on bad features."""
    bad = _make_bad_features()
    critiques, warnings = _apply_rule_critiques(bad)
    # Should have 2 CRITICALs and 3 WARNINGs
    combined = critiques + warnings
    assert any("silence gap" in c.lower() for c in combined)
    assert any("angle" in c.lower() or "power hierarchy" in c.lower() for c in combined)
    assert any("hook" in c.lower() or "passive" in c.lower() for c in combined)
    assert any("victim" in c.lower() or "washed" in c.lower() or "brightness" in c.lower() for c in combined)
    assert any("awakening" in c.lower() or "act2" in c.lower() or "slow" in c.lower() for c in combined)


def test_result_keys():
    """validate_edit returns all required keys."""
    from unittest.mock import patch
    scorer = _make_scorer(80.0)
    with patch("mognet.validator.extract_video_features", return_value=_make_good_features()):
        result = validate_edit("/fake/path.mp4", scorer)
    for key in ["decision", "viral_score", "strengths", "warnings", "critiques"]:
        assert key in result, f"Missing key: {key}"


def test_reject_on_critical_rule(tmp_path):
    """CRITICAL rule should force REJECT even with high score."""
    from unittest.mock import patch
    scorer = _make_scorer(85.0)
    bad = _make_good_features()
    bad["visual"]["angle_inversion_count"] = 0  # CRITICAL
    bad["visual"]["cuts_per_second_act2"] = 2.0  # CRITICAL
    with patch("mognet.validator.extract_video_features", return_value=bad):
        result = validate_edit(str(tmp_path / "edit.mp4"), scorer)
    assert result["decision"] == "REJECT"
```

### Step 2: Run to verify it fails

```bash
pytest tests/test_validator.py -v 2>&1 | head -20
```
Expected: `ImportError` for `mognet.validator`.

### Step 3: Write the implementation

```python
# data/mognet/validator.py
#!/usr/bin/env python3
"""
MogNet — Edit Validator
Self-critique loop: extracts features from a generated edit, scores via
ViralScorer, applies rule-based critiques, and returns APPROVE/REJECT.
"""
from __future__ import annotations

import logging
from pathlib import Path

from mognet.feature_extractor import extract_video_features
from mognet.viral_scorer import ViralScorer

log = logging.getLogger("mognet.validator")

# APPROVE threshold — edits scoring below this are auto-rejected
APPROVE_THRESHOLD = 75.0


def _apply_rule_critiques(features: dict) -> tuple[list[str], list[str]]:
    """
    Apply the 5 mandatory rule-based critiques.

    Returns
    -------
    (critiques, warnings) — critiques are CRITICAL failures, warnings are advisory
    """
    v = features.get("visual", {})
    a = features.get("audio", {})
    t = features.get("text", {})

    critiques: list[str] = []
    warnings:  list[str] = []

    # Rule 1: Silence gap too short
    gap_ms = float(a.get("avg_silence_gap_ms", 0.0))
    if gap_ms > 0 and gap_ms < 40.0:
        warnings.append(
            f"WARNING: silence gap too short ({gap_ms:.0f}ms < 40ms) — "
            "increase pre-drop breath room for impact"
        )

    # Rule 2: No angle inversion (CRITICAL)
    inversions = int(v.get("angle_inversion_count", 0))
    if inversions == 0:
        critiques.append(
            "CRITICAL: no angle inversion detected — breaks power hierarchy; "
            "victim must look away before mogger faces camera"
        )

    # Rule 3: Hook aggression too low
    aggression = float(t.get("hook_aggression_score", 5.0))
    if aggression < 6.0:
        warnings.append(
            f"WARNING: hook too passive (aggression={aggression:.1f}/10) — "
            "opening text must provoke or confront viewer directly"
        )

    # Rule 4: Brightness contrast too low (victim not washed out enough)
    contrast = float(v.get("brightness_contrast_ratio", 1.0))
    if contrast < 1.3:
        warnings.append(
            f"WARNING: victim not washed out enough (brightness_contrast={contrast:.2f} < 1.3) — "
            "increase victim clip exposure or darken mogger clips"
        )

    # Rule 5: Act 2 (awakening) too slow (CRITICAL)
    cps_act2 = float(v.get("cuts_per_second_act2", 0.0))
    if cps_act2 < 5.0:
        critiques.append(
            f"CRITICAL: awakening too slow (act2 cuts={cps_act2:.1f}/sec < 5) — "
            "Act 2 (3–8s) must build rapid-fire energy at ≥5 cuts/sec"
        )

    return critiques, warnings


def _identify_strengths(features: dict, viral_score: float) -> list[str]:
    """Return positive observations about the edit."""
    v = features.get("visual", {})
    a = features.get("audio", {})
    t = features.get("text", {})
    strengths: list[str] = []

    if v.get("color_temp_shift"):
        strengths.append("Warm→cold color temperature shift present — cinematic arc")
    if int(v.get("angle_inversion_count", 0)) >= 2:
        strengths.append(f"Strong angle inversions ({v['angle_inversion_count']}) — power hierarchy established")
    if a.get("build_detected"):
        strengths.append("Audio build detected in Act 2 — phonk tension arc present")
    if float(t.get("hook_aggression_score", 0)) >= 7.0:
        strengths.append(f"High hook aggression ({t['hook_aggression_score']}/10) — scroll-stopping")
    if float(v.get("brightness_contrast_ratio", 1.0)) >= 1.5:
        strengths.append("Strong brightness contrast — victim/mogger polarity clear")
    if int(v.get("zoom_pulse_count", 0)) >= 2:
        strengths.append(f"Zoom pulses ({v['zoom_pulse_count']}) aligned with drops")
    if viral_score >= 85:
        strengths.append(f"High viral score ({viral_score:.0f}/100) — model predicts strong performance")

    return strengths if strengths else ["Features within acceptable range"]


def validate_edit(edit_path: str, scorer: ViralScorer) -> dict:
    """
    Validate a generated edit.

    Parameters
    ----------
    edit_path : str
        Path to the generated .mp4 file.
    scorer : ViralScorer
        Trained scorer instance.

    Returns
    -------
    {
        "decision":    "APPROVE" | "REJECT",
        "viral_score": float,
        "strengths":   List[str],
        "warnings":    List[str],
        "critiques":   List[str],   # CRITICAL fix suggestions
    }
    """
    log.info("Validating: %s", edit_path)

    features = extract_video_features(edit_path)
    score_result = scorer.predict(features)
    viral_score = score_result["score"]

    critiques, warnings = _apply_rule_critiques(features)
    strengths = _identify_strengths(features, viral_score)

    # Decision logic:
    # - Score < APPROVE_THRESHOLD → always REJECT
    # - Any CRITICAL critique → REJECT (even if score passes)
    # - Otherwise → APPROVE
    has_critical = len(critiques) > 0
    score_pass   = viral_score >= APPROVE_THRESHOLD

    if not score_pass:
        decision = "REJECT"
        critiques.append(
            f"REJECT: viral score {viral_score:.1f} below threshold {APPROVE_THRESHOLD}"
        )
    elif has_critical:
        decision = "REJECT"
    else:
        decision = "APPROVE"

    log.info("Validation result: %s  score=%.1f  critiques=%d  warnings=%d",
             decision, viral_score, len(critiques), len(warnings))

    return {
        "decision":    decision,
        "viral_score": viral_score,
        "strengths":   strengths,
        "warnings":    warnings,
        "critiques":   critiques,
    }
```

### Step 4: Run tests

```bash
pytest tests/test_validator.py -v
```
Expected: all 5 tests PASS.

### Step 5: Commit

```bash
git add data/mognet/validator.py tests/test_validator.py
git commit -m "feat(mognet): validator — rule-based APPROVE/REJECT with 5 critique rules"
```

---

## Task 5: `data/mognet/feedback_loop.py` + `mognet_performance` DB table

**Files:**
- Create: `data/mognet/feedback_loop.py`
- Modify: `data/engine_db.py` — add `mognet_performance` table to `init()`
- Test: `tests/test_feedback_loop.py`

### Step 1: Add `mognet_performance` table to `EngineDB.init()`

In `data/engine_db.py`, find the `executescript` call inside `init()` and add this table **before** the `CREATE INDEX` lines:

```python
            CREATE TABLE IF NOT EXISTS mognet_performance (
                edit_id         TEXT PRIMARY KEY,
                edit_path       TEXT,
                predicted_score REAL,
                actual_score    REAL,
                views           INTEGER DEFAULT 0,
                shares          INTEGER DEFAULT 0,
                saves           INTEGER DEFAULT 0,
                watch_pct       REAL    DEFAULT 0.0,
                features_json   TEXT,
                created_at      TEXT    DEFAULT (datetime('now'))
            );
```

Also add these two new methods to `EngineDB` (after `approve_generation`):

```python
    # ── MogNet performance tracking ───────────────────────────────────────────

    def save_mognet_prediction(self, edit_id: str, edit_path: str,
                                predicted_score: float, features_json: str = ""):
        self.conn.execute(
            "INSERT OR REPLACE INTO mognet_performance "
            "(edit_id, edit_path, predicted_score, features_json) "
            "VALUES (?, ?, ?, ?)",
            (edit_id, edit_path, predicted_score, features_json)
        )
        self.conn.commit()

    def update_mognet_actuals(self, edit_id: str, views: int, shares: int,
                               saves: int, watch_pct: float):
        actual_score = views * watch_pct + shares * 10 + saves * 5
        self.conn.execute(
            "UPDATE mognet_performance SET views=?, shares=?, saves=?, "
            "watch_pct=?, actual_score=? WHERE edit_id=?",
            (views, shares, saves, watch_pct, actual_score, edit_id)
        )
        self.conn.commit()

    def get_mognet_training_rows(self) -> list[dict]:
        """Return rows with both predicted and actual scores for retraining."""
        rows = self.conn.execute(
            "SELECT edit_id, features_json, actual_score FROM mognet_performance "
            "WHERE actual_score IS NOT NULL AND features_json != '' "
            "ORDER BY created_at DESC"
        ).fetchall()
        return [{"edit_id": r[0], "features_json": r[1], "actual_score": r[2]}
                for r in rows]
```

### Step 2: Write the failing test

```python
# tests/test_feedback_loop.py
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from engine_db import EngineDB
from mognet.feedback_loop import record_actual_performance, retrain_from_feedback
from mognet.viral_scorer import ViralScorer


def _make_temp_db() -> EngineDB:
    tmp = tempfile.mktemp(suffix=".db")
    db = EngineDB(db_path=Path(tmp))
    db.init()
    return db


def _make_dummy_features_json() -> str:
    return json.dumps({
        "visual": {
            "cuts_per_second": 3.0, "cuts_per_second_act1": 1.0,
            "cuts_per_second_act2": 8.0, "cuts_per_second_act3": 3.0,
            "camera_angles": [0.0], "angle_inversion_count": 2,
            "brightness_per_clip": [120.0], "brightness_contrast_ratio": 1.5,
            "color_temp_per_clip": [1.1], "color_temp_shift": True,
            "zoom_pulse_count": 2, "shake_events": 3,
            "slow_mo_frames": 5, "direct_stare_clips": 2,
        },
        "audio": {
            "bpm": 114.0, "drop_timestamps": [8.0], "silence_gaps_before_drop": [55.0],
            "avg_silence_gap_ms": 55.0, "build_detected": True, "drop_intensity_db": -12.0,
        },
        "text": {
            "hook_text": "you are below average", "hook_aggression_score": 7.5,
            "has_second_person": True, "text_density": 1.5,
        },
    })


def test_record_and_retrieve(monkeypatch):
    db = _make_temp_db()
    monkeypatch.setattr("mognet.feedback_loop._get_db", lambda: db)
    record_actual_performance("edit_001", views=50000, shares=200, saves=100, watch_pct=0.45, db=db)
    rows = db.get_mognet_training_rows()
    # No features_json yet so row won't appear in training set — that's fine
    assert isinstance(rows, list)


def test_mognet_performance_table_exists():
    db = _make_temp_db()
    tables = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mognet_performance'"
    ).fetchall()
    assert len(tables) == 1, "mognet_performance table not created"


def test_save_and_retrieve_prediction():
    db = _make_temp_db()
    db.save_mognet_prediction("edit_test", "/out/test.mp4", 82.5, _make_dummy_features_json())
    row = db.conn.execute(
        "SELECT predicted_score FROM mognet_performance WHERE edit_id='edit_test'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 82.5) < 0.1


def test_retrain_skips_when_no_data(monkeypatch):
    """retrain_from_feedback should log and return cleanly when no data."""
    db = _make_temp_db()
    scorer = ViralScorer()
    # Should not raise
    retrain_from_feedback(scorer, db=db)
```

### Step 3: Write the implementation

```python
# data/mognet/feedback_loop.py
#!/usr/bin/env python3
"""
MogNet — Feedback Loop
Records actual performance metrics and retrains the viral scorer from them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("mognet.feedback_loop")

_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _ROOT / "library" / "engine.db"


def _get_db():
    """Get default EngineDB instance (lazy import to avoid circular imports)."""
    import sys
    sys.path.insert(0, str(_ROOT / "data"))
    from engine_db import EngineDB
    db = EngineDB(db_path=_DB_PATH)
    db.init()
    return db


def record_actual_performance(
    edit_id: str,
    views: int,
    shares: int,
    saves: int,
    watch_pct: float,
    db=None,
):
    """
    Record real TikTok performance metrics for a previously generated edit.

    Parameters
    ----------
    edit_id   : matches the edit_id used when saving the prediction
    views     : total view count
    shares    : share count
    saves     : save/bookmark count
    watch_pct : average watch percentage (0.0–1.0)
    db        : optional EngineDB instance (creates default if None)
    """
    if db is None:
        db = _get_db()
    db.update_mognet_actuals(edit_id, views=views, shares=shares,
                              saves=saves, watch_pct=watch_pct)
    log.info("Recorded actuals for %s: views=%d watch_pct=%.2f", edit_id, views, watch_pct)


def retrain_from_feedback(scorer, db=None):
    """
    Retrain the viral scorer from accumulated actual performance data.

    Parameters
    ----------
    scorer : ViralScorer instance to retrain in-place
    db     : optional EngineDB instance
    """
    if db is None:
        db = _get_db()

    rows = db.get_mognet_training_rows()
    if len(rows) < 3:
        log.info("Not enough feedback rows (%d) to retrain — need at least 3", len(rows))
        return

    features_list = []
    metrics_list  = []

    for row in rows:
        try:
            feats = json.loads(row["features_json"])
            actual = float(row["actual_score"])
            # Reconstruct a synthetic metrics dict from actual_score
            # actual_score = views * watch_pct + shares * 10 + saves * 5
            # We store the raw score; pass as views with watch_pct=1.0
            features_list.append(feats)
            metrics_list.append({
                "views": int(actual),
                "watch_pct": 1.0,
                "shares": 0,
                "saves": 0,
            })
        except Exception as e:
            log.warning("Skipping malformed feedback row: %s", e)

    if len(features_list) < 3:
        log.info("Not enough valid feedback rows after parsing (%d)", len(features_list))
        return

    scorer.train(features_list, metrics_list)
    log.info("Retrained ViralScorer on %d feedback rows", len(features_list))

    # Save updated model
    model_path = _ROOT / "data" / "mognet" / "viral_scorer.pkl"
    scorer.save(str(model_path))
    log.info("Saved retrained model to %s", model_path)
```

### Step 4: Run tests

```bash
pytest tests/test_feedback_loop.py -v
```
Expected: all 4 tests PASS.

### Step 5: Commit

```bash
git add data/engine_db.py data/mognet/feedback_loop.py tests/test_feedback_loop.py
git commit -m "feat(mognet): feedback_loop + mognet_performance table in engine.db"
```

---

## Task 6: `data/mognet/reference_analyzer.py` — Bootstrap from gold library

**Files:**
- Create: `data/mognet/reference_analyzer.py`
- Test: `tests/test_reference_analyzer.py`

### Step 1: Write the failing test

```python
# tests/test_reference_analyzer.py
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from mognet.reference_analyzer import analyze_gold_library, _synthetic_engagement_for


def test_synthetic_engagement_is_positive():
    score = _synthetic_engagement_for("bp_masterx_7589676463071268118.mp4", 0.85)
    assert score > 0
    assert score["views"] > 0


def test_synthetic_engagement_scales_with_fidelity():
    low  = _synthetic_engagement_for("some_video.mp4", 0.3)
    high = _synthetic_engagement_for("some_video.mp4", 0.9)
    low_eng  = low["views"] * low["watch_pct"] + low["shares"] * 10 + low["saves"] * 5
    high_eng = high["views"] * high["watch_pct"] + high["shares"] * 10 + high["saves"] * 5
    assert high_eng > low_eng


def test_analyze_gold_library_runs():
    """analyze_gold_library should complete without error (may skip if no gold)."""
    gold_dir = Path(__file__).resolve().parent.parent / "input" / "gold"
    if not gold_dir.exists() or not list(gold_dir.glob("*.mp4")):
        pytest.skip("No gold videos available")
    model_path = Path(__file__).resolve().parent / "test_viral_scorer.pkl"
    scorer = analyze_gold_library(model_path=str(model_path))
    assert scorer is not None
    model_path.unlink(missing_ok=True)
```

### Step 2: Run to verify it fails

```bash
pytest tests/test_reference_analyzer.py::test_synthetic_engagement_is_positive -v 2>&1 | head -15
```

### Step 3: Write the implementation

```python
# data/mognet/reference_analyzer.py
#!/usr/bin/env python3
"""
MogNet — Reference Analyzer
Extracts features from all gold reference videos and bootstraps the viral scorer.
Run once (or periodically) to initialize/update data/mognet/viral_scorer.pkl.

Usage:
    python data/mognet/reference_analyzer.py
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger("mognet.reference_analyzer")

_ROOT = Path(__file__).resolve().parent.parent.parent
_GOLD_DIR = _ROOT / "input" / "gold"
_DB_PATH  = _ROOT / "library" / "engine.db"
_DEFAULT_MODEL_PATH = _ROOT / "data" / "mognet" / "viral_scorer.pkl"

# Known-good creator name signals for synthetic engagement boosting
_HIGH_SIGNAL_NAMES = {"bp", "gold", "masterx", "4ever", "morph", "haifluke"}
_LOW_SIGNAL_NAMES  = {"black", "roge", "pilled", "editz"}


def _synthetic_engagement_for(filename: str, fidelity_score: float) -> dict:
    """
    Generate a synthetic engagement metrics dict for a gold video.

    Strategy:
    - Base engagement scaled from fidelity_score (0–1)
    - Name heuristics add multiplier: high-signal names (bp, gold) → 2×
    - Result = plausible TikTok engagement for a top-10% edit
    """
    name_lower = filename.lower()

    # Base: median well-performing edit (50k views, 0.45 watch, 200 shares, 100 saves)
    base_views    = 50_000
    base_watch    = 0.35
    base_shares   = 150
    base_saves    = 80

    # Fidelity multiplier: 0.3→×0.5, 0.7→×1.0, 1.0→×2.5
    fid_mult = 0.5 + fidelity_score * 2.0

    # Name signal multiplier
    name_mult = 1.0
    if any(s in name_lower for s in _HIGH_SIGNAL_NAMES):
        name_mult = 1.8
    elif any(s in name_lower for s in _LOW_SIGNAL_NAMES):
        name_mult = 0.8

    mult = fid_mult * name_mult

    return {
        "views":     int(base_views * mult),
        "watch_pct": min(0.85, base_watch + fidelity_score * 0.25),
        "shares":    int(base_shares * mult),
        "saves":     int(base_saves * mult),
    }


def _get_fidelity_scores() -> dict[str, float]:
    """
    Read fidelity_score from engine.db generations table.
    Falls back to 0.65 (slightly above average) if no data.
    """
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        rows = conn.execute(
            "SELECT output_file, fidelity_score FROM generations "
            "WHERE fidelity_score IS NOT NULL"
        ).fetchall()
        conn.close()
        scores = {}
        for output_file, score in rows:
            if output_file:
                key = Path(output_file).stem
                scores[key] = float(score)
        return scores
    except Exception as e:
        log.warning("Could not read fidelity scores from DB: %s", e)
        return {}


def analyze_gold_library(
    gold_dir: str | None = None,
    model_path: str | None = None,
) -> "ViralScorer":
    """
    Extract features from all gold reference videos and train the viral scorer.

    Parameters
    ----------
    gold_dir    : path to gold video directory (default: input/gold/)
    model_path  : where to save the trained model (default: data/mognet/viral_scorer.pkl)

    Returns
    -------
    Trained ViralScorer instance
    """
    import sys
    sys.path.insert(0, str(_ROOT / "data"))

    from mognet.feature_extractor import extract_video_features
    from mognet.viral_scorer import ViralScorer

    gold_path  = Path(gold_dir) if gold_dir else _GOLD_DIR
    save_path  = model_path or str(_DEFAULT_MODEL_PATH)
    fid_scores = _get_fidelity_scores()

    gold_videos = sorted(gold_path.glob("*.mp4"))
    if not gold_videos:
        raise FileNotFoundError(f"No .mp4 files found in {gold_path}")

    log.info("Analyzing %d gold reference videos in %s", len(gold_videos), gold_path)
    print(f"\n{'═'*58}")
    print("  MogNet Reference Analyzer")
    print(f"  Gold videos: {len(gold_videos)}")
    print(f"  Model output: {save_path}")
    print(f"{'═'*58}")

    features_list: list[dict] = []
    metrics_list:  list[dict] = []
    failed = 0

    for vid_path in gold_videos:
        print(f"  ▶ Extracting: {vid_path.name[:50]}", flush=True)
        try:
            feats = extract_video_features(str(vid_path))

            # Get fidelity proxy from DB (or use name-based default)
            stem = vid_path.stem
            fidelity = fid_scores.get(stem, None)
            if fidelity is None:
                # Heuristic: gold-prefixed videos assumed higher quality
                name_lower = vid_path.name.lower()
                if any(s in name_lower for s in _HIGH_SIGNAL_NAMES):
                    fidelity = 0.80
                elif any(s in name_lower for s in _LOW_SIGNAL_NAMES):
                    fidelity = 0.60
                else:
                    fidelity = 0.70

            metrics = _synthetic_engagement_for(vid_path.name, fidelity)
            features_list.append(feats)
            metrics_list.append(metrics)
            print(
                f"    ✓ cps={feats['visual']['cuts_per_second']:.1f}  "
                f"bpm={feats['audio']['bpm']:.0f}  "
                f"aggression={feats['text']['hook_aggression_score']:.1f}  "
                f"fidelity={fidelity:.2f}  eng={metrics['views']:,}",
                flush=True,
            )
        except Exception as e:
            log.warning("Failed to extract features from %s: %s", vid_path.name, e)
            print(f"    ✗ Error: {e}", flush=True)
            failed += 1

    if len(features_list) < 2:
        raise RuntimeError(
            f"Need at least 2 successful extractions, got {len(features_list)} "
            f"({failed} failed)"
        )

    print(f"\n  Training ViralScorer on {len(features_list)} samples ...", flush=True)
    scorer = ViralScorer()
    scorer.train(features_list, metrics_list)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    scorer.save(save_path)
    print(f"  ✅ Model saved to {save_path}")

    return scorer


if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(level=logging.INFO)
    gold_arg  = _sys.argv[1] if len(_sys.argv) > 1 else None
    model_arg = _sys.argv[2] if len(_sys.argv) > 2 else None
    analyze_gold_library(gold_dir=gold_arg, model_path=model_arg)
```

### Step 4: Run tests

```bash
pytest tests/test_reference_analyzer.py -v
```
Expected: first 2 tests PASS (synthetic engagement logic). Third test PASS if gold videos exist, SKIP otherwise.

### Step 5: Run the bootstrap

```bash
python data/mognet/reference_analyzer.py
```
Expected: extracts features from all 10 gold videos, trains scorer, saves `data/mognet/viral_scorer.pkl`.

### Step 6: Commit

```bash
git add data/mognet/reference_analyzer.py tests/test_reference_analyzer.py
git commit -m "feat(mognet): reference_analyzer — bootstrap viral scorer from gold library"
```

---

## Task 7: Wire MogNet into `generate_batch.py`

**Files:**
- Modify: `data/generate_batch.py`

### Overview of changes

1. Add module-level MogNet import block (matching existing `_NARRATIVE_ENGINE` pattern)
2. Load `ViralScorer` once at startup (lazy: from pkl if exists, else train from gold)
3. Add `_validate_and_maybe_retry()` helper
4. Call it after each `generate_from_template()` in `main()`
5. Add `validation` key to result dict
6. Save prediction to engine.db via `save_mognet_prediction()`

### Step 1: Add MogNet import block

After the existing `_NARRATIVE_ENGINE` import block in `generate_batch.py`, add:

```python
# MogNet validator — imported at module level, None if not available
try:
    import sys as _sys_mn; _sys_mn.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    from mognet.viral_scorer import ViralScorer as _MN_ViralScorer
    from mognet.validator import validate_edit as _MN_validate
    from mognet.feedback_loop import record_actual_performance as _MN_record
    _MOGNET = (_MN_ViralScorer, _MN_validate, _MN_record)
    del _sys_mn
except ImportError:
    _MOGNET = None


_MODEL_PATH = ROOT / "data" / "mognet" / "viral_scorer.pkl"


def _load_mognet_scorer():
    """Load or bootstrap the MogNet viral scorer. Returns None if unavailable."""
    if _MOGNET is None:
        return None
    ViralScorer, _, _ = _MOGNET
    scorer = ViralScorer()
    if _MODEL_PATH.exists():
        try:
            scorer.load(str(_MODEL_PATH))
            return scorer
        except Exception as e:
            print(f"  [mognet] Could not load model ({e}), bootstrapping from gold ...", flush=True)
    # Bootstrap from gold library
    try:
        import sys as _s; _s.path.insert(0, str(ROOT / "data"))
        from mognet.reference_analyzer import analyze_gold_library
        return analyze_gold_library()
    except Exception as e:
        print(f"  [mognet] Bootstrap failed: {e}", flush=True)
        return None
```

### Step 2: Add `_validate_and_maybe_retry()` helper

Add this function to `generate_batch.py` (after `_load_mognet_scorer`):

```python
def _validate_and_maybe_retry(
    template: dict,
    initial_result: dict,
    scorer,
    max_retries: int = 3,
) -> dict:
    """
    Validate generated edit via MogNet. If REJECT, retry up to max_retries
    times with parameter jitter. Returns final result with 'validation' key.
    """
    if scorer is None or "output" not in initial_result:
        return initial_result

    _, validate_fn, _ = _MOGNET
    result = initial_result

    for attempt in range(max_retries + 1):
        edit_path = result.get("output", "")
        if not edit_path or not Path(edit_path).exists():
            break

        validation = validate_fn(edit_path, scorer)
        result["validation"] = validation

        decision = validation["decision"]
        score    = validation["viral_score"]

        print(
            f"  {'✅' if decision == 'APPROVE' else '⚠️ '} "
            f"MogNet: {decision}  score={score:.1f}/100  "
            f"(attempt {attempt + 1}/{max_retries + 1})",
            flush=True,
        )
        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"       {w}", flush=True)
        if validation["critiques"]:
            for c in validation["critiques"]:
                print(f"       {c}", flush=True)

        if decision == "APPROVE" or attempt == max_retries:
            break

        # Jitter params for retry: vary cut density
        print(f"  ↩️  Regenerating with adjusted params (attempt {attempt + 2}) ...", flush=True)
        jitter_template = dict(template)
        jitter_template["total_clips"] = template.get("total_clips", 8) + (attempt + 1) * 2
        retry_result = generate_from_template(jitter_template, dry_run=False)
        if "output" in retry_result:
            result = retry_result

    return result
```

### Step 3: Wire into `main()`

In the `main()` function, load scorer once, then wrap each result:

After `templates = load_top_templates(args.top_n)`, add:
```python
    mognet_scorer = _load_mognet_scorer() if not args.dry_run else None
    if mognet_scorer:
        print("  ✅ MogNet ViralScorer loaded")
    else:
        print("  ~ MogNet not available — skipping validation")
```

Then change the generation loop from:
```python
    for t in templates:
        r = generate_from_template(t, dry_run=args.dry_run)
        results.append(r)
```

To:
```python
    for t in templates:
        r = generate_from_template(t, dry_run=args.dry_run)
        if mognet_scorer and not args.dry_run:
            r = _validate_and_maybe_retry(t, r, mognet_scorer)
            # Save prediction to DB
            try:
                import sys as _s; _s.path.insert(0, str(ROOT / "data"))
                from engine_db import EngineDB
                import json as _json
                db = EngineDB(); db.init()
                edit_id = r.get("name", "unknown")
                pred    = r.get("validation", {}).get("viral_score", 0.0)
                feats   = r.get("validation", {})
                db.save_mognet_prediction(edit_id, r.get("output",""), pred, "")
            except Exception as _e:
                print(f"  [mognet] DB save failed: {_e}", flush=True)
        results.append(r)
```

### Step 4: Update results printing to include validation

In the results printout section, after the `avg_cut_sec` line, add:

```python
            val = r.get("validation", {})
            if val:
                print(f"       MogNet: {val.get('decision','?')}  score={val.get('viral_score',0):.1f}/100")
                for s in val.get("strengths", [])[:2]:
                    print(f"         + {s}")
```

### Step 5: Test dry-run still works

```bash
python data/generate_batch.py --dry-run
```
Expected: runs without error, MogNet prints "not available" or is skipped (dry-run bypasses validation).

### Step 6: Commit

```bash
git add data/generate_batch.py
git commit -m "feat(generate_batch): wire MogNet validation loop — retry up to 3× on REJECT"
```

---

## Task 8: Full end-to-end test run

### Step 1: Run the full test suite

```bash
pytest tests/ -v --tb=short 2>&1 | tail -40
```
Expected: all tests PASS (or SKIP for tests requiring gold videos if they're not present).

### Step 2: Run reference analyzer to bootstrap model

```bash
python data/mognet/reference_analyzer.py
```
Expected: extracts features from all gold videos, saves `data/mognet/viral_scorer.pkl`.

### Step 3: Run full generation with validation

```bash
python data/generate_batch.py --top-n 2
```
Expected output includes:
- `MogNet ViralScorer loaded`
- Per-edit: `✅ MogNet: APPROVE  score=XX.X/100` or `⚠️ MogNet: REJECT` with critiques
- If any REJECT: retry attempts logged, up to 3×
- Final manifest includes `validation` key per result

### Step 4: Show manifest results

```bash
python -c "
import json
from pathlib import Path
m = json.loads((Path('out/generate_batch_manifest.json')).read_text())
for r in m:
    if 'validation' in r:
        v = r['validation']
        print(f\"{r['name']}: {v['decision']} score={v['viral_score']:.1f}\")
        for s in v.get('strengths',[])[:2]: print(f'  + {s}')
        for c in v.get('critiques',[]): print(f'  ! {c}')
"
```

### Step 5: Final commit

```bash
git add .
git commit -m "feat(mognet): complete MogNet self-validating edit engine

- feature_extractor: visual/audio/text features via OpenCV/librosa/EasyOCR
- viral_scorer: XGBoost+LightGBM ensemble, 0-100 viral score prediction
- validator: 5-rule critique system, APPROVE/REJECT with retry loop
- feedback_loop: mognet_performance table, post-upload retraining
- reference_analyzer: bootstrap from 10 gold videos
- generate_batch: validation loop with 3× retry on REJECT"
```
