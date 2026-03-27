#!/usr/bin/env python3
"""
Ascension Engine — Hook Generator

Generates the scroll-stopping hook specification for frames 1-15 (first ~0.5s).

Formula:
  1. Select "flaw clip" — highest brightness, softest lighting from victim_contrast/
  2. Zoom-out: 150%→100% over 0.5s
  3. On first bass hit:
     - Shake spike: 30→100→0 over 0.2s
     - Zoom pulse: 100%→115%→100% over 0.3s
     - Grade shift: warm→cold

Returns HookSpec as JSON consumed by BrutalBeatMontage + sequencer.

Usage:
    from data.hook_generator import generate_hook_spec
    spec = generate_hook_spec(clips=clips, beats=beats)
    print(spec.to_json())
"""

from __future__ import annotations
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _ROOT / "library" / "engine.db"
_CLIP_MANIFEST = _ROOT / "clip-manifest.json"

log = logging.getLogger("hook_generator")

SOFT_LIGHTING_TAGS = frozenset({
    "flat_lighting_cope", "soft_light", "natural_indoor",
    "overhead_light", "flat_lighting",
})


@dataclass
class HookSpec:
    """Complete spec for the scroll-stopping hook sequence."""
    flaw_clip_id: str
    flaw_clip_path: str
    flaw_clip_brightness: float

    zoom_start_pct: float = 150.0
    zoom_end_pct: float = 100.0
    zoom_duration_sec: float = 0.5

    bass_hit_time_sec: float = 0.0
    shake_start: float = 30.0
    shake_peak: float = 100.0
    shake_end: float = 0.0
    shake_duration_sec: float = 0.2

    zoom_pulse_start_pct: float = 100.0
    zoom_pulse_peak_pct: float = 115.0
    zoom_pulse_end_pct: float = 100.0
    zoom_pulse_duration_sec: float = 0.3

    grade_start: str = "warm"
    grade_end: str = "cold"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _estimate_brightness(clip: dict) -> float:
    """Invert mog_score: victim clips (low mog) = bright/soft = high brightness."""
    return round(1.0 - float(clip.get("mog_score", 0.5)), 3)


def select_flaw_clip(clips: list[dict]) -> dict | None:
    """
    Pick the best "flaw clip" for the hook.
    - Must be victim_contrast track
    - Highest estimated brightness (lowest mog_score)
    - Bonus for soft-lighting tags
    Falls back to lowest-rank clip if no victim clips exist.
    """
    victim_clips = [
        c for c in clips
        if c.get("mog_track") == "victim_contrast" or c.get("track") == "victim_contrast"
    ]

    if not victim_clips:
        victim_clips = sorted(clips, key=lambda c: float(c.get("rank", 0.5)))[:3]

    if not victim_clips:
        return None

    def _score(c: dict) -> float:
        brightness = _estimate_brightness(c)
        soft_bonus = 0.2 if set(c.get("tags", [])) & SOFT_LIGHTING_TAGS else 0.0
        return brightness + soft_bonus

    return max(victim_clips, key=_score)


def _load_clips_from_manifest() -> list[dict]:
    if not _CLIP_MANIFEST.exists():
        return []
    try:
        return json.loads(_CLIP_MANIFEST.read_text()).get("clips", [])
    except Exception:
        return []


def _load_victim_clips_from_db() -> list[dict]:
    if not _DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT clip_id, file_path, rank, tags FROM clips WHERE track = 'victim_contrast'"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def generate_hook_spec(
    beats: list[float] | None = None,
    clips: list[dict] | None = None,
    dry_run: bool = False,
) -> HookSpec | None:
    """
    Generate a HookSpec for the scroll-stopping hook sequence.

    Parameters
    ----------
    beats:   Detected beat timestamps — first beat = first bass hit
    clips:   Clip list (loaded from clip-manifest.json if None)
    dry_run: Log-only mode, skip DB reads

    Returns
    -------
    HookSpec or None if no suitable flaw clip found
    """
    # If caller passed an explicit list (even empty), respect it — skip auto-loading.
    explicit_clips = clips is not None
    _clips = clips if explicit_clips else _load_clips_from_manifest()

    if not _clips and not explicit_clips:
        _clips = _load_victim_clips_from_db()

    if not _clips:
        log.warning("hook_generator: no clips available")
        return None

    flaw_clip = select_flaw_clip(_clips)
    if flaw_clip is None:
        log.warning("hook_generator: no victim/flaw clip found")
        return None

    clip_id   = flaw_clip.get("clip_id") or flaw_clip.get("id", "unknown")
    clip_path = flaw_clip.get("file_path") or flaw_clip.get("file", "")
    brightness = _estimate_brightness(flaw_clip)
    bass_hit   = beats[0] if beats else 0.0

    spec = HookSpec(
        flaw_clip_id=clip_id,
        flaw_clip_path=clip_path,
        flaw_clip_brightness=brightness,
        bass_hit_time_sec=bass_hit,
    )

    log.info("HookSpec: flaw_clip=%s brightness=%.2f bass_hit=%.2fs", clip_id, brightness, bass_hit)
    return spec


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    spec = generate_hook_spec()
    if spec:
        print(spec.to_json())
    else:
        print("[error] No flaw clip found — ingest victim_contrast clips first")
        sys.exit(1)
