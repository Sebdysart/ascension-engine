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
    - Name heuristics add multiplier: high-signal names (bp, gold) → ~2×
    - Result = plausible TikTok engagement for a top-10% edit
    """
    name_lower = filename.lower()

    # Base: median well-performing edit (50k views, 0.35 watch, 150 shares, 80 saves)
    base_views    = 50_000
    base_watch    = 0.35
    base_shares   = 150
    base_saves    = 80

    # Fidelity multiplier: 0.3→×1.1, 0.7→×1.9, 1.0→×2.5
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
    Falls back to empty dict if no data or DB unavailable.
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
    import sys as _sys
    _sys.path.insert(0, str(_ROOT / "data"))

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
