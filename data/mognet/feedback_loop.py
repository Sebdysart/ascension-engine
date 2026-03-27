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
