#!/usr/bin/env python3
"""
MogNet — Viral Score Model
XGBoost regressor trained on reference video features + engagement metrics.
Ensemble with LightGBM for robustness with small training sets.
"""
from __future__ import annotations

import logging
import math

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
            self._lgbm.fit(X, y, feature_name=_FEATURE_NAMES)
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
            "confidence": float (0–1),  # static proxy: 0.85=ensemble, 0.70=XGBoost-only
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
            confidence = 0.85  # ensemble path — static proxy, not a calibrated interval
        else:
            raw_score = xgb_score
            confidence = 0.70  # XGBoost-only path

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
