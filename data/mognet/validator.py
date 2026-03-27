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
    if gap_ms > 0 and gap_ms < 40.0:  # 0.0 means no gap data — skip rather than false-positive
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
        "features":    features,  # included for DB persistence in generate_batch
    }
