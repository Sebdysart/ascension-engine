# MogNet Self-Validating Edit Engine — Design

_Date: 2026-03-27_

## Overview

A self-improving intelligence layer that learns viral edit patterns from gold reference TikToks and automatically validates generated edits before they leave the pipeline. Edits that score below threshold are rejected and regenerated with adjusted params (up to 3 attempts).

## Architecture

```
input/gold/*.mp4
      │
      ▼
reference_analyzer.py  ──► feature_extractor.py  ──► viral_scorer.pkl
                                                              │
generate_batch.py  ──► [each edit]  ──► validator.py ◄───────┘
                                            │
                              APPROVE / REJECT (→ retry ×3)
                                            │
                                      engine.db mognet_performance
                                            │
                                    feedback_loop.py (retrain)
```

## Modules

### `data/mognet/feature_extractor.py`
- **Scene cuts**: OpenCV VideoCapture + frame-diff histogram delta → cuts_per_second per act window
- **Brightness**: YUV mean luminance per clip
- **Color temp**: R/B channel ratio heuristic → warm (>1.2) / cold (<0.9)
- **Shake**: frame-to-frame displacement via cv2.calcOpticalFlowFarneback mean magnitude
- **Zoom pulse**: scale spike detection via center-crop area variance
- **Audio**: librosa beat_track + onset_strength for drops, silence gaps
- **OCR**: EasyOCR on frames 0–15 for hook text extraction
- **Aggression score**: `claude` CLI subprocess with 1-sentence prompt → float 0–10

### `data/mognet/viral_scorer.py`
- XGBoost `XGBRegressor` (primary) with LightGBM `LGBMRegressor` as ensemble fallback
- Target: `engagement_rate = views * watch_pct + shares * 10 + saves * 5`
- Bootstrap: 10 gold videos → synthetic engagement from fidelity_score proxy
- Output: `{score: 0–100, confidence: float, breakdown: dict}`
- Persistence: `data/mognet/viral_scorer.pkl` (joblib)

### `data/mognet/validator.py`
- Extracts features from generated edit path
- Scores via ViralScorer
- Rule-based critiques (5 rules):
  1. silence_gap < 40ms → WARNING
  2. angle_inversion_count == 0 → CRITICAL
  3. hook_aggression < 6.0 → WARNING
  4. brightness_contrast < 1.3 → WARNING
  5. cuts_per_second_act2 < 5 → CRITICAL
- Auto-REJECT if viral_score < 75

### `data/mognet/feedback_loop.py`
- `record_actual_performance()` → INSERT into `mognet_performance`
- `retrain_from_feedback()` → SELECT actuals, retrain ViralScorer, save model

### `data/mognet/reference_analyzer.py`
- Iterates `input/gold/*.mp4`
- Extracts features via feature_extractor
- Assigns synthetic engagement from filename heuristics + fidelity proxies
- Trains and saves initial `viral_scorer.pkl`

## DB Schema Addition

```sql
CREATE TABLE IF NOT EXISTS mognet_performance (
    edit_id         TEXT PRIMARY KEY,
    edit_path       TEXT,
    predicted_score REAL,
    actual_score    REAL,
    views           INTEGER,
    shares          INTEGER,
    saves           INTEGER,
    watch_pct       REAL,
    features_json   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
```

Added to `EngineDB.init()` — CREATE IF NOT EXISTS, no migration.

## generate_batch.py Wiring

After each `generate_from_template()`:
1. Call `validate_edit(out_final, scorer)`
2. If REJECT → jitter params (±10% cut rate, swap grade) → regenerate → repeat up to 3×
3. Add `validation` key to result dict
4. Call `record_actual_performance()` with predicted score (actuals filled later)

## Dependencies Added

```
xgboost
lightgbm
joblib
```

## Success Criteria

- feature_extractor returns all required keys without crashing on any gold video
- viral_scorer trains on ≥5 bootstrap samples without error
- validator produces APPROVE/REJECT for any generated .mp4
- generate_batch validation loop runs end-to-end (dry-run safe)
- viral scores and critique reports appear in manifest JSON
