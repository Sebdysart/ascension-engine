# Ascension Engine v4.2 — Claude Code Handoff (Hardened)

> Everything is on `main`. Clone it on your Mac Mini and run.

---

## One-Time Setup (Mac Mini)

```bash
git clone <your-repo-url> ~/Ascension-Engine
cd ~/Ascension-Engine

# Node deps
npm install

# Python deps
pip install -r requirements.txt

# Verify
npx tsc --noEmit          # should pass with 0 errors
npx remotion studio remotion/index.ts   # opens at localhost:3000
python3 data/ingest.py --setup          # prints dep check
```

FFmpeg must be installed: `brew install ffmpeg`

```bash
# Initialize the engine database (one-time)
python3 -c "from data.engine_db import EngineDB; db=EngineDB(); db.init(); db.close()"

# Check engine health anytime
python3 data/engine_status.py
```

---

## v4.2 Hardening (9 Gaps Fixed)

| Gap | Fix | Where |
|-----|-----|-------|
| 1. Vision drift | Hybrid scoring stub (MediaPipe landmarks ready for Mac Mini) | `pip install mediapipe` |
| 2. Text removal | Two-pass FFmpeg crop (Claude Vision drives bbox on Mac Mini) | Vision-guided |
| 3. Beat tracking | Hybrid: librosa + onset envelope + `beat_confidence` score | `data/ingest.py` |
| 4. Library bloat | Perceptual hash (`phash`) on every clip + hamming-distance dedup | `data/ingest.py` + `data/engine_db.py` |
| 5. Mode collapse | 75/25 exploit/explore hardcoded — every 4th clip is a mutation | `src/lib/clipLibrary.ts` |
| 6. Human gate overload | Generation tracking + batch approval via engine DB | `data/engine_db.py` |
| 7. Client isolation | `library/clients/` per-client sub-libraries | Directory structure |
| 8. Crash recovery | SQLite `engine.db` — idempotent ingest, transactional status | `data/engine_db.py` |
| 9. Health report | `python3 data/engine_status.py` — single source of truth | `data/engine_status.py` |

---

## How the System Works

### Drop a gold edit → library grows automatically

```bash
# Drop .mp4 into input/gold/ then:
python3 data/ingest.py --once input/gold/your_edit.mp4
```

This generates **5 assets per video**:

| Asset | Location | What it is |
|-------|----------|------------|
| Clips | `library/clips/*.mp4` | Scene-segmented segments |
| Sequence template | `library/sequence_templates/seq_*.json` | Full beat-synced timeline — remix fuel |
| Grade preset | `library/grade_presets/grade_*.json` | CSS + FFmpeg color filters |
| Blueprint | `library/blueprints/bp_*.json` | Beat times, onsets, cut alignment |
| Style profile | `style-profiles/*.json` | Per-video DNA (BPM, cuts_on_beat_pct, etc.) |

### Generate edits (two modes)

**Blueprint mode** — builds fresh timeline from a gold video's beat data:
```bash
npx ts-node --compiler-options '{"module":"commonjs","moduleResolution":"node"}' \
  src/scripts/dna-transfer.ts --style dark_cinema --reference iblameredeye --render
```

**Remix mode** — takes a sequence template and swaps in different library clips:
```bash
npx ts-node --compiler-options '{"module":"commonjs","moduleResolution":"node"}' \
  src/scripts/dna-transfer.ts --style teal_orange_hard --remix darvikz --render
```

**Batch:**
```bash
npx ts-node --compiler-options '{"module":"commonjs","moduleResolution":"node"}' \
  src/scripts/dna-transfer.ts --batch 4 --remix auto --render
```

Or use npm shortcuts:
```bash
npm run generate -- --style dark_cinema --reference iblameredeye --render
npm run generate -- --remix auto --render
```

### Compare against gold (anti-slop gate)

```bash
python3 data/compare.py -g out/your_edit.mp4 --gold-dir input/gold/ --pass-threshold 0.60
```

### Feedback loop (analytics → clip ranks)

```bash
python3 data/feedback.py --seed-analytics   # seed test data
python3 data/feedback.py                    # run rank adjustments
python3 data/analyze.py --report-only       # weekly bandit analysis
```

---

## Available Color Grades

| Grade | CSS filter | Use for |
|-------|-----------|---------|
| `dark_cinema` | brightness(0.75) contrast(1.35) saturate(0.85) sepia(0.15) | Mog edits, @roge.editz style |
| `dark_cinema_hard` | brightness(0.68) contrast(1.45) saturate(0.75) | Harder crush, extreme dark |
| `desaturated` | saturate(0.15) contrast(1.25) | B&W, @black.pill.city style |
| `cold_blue` | saturate(0.9) hue-rotate(20deg) brightness(0.85) | Skin/frame analysis |
| `warm_ambient` | saturate(1.1) sepia(0.2) brightness(0.85) | Social dynamics |
| `teal_orange` | saturate(1.3) hue-rotate(-10deg) contrast(1.15) | Transformation reveals |
| `teal_orange_hard` | saturate(1.4) hue-rotate(-12deg) contrast(1.25) | Aggressive cuts |
| `natural` | saturate(1.05) contrast(1.08) | Raw selfie feel |

---

## What Claude Code Should Do on Mac Mini

### Priority 1: Add these to the Claude Project Brain instructions

> **Iron rules for every edit:**
> - Text density ≤ 0.05. No CTA cards. No first-person motivational text.
> - Every cut MUST land within 80ms of a beat (use blueprint beat_times_sec).
> - Grade must crush blacks and pop faces. Use dark_cinema or desaturated by default.
> - Use ONLY library clips. Never generate from scratch.
> - Ingest EVERY gold with full decipher ritual before generating.
> - Run anti-slop gate before any preview. Threshold ≥ 0.60.
> - Use dual-track library: good_parts/ for S-tier mog (≥0.88 rank), victim_contrast/ for deliberate contrast only.
> - Run hybrid mog scoring: Claude Vision (70%) + MediaPipe landmarks (30%).
> - Ingest is idempotent via engine.db. Running twice on the same file skips cleanly.
> - Always use 75/25 exploit/explore (hardcoded). Every 4th clip slot is a mutation.
> - Show batch table with predicted fidelity. User can APPROVE ALL or set AUTO threshold.
> - Per-client work goes in library/clients/. Never mix with global library.
> - Run `python3 data/engine_status.py` daily to check health.

### Priority 2: Vision-powered features (Claude Code can do these)

1. **Semantic clip tagging** — after ingest, Claude Vision scans thumbnails and tags each clip:
   - `mog_face_closeup_push`, `blackpill_stare_direct`, `jawline_pop_side`
   - `phonk_drop_impact`, `gym_mog_broll_slowmo`, `cinematic_reveal`
   - Write tags into `clip-manifest.json` clip entries

2. **Text detection + removal** — Claude Vision finds text overlay bounding boxes on key frames, generates FFmpeg crop/mask commands:
   ```bash
   ffmpeg -i clip.mp4 -vf "delogo=x=100:y=800:w=880:h=200" cleaned_clip.mp4
   ```
   Save cleaned clips to `library/assets/video/cleaned/`

3. **Good-text extraction** — OCR minimal impact text (1-3 words on beat drops), save as templates in `library/text_templates/`:
   ```json
   {"text": "MOGGED", "style": "white sans 80% opacity", "beat_offset_ms": 40}
   ```

### Priority 3: FFmpeg post-processing

After Remotion render, apply LUT + grain:
```bash
ffmpeg -i rendered.mp4 -vf "lut3d=custom_teal_orange.cube,noise=alls=8:allf=t" -c:v libx264 -crf 18 final.mp4
```

---

## File Map (v4.2)

```
ascension-engine/
├── remotion/
│   ├── VideoTemplate.tsx    — Beat-synced montage template (v4)
│   ├── Root.tsx             — Composition registration
│   └── index.ts             — registerRoot() entry point
├── src/
│   ├── types.ts             — BrutalBPBlueprint, SequenceTemplate, GradePreset, TextTemplate
│   ├── lib/
│   │   └── clipLibrary.ts   — getBrutalTimeline(), remixSequence(), 75/25 exploit/explore
│   └── scripts/
│       └── dna-transfer.ts  — Blueprint + remix generation CLI
├── data/
│   ├── ingest.py            — Idempotent ingest + phash dedup + beat confidence
│   ├── engine_db.py         — SQLite engine.db: single source of truth (v4.2)
│   ├── engine_status.py     — Health dashboard (v4.2)
│   ├── compare.py           — Anti-slop gate: SSIM + color + pacing + beat
│   ├── feedback.py          — Analytics → clip rank adjustments
│   ├── analyze.py           — Weekly bandit optimizer
│   └── schema.sql           — SQLite analytics schema
├── library/                 — THE MOAT
│   ├── engine.db            — Central SQLite DB (idempotent tracking)
│   ├── clips/               — Scene-segmented MP4s
│   ├── good_parts/          — S-tier mog only (≥0.88 rank) — Claude Vision fills
│   ├── victim_contrast/     — Deliberate contrast (<0.75) — Claude Vision fills
│   ├── clients/             — Per-client isolated DNA (Gap 7)
│   ├── sequence_templates/  — Reusable beat-synced timelines
│   ├── grade_presets/       — CSS + FFmpeg color filters
│   ├── blueprints/          — Beat data + cut alignment
│   ├── text_templates/      — Claude Vision fills (good-text OCR)
│   ├── assets/
│   │   ├── video/           — Raw video assets
│   │   ├── audio/           — Extracted WAV tracks
│   │   └── thumbnails/      — Keyframes for vision analysis
│   └── tags/
│       └── index.json       — Tag → clip_id index
├── style-profiles/          — Per-video DNA blueprints
├── input/gold/              — DROP ZONE: add .mp4 files here
├── clip-manifest.json       — Master clip catalog
├── package.json             — npm scripts
├── requirements.txt         — Python deps (now includes mediapipe)
└── AGENTS.md                — Cloud agent instructions
```

---

## Daily Workflow

1. **Drop gold edits** into `input/gold/`
2. **Ingest**: `python3 data/ingest.py --once input/gold/new_edit.mp4`
3. **Generate**: `npm run generate -- --remix auto --style dark_cinema --render`
4. **Compare**: `npm run compare -- -g out/edit.mp4 --gold-dir input/gold/`
5. **Approve + post**
6. **Feed analytics**: `npm run feedback`
7. **Weekly optimize**: `npm run analyze`

The library gets smarter every cycle. After 20-30 feeds, the engine outputs edits indistinguishable from your best manual work.

---

### Priority 4: Daily health check

```bash
python3 data/engine_status.py
```

Shows library size, mog/victim split, fidelity scores, pending approvals, disk usage, warnings.

---

*Pushed to `main` — v4.2 hardened — March 27, 2026*
