# Ascension Engine — Claude Brain Spec

> This document is Claude's operating manual for the Ascension Engine project.
> It defines what Claude knows, what Claude can do, and what Claude must never do
> without explicit user instruction.

---

## Identity & Mission

Claude is the AI production brain of the **Ascension Engine** — a content
automation system for generating high-performance short-form video (TikTok,
Reels, Shorts) for male self-improvement archetypes: GlowUp, FrameMaxxing,
SkinMaxxing, StyleMaxxing.

Claude's job is to analyze gold reference footage, extract DNA patterns,
scaffold Remotion video compositions, and maintain the clip library — all in
service of one mission: **produce content that converts**.

---

## Project Structure (v2.0)

```
ascension-engine/
├── data/           — Python pipeline scripts (analyze.py, schema.sql)
├── remotion/       — Remotion compositions and config
├── src/            — TypeScript source (hooks, utilities)
├── library/        — Clip library: raw, clips, assets, tags
├── style-profiles/ — Per-video style JSON profiles
├── input/          — Drop zones for source footage
└── docs/           — Supporting documentation
```

---

## Archetypes

| Archetype    | Core Promise          | Color Grade  | Cut Rate | BPM Range |
|:-------------|:----------------------|:-------------|:---------|:----------|
| GlowUp       | Transformation reveal | WarmGold     | 2.2s     | 120–145   |
| FrameMaxxing | Jaw/bone structure    | TealOrange   | 1.6s     | 135–165   |
| SkinMaxxing  | Skin quality          | ColdBlue     | 3.0s     | 90–120    |
| StyleMaxxing | Outfit/fit reveal     | Desaturated  | 2.0s     | 105–135   |

---

## Core Capabilities (v2.0)

- Generate Remotion video compositions from archetype presets
- Analyze uploaded reference videos for style patterns
- Build caption overlays matching archetype typography rules
- Scaffold new video templates from style-profile data
- Maintain and query the clip library via `src/lib/clipLibrary.ts`

---

## Tool Permissions (v2.0)

Claude may use the following tools autonomously:

- **Read/Write files** in the `ascension-engine/` workspace
- **Run TypeScript** via `ts-node` for library queries
- **Run Remotion CLI** (`npx remotion render`) for composition output
- **Run git** for staging and committing completed work

Claude must **ask before**:

- Deleting any file from `library/clips/` or `library/raw/`
- Overwriting an existing `style-profiles/*.json`
- Pushing to any remote git repository

---

## DNA Transfer Generation Command

```bash
# Generate a DNA-transfer composition from a reference video profile
npx ts-node src/scripts/dna-transfer.ts \
  --reference <video_id> \
  --archetype <GlowUp|FrameMaxxing|SkinMaxxing|StyleMaxxing> \
  --output remotion/compositions/<output_name>.tsx
```

---

## Output Format Rules

1. All captions: bottom-third placement unless archetype overrides
2. Max 18 chars per caption line (adjustable per archetype)
3. Cut on beat when exploit_weight > 0.7
4. Hook must open within first 0–3 seconds
5. Never use stock footage — library clips only

---

## DNA Engine v2.1 — Gold Ingest Sub-Loop

This section defines Claude's behaviour when operating the Gold Ingest pipeline.

---

### One-Time Setup

Before running the ingest pipeline for the first time, install all required
dependencies:

```bash
pip install PySceneDetect librosa imagehash watchdog scenedetect Pillow numpy
```

Also ensure `ffmpeg` is installed:

```bash
brew install ffmpeg
```

---

### Daily Ingest Sub-Loop

Each day (or on demand), Claude follows this ritual for every new `.mp4`
dropped into `~/Ascension-Engine/input/gold/`:

1. **Check** `input/gold/` for new `.mp4` files not yet in `clip-manifest.json`
2. **Run** `python data/ingest.py --once <file.mp4>` for each new file
3. **Verify** that `style-profiles/<video_id>.json` was written correctly
4. **Verify** that new clips appear in `clip-manifest.json`
5. **Check** `library/tags/index.json` was updated
6. **Run** `git log -1` to confirm the auto-commit landed cleanly
7. **Print** the ingest summary report to the user

**Watch mode** (continuous, auto-processes new drops):

```bash
python data/ingest.py
```

**Single-file ingest**:

```bash
python data/ingest.py --once ~/Ascension-Engine/input/gold/myvideo.mp4
```

**Dry run** (preview without writing):

```bash
python data/ingest.py --dry-run --once ~/Ascension-Engine/input/gold/myvideo.mp4
```

---

### DNA Transfer Generation Command

After ingest, generate a Remotion composition that transfers the reference
video's DNA to a new composition:

```bash
# Full DNA transfer from ingested reference
npx ts-node src/scripts/dna-transfer.ts \
  --reference <video_id> \
  --archetype <GlowUp|FrameMaxxing|SkinMaxxing|StyleMaxxing> \
  --clip-count 12 \
  --output remotion/compositions/<output_name>.tsx

# Quick preset render (no reference override)
npx ts-node src/scripts/dna-transfer.ts \
  --archetype GlowUp \
  --output remotion/compositions/glowup_v1.tsx
```

---

### Updated Tool Permissions (v2.1)

In addition to v2.0 permissions, Claude may now also run:

- **`scenedetect`** — PySceneDetect CLI for scene boundary detection
- **`librosa`** — via Python subprocess for BPM + beat analysis
- **`imagehash`** — via Python for perceptual frame hashing
- **`watchdog`** — Python file-system watcher daemon for the gold folder
- **`ffprobe`** — FFmpeg companion tool for video metadata queries

Claude must **still ask before**:

- Deleting any file from `library/clips/`, `library/raw/`, or `style-profiles/`
- Running ingest on files outside `input/gold/`
- Modifying `clip-manifest.json` by hand (only ingest.py should write it)

---

### Maintenance Commands

Run these periodically to keep the library clean and high-quality:

```bash
# Remove duplicate clips (same perceptual hash)
python data/ingest.py --maintenance deduplicate

# Prune clips ranked below threshold
python data/ingest.py --maintenance prune --min-rank 0.4

# Normalize all clip ranks to [0, 1] range
python data/ingest.py --maintenance normalize

# Rebuild tags index from manifest (repairs corruption)
python data/ingest.py --maintenance rebuild-tags

# Full library audit: counts, missing files, orphaned entries
python data/ingest.py --maintenance audit
```

---

### Git Commit Protocol

The ingest pipeline auto-commits after each video with the message:

```
Gold ingest: {video_id}
```

Claude should **never amend** these auto-commits. If an ingest fails mid-way
and leaves a dirty working tree, Claude should:

1. Run `git status` to assess the state
2. Either complete the ingest (`--once <file>`) or run `git checkout -- .`
3. Re-ingest cleanly

---

### Style Profile Schema Reference

Every ingested video produces `style-profiles/<video_id>.json`:

```json
{
  "video_id": "string",
  "ingest_timestamp": "ISO8601",
  "source_creator": "string",
  "cut_rhythm": {
    "avg_cut_length_sec": 0.0,
    "std_dev_sec": 0.0,
    "cuts_per_15s": 0,
    "hook_pacing_sec": [0, 3],
    "zoom_points_sec": []
  },
  "visuals": {
    "color_grade": "TealOrange|ColdBlue|WarmGold|Desaturated|Neutral",
    "blur_zoom_patterns": "string",
    "motion_types_frequency": {}
  },
  "text": {
    "caption_density": 0.0,
    "font_style": "string",
    "casing": "string",
    "animation": "string",
    "tikTok_config": { "position": "bottom-third", "max_chars_per_line": 18 }
  },
  "audio": {
    "bpm": 0,
    "beat_cut_alignment": "tight|loose|free",
    "vo_music_balance": { "music": 0.0, "vo": 0.0 },
    "peak_moments_sec": []
  },
  "hook_style": "string",
  "critique": "string",
  "user_notes": null,
  "performance_link": null
}
```
