# Ascension Engine v2.1

> AI-powered short-form video production system for male self-improvement content.
> Built on Remotion + TypeScript + Python, optimized for TikTok/Reels/Shorts.

---

## What Is This

The Ascension Engine is a full content automation stack that:

- **Analyzes** gold reference footage for style DNA (cuts, color, BPM, hooks)
- **Stores** extracted clips in a structured, tag-indexed library
- **Generates** Remotion compositions tuned to each archetype's proven patterns
- **Renders** production-ready short-form videos at scale

Supported archetypes: **GlowUp**, **FrameMaxxing**, **SkinMaxxing**, **StyleMaxxing**

---

## Quick Start

```bash
# Install all dependencies
npm install
pip install PySceneDetect librosa imagehash watchdog scenedetect Pillow numpy
brew install ffmpeg

# Start the Remotion Studio
npx remotion studio

# Render a composition
npx remotion render remotion/VideoTemplate GlowUp_v1 out/glowup_v1.mp4
```

---

## Project Structure

```
ascension-engine/
├── data/
│   ├── ingest.py         — Gold ingest pipeline (v2.1)
│   ├── analyze.py        — Manual analysis utilities
│   └── schema.sql        — Database schema for clip metadata
├── remotion/
│   ├── VideoTemplate.tsx — Main Remotion composition
│   └── remotion-presets.ts — Archetype preset configs (v2.1)
├── src/
│   └── lib/
│       └── clipLibrary.ts — Clip library TypeScript module (v2.1)
├── library/
│   ├── raw/              — Original source video copies
│   ├── clips/            — Extracted .mp4 scene segments
│   ├── assets/
│   │   ├── video/        — Additional video assets
│   │   ├── audio/        — Extracted audio tracks (.wav)
│   │   └── thumbnails/   — Scene thumbnail frames (.jpg)
│   └── tags/
│       └── index.json    — Tag-to-clip-ID index
├── style-profiles/       — Per-video style-profile.json files
├── input/
│   └── gold/             — DROP ZONE: add .mp4 files here
├── clip-manifest.json    — Master clip catalog
├── claude-brain-spec.md  — Claude's operating instructions
└── README.md             — This file
```

---

## Archetypes & Presets

| Archetype    | Color Grade  | Cut Rate | BPM     | Hook Style         |
|:-------------|:-------------|:---------|:--------|:-------------------|
| GlowUp       | WarmGold     | 2.2s     | 120–145 | Before/After Cut   |
| FrameMaxxing | TealOrange   | 1.6s     | 135–165 | Face Reveal        |
| SkinMaxxing  | ColdBlue     | 3.0s     | 90–120  | Face Reveal        |
| StyleMaxxing | Desaturated  | 2.0s     | 105–135 | Transform Freeze   |

Presets are defined in `remotion/remotion-presets.ts` and can be overridden
at render time with a matching `style-profiles/<video_id>.json`.

---

## DNA Engine v2.1 — Gold Ingest

The DNA Engine transforms raw reference footage into structured, reusable
clip DNA that powers Remotion compositions.

---

### How to Ingest a Gold Video

1. **Drop** your `.mp4` file into `~/Ascension-Engine/input/gold/`
2. The ingest pipeline runs automatically (watch mode) or you can trigger manually:

```bash
# Watch mode — auto-processes any new .mp4 dropped into input/gold/
python data/ingest.py

# Single-file ingest
python data/ingest.py --once ~/Ascension-Engine/input/gold/myvideo.mp4

# Dry run — preview all steps without writing files
python data/ingest.py --dry-run --once ~/Ascension-Engine/input/gold/myvideo.mp4
```

---

### What the Pipeline Does Automatically

For each new `.mp4` file the ingest ritual:

1. **Scene Detection** — runs PySceneDetect (threshold 0.4) to find cut boundaries
2. **Keyframe Extraction** — pulls one frame per scene + every 1s via FFmpeg
3. **Audio Analysis** — extracts WAV, then runs librosa for BPM, beat grid, energy peaks
4. **Color Profiling** — samples frames to classify color grade (WarmGold, ColdBlue, etc.)
5. **Style Profile** — writes `style-profiles/<video_id>.json` with all DNA fields
6. **Clip Segmentation** — exports clean `.mp4` per scene to `library/clips/`
7. **Manifest Update** — adds all new clips to `clip-manifest.json`
8. **Tag Index Update** — updates `library/tags/index.json`
9. **Git Commit** — auto-commits everything with `Gold ingest: <video_id>`
10. **Summary Report** — prints duration, scene count, BPM, color grade, cut stats

---

### Clip Library Folder Structure

```
library/
├── raw/                          — Original .mp4 copies (source of truth)
├── clips/                        — Scene-segmented .mp4 clips
│   ├── jordan_abc12345_scene000.mp4
│   ├── jordan_abc12345_scene001.mp4
│   └── ...
├── assets/
│   ├── video/                    — Supplemental video assets
│   ├── audio/                    — Extracted .wav audio tracks
│   │   └── jordan_abc12345.wav
│   └── thumbnails/               — Scene thumbnail .jpg frames
│       ├── jordan_abc12345/
│       │   ├── every1s_0001.jpg
│       │   ├── scene_0000_0.00s.jpg
│       │   └── ...
│       └── jordan_abc12345_scene000.jpg  (mid-clip thumb)
└── tags/
    └── index.json                — Tag → [clip_id, ...] lookup table
```

---

### Using clipLibrary.ts in Remotion

```typescript
import {
  getClipsByTags,
  getBlueprintFromReference,
  getTopClips,
  updateClipRank,
} from '../src/lib/clipLibrary';

// Pull 6 face-closeup clips ranked above 0.6
const closeups = getClipsByTags(['face_closeup', 'glow_up_reveal'], 6, 0.6);

// Build a full timeline blueprint from a reference video's style profile
const blueprint = getBlueprintFromReference('jordan_abc12345');
// blueprint.clip_sequence → ordered array of clip IDs
// blueprint.cut_rate_sec  → avg cut rate
// blueprint.bpm           → detected BPM for music sync

// Get top 10 clips from the whole library
const best = getTopClips(10, 0.7);

// Adjust a clip's quality rank after review
updateClipRank('jordan_abc12345_scene003', +0.1);
```

---

### DNA Transfer Generation

```bash
# Full DNA transfer: reference video → new Remotion composition
npx ts-node src/scripts/dna-transfer.ts \
  --reference jordan_abc12345 \
  --archetype GlowUp \
  --clip-count 12 \
  --output remotion/compositions/glowup_jordan_v1.tsx

# Quick render from preset only (no reference video)
npx ts-node src/scripts/dna-transfer.ts \
  --archetype FrameMaxxing \
  --output remotion/compositions/framemaxxing_v1.tsx
```

---

### Maintenance Commands

```bash
# Deduplicate clips with the same perceptual hash
python data/ingest.py --maintenance deduplicate

# Remove clips ranked below 0.4
python data/ingest.py --maintenance prune --min-rank 0.4

# Normalize all ranks to [0, 1]
python data/ingest.py --maintenance normalize

# Rebuild the tags index from the manifest
python data/ingest.py --maintenance rebuild-tags

# Full library audit
python data/ingest.py --maintenance audit
```

---

## Contributing

1. Drop gold footage into `input/gold/`
2. Let the ingest pipeline run
3. Review the style-profile and adjust `user_notes` if needed
4. Tag clips manually in `clip-manifest.json` or via `updateClipRank()`
5. Run a DNA transfer to generate a new composition
6. Render and review — update `performance_link` in the style-profile after posting

---

## License

Private. All content and tooling is proprietary to the Ascension Engine project.
