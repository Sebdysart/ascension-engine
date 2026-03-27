# Ascension Engine v4.0 — Brutal BP/Mog DNA Library Factory

AI-powered, self-evolving video production system for pure blackpill / looksmax / mog cinematic montages. Built exclusively for 15-second, 1080x1920 TikTok/Reels/Shorts that match the exact DNA of your gold references (@black.pill.city, @roge.editz, @maxxed.bp, @marlon3edxtz, @wxyne.psl, @darvikz710, @editaccount192, @iblameredeye, @jordangamercool23, @justgetbetter72).

The library **is** the editor. Everything else exists to serve it.

### What Is This
A closed-loop, max-tier BP/mog editor that runs on a single Mac mini using Claude Computer Use + Remotion + Python.

It does three things at god-tier level:
- **Deciphers** every edit you drop (gold or its own outputs) → removes text, extracts only the best parts, builds reusable sequence templates and grade presets.
- **Enriches** a living DNA library with cleaned clips, good_parts (text-removed), sequence templates, minimal text templates, and brutal_bp_blueprints.
- **Generates** 8–12 new brutal mog montages per batch using only library DNA — 0.3–1.5s beat-synced cuts on phonk, dark cinematic teal/orange grade with crushed blacks, faces as the entire hook, near-zero text, no CTA cards, no motivational slop.

The engine never asks for archetypes. It never outputs personal-brand or glow-up motivational format. It stays locked to the exact brutal BP/mog style of your gold references forever.

### Core Features (v4.0)
- Full Decipher Ritual on every ingest (text removal + good-part cropping + sequence template extraction)
- Self-ranking library with good_parts, sequence_templates, grade_presets, text_templates
- BrutalBeatMontage.tsx — pure beat-synced montage composer
- BP Fidelity Gate (SSIM + perceptual hash + beat alignment + vision critique)
- Closed daily loop: ingest → decipher → generate → gate → human APPROVE → post
- Auto Git commits + library index updates
- Zero external services beyond Runway (optional hero shots) and Claude API

### Quick Start

```bash
cd ~/Ascension-Engine

# Install dependencies (one-time)
npm install
pip install PySceneDetect librosa imagehash watchdog scenedetect Pillow numpy opencv-python

brew install ffmpeg

# Start Remotion Studio for preview
npx remotion studio

# Trigger full ingest on anything in input/gold/ or input/raw/
python decipher_ingest.py --watch

# Or manual batch
python decipher_ingest.py --once input/gold/myvideo.mp4
```

### Project Structure (Updated for v4.0)

```
Ascension-Engine/
├── library/                      # ← THE MOAT — everything lives here
│   ├── raw/                      # Original gold MP4s
│   ├── good_parts/               # Text-removed, cropped impact clips
│   ├── assets/
│   │   ├── video/                # Cleaned full clips
│   │   ├── audio/                # Isolated phonk drops + beats
│   │   └── thumbnails/
│   ├── clips/                    # Scene-segmented manifests
│   ├── sequence_templates/       # Full beat-synced timelines from golds
│   ├── text_templates/           # Extracted minimal impact text
│   ├── grade_presets/            # LUTs + histogram profiles (.cube + json)
│   ├── blueprints/               # brutal_bp_blueprint.json files
│   └── tags/
│       └── index.json            # Hierarchical tag → ranked clips
├── style-profiles/               # global_brutal_bp.json + per-video
├── input/
│   ├── gold/                     # DROP ZONE for your manual bangers
│   └── raw/                      # For generated or client edits
├── remotion/
│   ├── BrutalBeatMontage.tsx     # ← Main composition (pure montage)
│   └── remotion-presets.ts
├── src/
│   └── lib/
│       └── clipLibrary.ts        # Library query engine
├── decipher_ingest.py            # ← The new core ingest script
├── editing-rules.md
├── claude-brain-spec.md          # v4.0 instructions (paste this into Claude)
└── README.md                     # This file
```

### How the DNA Library Works (The Only Thing That Matters)

Every time you drop an edit, the **Decipher Ritual** runs automatically:

1. PySceneDetect (threshold 0.32, min-scene-len 0.22s) + librosa beat detection
2. Claude Vision scans every key frame
3. FFmpeg removes any text overlays while preserving faces
4. Extracts "good parts" (jawline pops, micro-zooms, impact frames)
5. Builds sequence_template.json (full timeline DNA)
6. Saves grade preset + minimal text templates
7. Updates global_brutal_bp.json with rolling averages
8. Ranks every new asset in the library

Future generations remix only from this library → the engine literally evolves your exact taste.

### Generation Flow

After ingest, run in Claude:

```
Run full library status then generate the first 8 brutal mog edits using current library DNA. Use only BrutalBeatMontage. Enforce dark cinematic grade, 0.3-1.5s beat-synced cuts, text density ≤0.05, no CTA, no archetypes. Show previews + fidelity scores.
```

Claude will:
- Pull high-rank good_parts + sequence templates
- Render via BrutalBeatMontage
- Run the BP Fidelity Gate (SSIM ≥0.94, beat offset ≤40ms, etc.)
- Show previews + scores
- Wait for your `APPROVE`, `APPROVE ALL`, or `REGENERATE`

### Maintenance

```bash
python decipher_ingest.py --maintenance deduplicate
python decipher_ingest.py --maintenance prune --min-rank 0.4
python decipher_ingest.py --maintenance audit
```

### Claude Brain Instructions (Critical)
The file `claude-brain-spec.md` contains the exact v4.0 prompt. Paste its entire content into your Claude Project as the system prompt. This locks the behavior permanently.

You're now running the real max-tier factory we designed.

Drop your next gold video into `input/gold/` and run the generation command above. The library will grow, the edits will get sharper, and the engine will feel like a top 1% BP editor that never sleeps.
