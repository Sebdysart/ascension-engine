#!/bin/bash
# Render first batch of 3 videos from gold reference clips
# Run from project root: bash scripts/render_first_batch.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p out

echo "======================================="
echo " ASCENSION ENGINE — First Batch Render"
echo "======================================="

# ── VIDEO 1: Selfie Mog (bp_masterx style, dark_cinema grade) ──
echo ""
echo "▶ Rendering Video 1: Selfie Mog (dark_cinema)"
npx remotion render remotion/index.tsx MogEdit out/v1_selfie_mog.mp4 \
  --props='{"clips":[{"src":"clips/bp_masterx_7589676463071268118.mp4","trimStart":0},{"src":"clips/roge.editz_7588525512478166294.mp4","trimStart":10}],"colorGrade":"dark_cinema","watermark":"ASCENSION","showLyric":false}' \
  --frames=0-449 \
  2>&1 | tail -5
echo "✅ Video 1 done: out/v1_selfie_mog.mp4"

# ── VIDEO 2: Selfie Mog (natural_indoor grade) ──
echo ""
echo "▶ Rendering Video 2: Selfie Mog (natural_indoor)"
npx remotion render remotion/index.tsx MogEdit out/v2_natural_mog.mp4 \
  --props='{"clips":[{"src":"clips/bp_masterx_7589676463071268118.mp4","trimStart":0},{"src":"clips/morphedmanlet_7540813336661740855.mp4","trimStart":5}],"colorGrade":"natural_indoor","watermark":"ASCENSION","showLyric":false}' \
  --frames=0-449 \
  2>&1 | tail -5
echo "✅ Video 2 done: out/v2_natural_mog.mp4"

# ── VIDEO 3: BP Rating reveal ──
echo ""
echo "▶ Rendering Video 3: BP Rating reveal"
npx remotion render remotion/index.tsx BpRating out/v3_bp_rating.mp4 \
  --props='{"clipPath":"clips/bp4ever.ae_7542208986699844869.mp4","revealWord":"MOGGED","colorGrade":"natural","watermark":"ASCENSION"}' \
  --frames=0-299 \
  2>&1 | tail -5
echo "✅ Video 3 done: out/v3_bp_rating.mp4"

echo ""
echo "======================================="
echo " All renders complete!"
ls -lh out/*.mp4
echo "======================================="
