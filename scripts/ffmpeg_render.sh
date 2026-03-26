#!/bin/bash
# FFmpeg render pipeline — TikTok-ready 1080x1920 MP4s
# Color grades + cinematic push-in zoom + ASCENSION watermark overlay
# Audio: original track from each source video
# Run from project root: bash scripts/ffmpeg_render.sh

set -e
cd "$(dirname "$0")/.."
CLIPS="input/gold"
ASSETS="assets"
OUT="out"
mkdir -p "$OUT"

# Ensure overlays exist
if [ ! -f "$ASSETS/watermark.png" ]; then
  echo "Generating overlays..."
  python3 scripts/gen_overlays.py
fi

echo "======================================="
echo " ASCENSION ENGINE — FFmpeg Batch Render"
echo "======================================="

CROP_SCALE="crop=ih*9/16:ih,scale=1080:1920:flags=lanczos"
DARK_CINEMA="eq=brightness=-0.1:contrast=1.35:saturation=0.85,hue=s=0.85"
NATURAL="eq=brightness=0:contrast=1.05:saturation=1.05"
NATURAL_INDOOR="eq=brightness=-0.03:contrast=1.08:saturation=1.0"
WARM_AMBIENT="eq=brightness=-0.08:contrast=1.2:saturation=1.1,hue=h=5:s=1.1"

# ── VIDEO 1: Selfie Mog — @bp_masterx, dark_cinema ──────────────
echo ""
echo "▶ Video 1: Selfie Mog (dark_cinema) — @bp_masterx"
ffmpeg -y \
  -i "$CLIPS/bp_masterx_7589676463071268118.mp4" \
  -i "$ASSETS/watermark.png" \
  -filter_complex "${CROP_SCALE},${DARK_CINEMA},zoompan=z='min(zoom+0.0008,1.06)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[v];[v][1:v]overlay=0:0[vout]" \
  -map "[vout]" -map 0:a \
  -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k \
  -t 15 \
  "$OUT/v1_selfie_mog_dark.mp4"
echo "✅ Video 1 done"

# ── VIDEO 2: Selfie Mog — @bp_masterx, natural_indoor ───────────
echo ""
echo "▶ Video 2: Selfie Mog (natural_indoor) — @bp_masterx"
ffmpeg -y \
  -i "$CLIPS/bp_masterx_7589676463071268118.mp4" \
  -i "$ASSETS/watermark.png" \
  -filter_complex "${CROP_SCALE},${NATURAL_INDOOR},zoompan=z='min(zoom+0.0008,1.06)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[v];[v][1:v]overlay=0:0[vout]" \
  -map "[vout]" -map 0:a \
  -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k \
  -t 15 \
  "$OUT/v2_selfie_mog_natural.mp4"
echo "✅ Video 2 done"

# ── VIDEO 3: Mog Edit — @roge.editz, dark_cinema ─────────────────
echo ""
echo "▶ Video 3: Cinematic Mog Edit (dark_cinema) — @roge.editz"
ffmpeg -y \
  -i "$CLIPS/roge.editz_7588525512478166294.mp4" \
  -i "$ASSETS/watermark.png" \
  -filter_complex "scale=1080:1920:flags=lanczos,${DARK_CINEMA},zoompan=z='min(zoom+0.001,1.08)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[v];[v][1:v]overlay=0:0[vout]" \
  -map "[vout]" -map 0:a \
  -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k \
  -t 15 \
  "$OUT/v3_mog_edit_dark.mp4"
echo "✅ Video 3 done"

# ── VIDEO 4: BP Rating — @bp4ever.ae + MOGGED word reveal ─────────
echo ""
echo "▶ Video 4: BP Rating (MOGGED reveal) — @bp4ever.ae"
ffmpeg -y \
  -i "$CLIPS/bp4ever.ae_7542208986699844869.mp4" \
  -i "$ASSETS/mogged_M.png" \
  -i "$ASSETS/mogged_MO.png" \
  -i "$ASSETS/mogged_MOG.png" \
  -i "$ASSETS/mogged_MOGG.png" \
  -i "$ASSETS/mogged_MOGGE.png" \
  -i "$ASSETS/mogged_MOGGED.png" \
  -i "$ASSETS/watermark.png" \
  -filter_complex "
    ${CROP_SCALE},${NATURAL},zoompan=z='min(zoom+0.0005,1.05)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[base];
    [base][1:v]overlay=0:0:enable='between(t,1.0,2.4)'[v1];
    [v1][2:v]overlay=0:0:enable='between(t,2.5,3.9)'[v2];
    [v2][3:v]overlay=0:0:enable='between(t,4.0,5.4)'[v3];
    [v3][4:v]overlay=0:0:enable='between(t,5.5,6.9)'[v4];
    [v4][5:v]overlay=0:0:enable='between(t,7.0,8.4)'[v5];
    [v5][6:v]overlay=0:0:enable='gte(t,8.5)'[v6];
    [v6][7:v]overlay=0:0[vout]
  " \
  -map "[vout]" -map 0:a \
  -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k \
  -t 12 \
  "$OUT/v4_bp_rating_mogged.mp4"
echo "✅ Video 4 done"

# ── VIDEO 5: Jester Contrast — @unc_iiris, warm_ambient ──────────
echo ""
echo "▶ Video 5: Jester Contrast (warm_ambient) — @unc_iiris"
ffmpeg -y \
  -i "$CLIPS/unc_iiris_7537476213275757846.mp4" \
  -i "$ASSETS/watermark.png" \
  -filter_complex "${CROP_SCALE},${WARM_AMBIENT},zoompan=z='min(zoom+0.0005,1.04)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[v];[v][1:v]overlay=0:0[vout]" \
  -map "[vout]" -map 0:a \
  -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k \
  -t 15 \
  "$OUT/v5_jester_contrast.mp4"
echo "✅ Video 5 done"

echo ""
echo "======================================="
echo " All renders complete!"
echo "======================================="
ls -lh "$OUT"/*.mp4
