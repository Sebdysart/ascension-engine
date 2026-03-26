#!/bin/bash
# Download all 10 gold reference videos to input/gold/
set -e
OUTDIR="$(dirname "$0")/../input/gold"
mkdir -p "$OUTDIR"

URLS=(
  "https://www.tiktok.com/@black.pill.city/video/7600958206642326806"
  "https://www.tiktok.com/@bp.pilled/video/7580194747986283790"
  "https://www.tiktok.com/@roge.editz/video/7588525512478166294"
  "https://www.tiktok.com/@morphedmanlet/video/7540813336661740855"
  "https://www.tiktok.com/@unc_iiris/video/7537476213275757846"
  "https://www.tiktok.com/@saffyro_/video/7608533082261474582"
  "https://www.tiktok.com/@bp4ever.ae/video/7542208986699844869"
  "https://www.tiktok.com/@bp.editz04/video/7596127531955244318"
  "https://www.tiktok.com/@haifluke/video/7618372975192018189"
  "https://www.tiktok.com/@bp_masterx/video/7589676463071268118"
)

for URL in "${URLS[@]}"; do
  echo "Downloading: $URL"
  yt-dlp \
    --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    --merge-output-format mp4 \
    --output "$OUTDIR/%(uploader)s_%(id)s.%(ext)s" \
    --no-playlist \
    "$URL" || echo "FAILED: $URL"
done

echo "Done. Files:"
ls -lh "$OUTDIR"
