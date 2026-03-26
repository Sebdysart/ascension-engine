#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  ASCENSION ENGINE — First Edit
#  Multi-clip mog edit: bp_masterx + roge.editz + morphedmanlet
#  Audio: bp_masterx original track (phonk/mog)
#  Grade: dark_cinema | Overlay: ASCENSION watermark
# ─────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

CLIPS="input/gold"
ASSETS="assets"
OUT="out"
TMP="tmp_edit"
mkdir -p "$OUT" "$TMP"

DARK_CINEMA="eq=brightness=-0.1:contrast=1.35:saturation=0.85,hue=s=0.85"
CROP_16x9="crop=ih*9/16:ih,scale=1080:1920:flags=lanczos"
ZOOM_SLOW="zoompan=z='min(zoom+0.0006,1.05)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"
ZOOM_MED="zoompan=z='min(zoom+0.001,1.08)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"

echo ""
echo "════════════════════════════════════════"
echo " ASCENSION ENGINE — First Edit"
echo "════════════════════════════════════════"

# ── CLIP 1: bp_masterx, 0s-7s ────────────────────────────────────
echo "▶ Processing clip 1 (bp_masterx 0-7s)"
ffmpeg -y -ss 0 -t 7 \
  -i "$CLIPS/bp_masterx_7589676463071268118.mp4" \
  -vf "${CROP_16x9},${DARK_CINEMA},${ZOOM_SLOW}" \
  -c:v libx264 -preset fast -crf 17 -an \
  "$TMP/c1.mp4" 2>/dev/null
echo "  ✅ clip 1 done"

# ── CLIP 2: roge.editz, 4s-10s (already 9:16, no crop) ───────────
echo "▶ Processing clip 2 (roge.editz 4-10s)"
ffmpeg -y -ss 4 -t 6 \
  -i "$CLIPS/roge.editz_7588525512478166294.mp4" \
  -vf "scale=1080:1920:flags=lanczos,${DARK_CINEMA},${ZOOM_MED}" \
  -c:v libx264 -preset fast -crf 17 -an \
  "$TMP/c2.mp4" 2>/dev/null
echo "  ✅ clip 2 done"

# ── CLIP 3: morphedmanlet, 3s-10s ─────────────────────────────────
echo "▶ Processing clip 3 (morphedmanlet 3-10s)"
ffmpeg -y -ss 3 -t 7 \
  -i "$CLIPS/morphedmanlet_7540813336661740855.mp4" \
  -vf "${CROP_16x9},${DARK_CINEMA},${ZOOM_SLOW}" \
  -c:v libx264 -preset fast -crf 17 -an \
  "$TMP/c3.mp4" 2>/dev/null
echo "  ✅ clip 3 done"

# ── CONCAT the three video clips ──────────────────────────────────
echo "▶ Concatenating 3 clips (20s total)"
printf "file 'c1.mp4'\nfile 'c2.mp4'\nfile 'c3.mp4'\n" > "$TMP/list.txt"
ffmpeg -y -f concat -safe 0 -i "$TMP/list.txt" -c copy "$TMP/concat.mp4" 2>/dev/null
echo "  ✅ concat done"

# ── OVERLAY watermark + mix audio from bp_masterx ────────────────
echo "▶ Applying ASCENSION watermark + audio from bp_masterx"
ffmpeg -y \
  -i "$TMP/concat.mp4" \
  -i "$ASSETS/watermark.png" \
  -ss 0 -t 20 -i "$CLIPS/bp_masterx_7589676463071268118.mp4" \
  -filter_complex "[0:v][1:v]overlay=0:0[vout]" \
  -map "[vout]" \
  -map 2:a \
  -c:v libx264 -preset fast -crf 17 -c:a aac -b:a 192k \
  -shortest \
  "$OUT/first_edit.mp4"
echo "  ✅ watermark + audio done"

# ── CLEANUP ───────────────────────────────────────────────────────
rm -rf "$TMP"

echo ""
echo "════════════════════════════════════════"
echo " First edit complete!"
echo "════════════════════════════════════════"
ls -lh "$OUT/first_edit.mp4"
echo ""
echo "Post checklist:"
echo "  ✅ 1080x1920 (TikTok native)"
echo "  ✅ dark_cinema color grade"
echo "  ✅ ASCENSION watermark"
echo "  ✅ push-in zoom (cinematic)"
echo "  ✅ phonk audio from bp_masterx reference"
echo ""
echo "Caption formula: 'lol' or 'obviously 😌' + #looksmax #mog #ascension"
