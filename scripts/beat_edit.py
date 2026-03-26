#!/usr/bin/env python3
"""
Beat-timed edit builder for Ascension Engine.
Detects beats in source audio, selects best clip windows from gold footage,
then outputs an FFmpeg concat script timed to the music.
"""
import subprocess, json, os, sys
import numpy as np

GOLD = "/Users/sebastiandysart/Desktop/ascension-engine/input/gold"
OUT  = "/Users/sebastiandysart/Desktop/ascension-engine/out"
TMP  = "/Users/sebastiandysart/Desktop/ascension-engine/tmp_beat"
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ── 1. AUDIO SOURCE ───────────────────────────────────────────────
# bp_masterx: highest engagement, strong phonk track
AUDIO_SRC = f"{GOLD}/bp_masterx_7589676463071268118.mp4"

print("▶ Extracting audio for beat detection...")
wav_path = f"{TMP}/audio.wav"
subprocess.run([
    "ffmpeg", "-y", "-i", AUDIO_SRC,
    "-ac", "1", "-ar", "22050", "-vn", wav_path
], capture_output=True)

# ── 2. BEAT DETECTION ─────────────────────────────────────────────
print("▶ Running beat detection...")
import librosa
y, sr = librosa.load(wav_path, sr=22050)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='time')
beat_times = beat_frames  # already in seconds since units='time'

# Filter to first 20 seconds
beat_times = [t for t in beat_times if t <= 20.0]
print(f"  BPM: {float(tempo):.1f}  |  Beats in 20s: {len(beat_times)}")
print(f"  Beat timestamps: {[round(t,2) for t in beat_times[:16]]}")

# ── 3. CLIP SCHEDULE ──────────────────────────────────────────────
# Build cut points from beats — every 2 beats = one clip (feels energetic but not chaotic)
# For a 15-20s edit we want ~6-8 cuts
beat_interval = 60.0 / float(tempo)
print(f"\n▶ Beat interval: {beat_interval:.2f}s — scheduling cuts every 2 beats")

# Pick cut points: every 2 beats
cut_points = [beat_times[i] for i in range(0, len(beat_times)-1, 2)]
cut_points = cut_points[:9]  # max 9 cuts = 8 segments
if cut_points[-1] < 14.0:
    cut_points.append(min(20.0, cut_points[-1] + beat_interval * 2))

print(f"  Cut points: {[round(t,2) for t in cut_points]}")

# ── 4. SOURCE CLIPS ───────────────────────────────────────────────
# Assign source videos to segments — rotate through gold sources
# Using the highest-engagement clips for visual variety
# bp_masterx = selfie/mog close-up | roge.editz = cinematic | bp4ever.ae = lifestyle
# morphedmanlet = transformation | bp.pilled = atmospheric

sources = [
    # (file, start_offset, description)
    # Start strong: bp_masterx face close-up (best performing source)
    (f"{GOLD}/bp_masterx_7589676463071268118.mp4",  2.0, "bp_masterx close-up"),
    (f"{GOLD}/roge.editz_7588525512478166294.mp4",   6.0, "roge.editz cinematic"),
    (f"{GOLD}/bp_masterx_7589676463071268118.mp4",  10.0, "bp_masterx alt angle"),
    (f"{GOLD}/morphedmanlet_7540813336661740855.mp4", 4.0, "morphedmanlet"),
    (f"{GOLD}/bp.pilled_7580194747986283790.mp4",    8.0, "bp.pilled atmospheric"),
    (f"{GOLD}/bp_masterx_7589676463071268118.mp4",  18.0, "bp_masterx peak"),
    (f"{GOLD}/roge.editz_7588525512478166294.mp4",  12.0, "roge.editz alt"),
    (f"{GOLD}/bp4ever.ae_7542208986699844869.mp4",   2.0, "bp4ever.ae lifestyle"),
]

# ── 5. RENDER INDIVIDUAL BEAT CLIPS ──────────────────────────────
DARK_CINEMA = "eq=brightness=-0.1:contrast=1.35:saturation=0.85,hue=s=0.85"
CROP_16_9   = "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos"
SCALE_ONLY  = "scale=1080:1920:flags=lanczos"

segments = []
for i, cut_start in enumerate(cut_points[:-1]):
    cut_end   = cut_points[i + 1]
    duration  = cut_end - cut_start

    src_file, src_offset, desc = sources[i % len(sources)]

    # Detect if source is already 9:16
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", src_file
    ], capture_output=True, text=True)
    try:
        w, h = map(int, probe.stdout.strip().split(','))
        is_916 = abs((w / h) - (9/16)) < 0.05
    except:
        is_916 = False

    crop_filter = SCALE_ONLY if is_916 else CROP_16_9

    # Speed ramp: slow down the first beat-clip slightly for impact, fast on others
    # First clip: 0.9x speed (slightly slow, dramatic entrance)
    speed_vf = ""
    if i == 0:
        speed_vf = ",setpts=1.1*PTS"  # 10% slower on first clip = cinematic

    seg_path = f"{TMP}/seg_{i:02d}.mp4"
    vf = f"{crop_filter},{DARK_CINEMA}{speed_vf},zoompan=z='min(zoom+0.0006,1.04)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"

    print(f"  Rendering seg {i+1}: {desc} | {duration:.2f}s | src @{src_offset}s")

    result = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(src_offset),
        "-t",  str(duration + 0.1),  # tiny overlap, trim on concat
        "-i",  src_file,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        seg_path
    ], capture_output=True)

    if result.returncode == 0:
        segments.append(seg_path)
        print(f"    ✅ done ({os.path.getsize(seg_path) // 1024} KB)")
    else:
        print(f"    ❌ failed: {result.stderr[-200:]}")

print(f"\n▶ Rendered {len(segments)} segments")

# ── 6. CONCAT + AUDIO ────────────────────────────────────────────
print("▶ Concatenating segments...")
concat_list = f"{TMP}/concat.txt"
with open(concat_list, 'w') as f:
    for seg in segments:
        f.write(f"file '{seg}'\n")

concat_raw = f"{TMP}/concat_raw.mp4"
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", concat_list, "-c", "copy", concat_raw
], capture_output=True)

# ── 7. WATERMARK OVERLAY + AUDIO MIX ────────────────────────────
print("▶ Adding watermark + audio...")
assets = "/Users/sebastiandysart/Desktop/ascension-engine/assets"
final_path = f"{OUT}/beat_edit_v1.mp4"

total_dur = cut_points[-1]

result = subprocess.run([
    "ffmpeg", "-y",
    "-i", concat_raw,
    "-i", f"{assets}/watermark.png",
    "-ss", "0", "-t", str(total_dur), "-i", AUDIO_SRC,
    "-filter_complex", "[0:v][1:v]overlay=0:0[vout]",
    "-map", "[vout]", "-map", "2:a",
    "-c:v", "libx264", "-preset", "fast", "-crf", "17",
    "-c:a", "aac", "-b:a", "192k",
    "-shortest", final_path
], capture_output=True)

if result.returncode == 0:
    size_mb = os.path.getsize(final_path) / (1024*1024)
    print(f"\n✅ Beat edit complete: {final_path} ({size_mb:.1f} MB)")
    print(f"   Duration: ~{total_dur:.1f}s | {len(segments)} cuts | BPM: {float(tempo):.1f}")
    print(f"\n   Cut breakdown:")
    for i, (s, e) in enumerate(zip(cut_points[:-1], cut_points[1:])):
        src_file, src_offset, desc = sources[i % len(sources)]
        print(f"   [{s:.2f}s→{e:.2f}s] {desc}")
else:
    print(f"❌ Final render failed: {result.stderr[-400:]}")
