#!/usr/bin/env python3
"""
Cinematic beat edit — cuts every 4 beats (~2.6s per clip).
Slower, more brooding. Better for dark_cinema atmospheric style.
"""
import subprocess, os
import numpy as np

GOLD  = "/Users/sebastiandysart/Desktop/ascension-engine/input/gold"
OUT   = "/Users/sebastiandysart/Desktop/ascension-engine/out"
TMP   = "/Users/sebastiandysart/Desktop/ascension-engine/tmp_beat2"
os.makedirs(TMP, exist_ok=True)

AUDIO_SRC   = f"{GOLD}/bp_masterx_7589676463071268118.mp4"
DARK_CINEMA = "eq=brightness=-0.1:contrast=1.35:saturation=0.85,hue=s=0.85"
CROP_16_9   = "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos"
SCALE_ONLY  = "scale=1080:1920:flags=lanczos"
ASSETS      = "/Users/sebastiandysart/Desktop/ascension-engine/assets"

print("▶ Loading beats...")
import librosa
wav = f"{TMP}/audio.wav"
subprocess.run(["ffmpeg","-y","-i",AUDIO_SRC,"-ac","1","-ar","22050","-vn",wav],capture_output=True)
y, sr = librosa.load(wav, sr=22050)
tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
beat_times = [t for t in beat_times if t <= 26.0]
bpm = float(tempo)
print(f"  BPM: {bpm:.1f}")

# Cuts every 4 beats = more breathing room, cinematic feel
cut_points = [beat_times[i] for i in range(0, len(beat_times)-1, 4)]
cut_points = cut_points[:8]
print(f"  {len(cut_points)} cut points: {[round(t,2) for t in cut_points]}")

# Source clips — pick the visually strongest moments
# Key insight from DNA analysis: the FIRST clip must be the strongest face shot
# Transitions: wide → close → wide → close pattern (creates tension/release)
sources = [
    # (file, start, is_916_already, description)
    (f"{GOLD}/bp_masterx_7589676463071268118.mp4",  3.5, False, "bp_masterx — strongest face moment"),
    (f"{GOLD}/roge.editz_7588525512478166294.mp4",   8.0, True,  "roge.editz — cinematic wide"),
    (f"{GOLD}/bp_masterx_7589676463071268118.mp4", 14.0, False, "bp_masterx — second angle"),
    (f"{GOLD}/morphedmanlet_7540813336661740855.mp4", 6.0, False, "morphedmanlet — peak frame"),
    (f"{GOLD}/bp.pilled_7580194747986283790.mp4",   12.0, False, "bp.pilled — mood shot"),
    (f"{GOLD}/bp_masterx_7589676463071268118.mp4",  21.0, False, "bp_masterx — final hold"),
    (f"{GOLD}/roge.editz_7588525512478166294.mp4",  15.0, True,  "roge.editz — closing wide"),
]

segments = []
for i in range(len(cut_points) - 1):
    t_start = cut_points[i]
    t_end   = cut_points[i + 1]
    dur     = t_end - t_start

    src, src_off, already_916, desc = sources[i % len(sources)]
    crop = SCALE_ONLY if already_916 else CROP_16_9

    # Subtle push-in zoom — start slightly zoomed out, push in over the clip
    zoom_expr = "min(zoom+0.0004,1.035)"
    vf = f"{crop},{DARK_CINEMA},zoompan=z='{zoom_expr}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"

    seg = f"{TMP}/seg_{i:02d}.mp4"
    r = subprocess.run([
        "ffmpeg","-y","-ss",str(src_off),"-t",str(dur+0.05),"-i",src,
        "-vf",vf,"-c:v","libx264","-preset","fast","-crf","17","-an",seg
    ], capture_output=True)

    status = "✅" if r.returncode == 0 else "❌"
    kb = os.path.getsize(seg)//1024 if r.returncode==0 else 0
    print(f"  {status} seg {i+1}: {desc} | {dur:.2f}s | {kb}KB")
    if r.returncode == 0:
        segments.append(seg)

print(f"\n▶ Concatenating {len(segments)} segments...")
with open(f"{TMP}/list.txt","w") as f:
    for s in segments: f.write(f"file '{s}'\n")

concat = f"{TMP}/concat.mp4"
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",f"{TMP}/list.txt","-c","copy",concat], capture_output=True)

total = cut_points[len(segments)]
final = f"{OUT}/beat_edit_cinematic.mp4"
print(f"▶ Final render with watermark + audio ({total:.1f}s)...")
r = subprocess.run([
    "ffmpeg","-y",
    "-i", concat,
    "-i", f"{ASSETS}/watermark.png",
    "-ss","0","-t",str(total),"-i",AUDIO_SRC,
    "-filter_complex","[0:v][1:v]overlay=0:0[vout]",
    "-map","[vout]","-map","2:a",
    "-c:v","libx264","-preset","fast","-crf","17",
    "-c:a","aac","-b:a","192k","-shortest",final
], capture_output=True)

if r.returncode == 0:
    mb = os.path.getsize(final)/(1024*1024)
    print(f"\n✅ Cinematic edit: {final} ({mb:.1f} MB, {total:.1f}s)")
    for i in range(len(segments)):
        s, e = cut_points[i], cut_points[i+1]
        src, _, _, desc = sources[i % len(sources)]
        print(f"   [{s:.2f}→{e:.2f}s] {desc}")
else:
    print(f"❌ {r.stderr[-300:]}")
