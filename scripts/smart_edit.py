#!/usr/bin/env python3
"""
Smart edit builder:
1. Samples frames from source videos, scores them for visual quality
2. Picks the best windows (well-lit face, not silhouette, not blown out)
3. Applies warm cinematic grade matching gold reference aesthetic
4. Builds beat-timed edit
"""
import subprocess, os, json
import numpy as np
from PIL import Image
import io

GOLD   = "/Users/sebastiandysart/Desktop/ascension-engine/input/gold"
OUT    = "/Users/sebastiandysart/Desktop/ascension-engine/out"
TMP    = "/Users/sebastiandysart/Desktop/ascension-engine/tmp_smart"
ASSETS = "/Users/sebastiandysart/Desktop/ascension-engine/assets"
os.makedirs(TMP, exist_ok=True)

# ── WARM CINEMATIC GRADE (matches gold reference aesthetic) ───────
# Gold refs: warm amber, lifted shadows, reduced blue, controlled contrast
# NOT the aggressive dark_cinema grade — that was crushing blacks and blowing highlights
WARM_CINEMATIC = (
    "eq=brightness=-0.04:contrast=1.18:saturation=0.82,"
    "colorbalance=rs=0.08:gs=0.02:bs=-0.12:"   # warm: lift reds, cut blues
    "rm=0.04:gm=0.01:bm=-0.06:"                 # midtone warmth
    "rh=-0.02:gh=0.0:bh=-0.04"                  # slight highlight control
)

CROP_16_9  = "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos"
SCALE_ONLY = "scale=1080:1920:flags=lanczos"

def score_frame(png_bytes):
    """
    Score a frame for clip quality.
    Good frame: face region (center-upper) is 40-180 brightness range.
    Bad: too dark (silhouette) or too bright (blown out) or flat/no contrast.
    Returns 0-100 score.
    """
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        w, h = img.size
        # Focus on face region: center-upper third of frame
        face_box = (w//4, h//6, 3*w//4, h//2)
        face_region = img.crop(face_box)
        arr = np.array(face_region, dtype=float)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]

        mean_brightness = gray.mean()
        std_brightness  = gray.std()   # contrast proxy

        # Ideal: mean brightness 60-160 (not silhouette, not blown)
        # Ideal: std > 25 (some contrast — interesting, not flat)
        brightness_score = 100 - abs(mean_brightness - 110) * 0.8
        contrast_score   = min(std_brightness * 1.5, 50)
        total = max(0, brightness_score + contrast_score)
        return total, mean_brightness, std_brightness
    except:
        return 0, 0, 0

def find_best_window(video_path, clip_duration=2.5, sample_rate=0.5):
    """
    Sample frames every `sample_rate` seconds, score each,
    find the best `clip_duration`-second window.
    """
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path
    ], capture_output=True, text=True)
    try:
        total_dur = float(probe.stdout.strip())
    except:
        return 2.0, 50.0

    # Sample frames
    scores = []
    t = 1.0
    while t < total_dur - clip_duration:
        r = subprocess.run([
            "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"
        ], capture_output=True)
        if r.returncode == 0 and r.stdout:
            score, brightness, contrast = score_frame(r.stdout)
            scores.append((t, score, brightness, contrast))
        t += sample_rate

    if not scores:
        return 2.0, 50.0

    # Find best window: max average score over clip_duration window
    best_start = scores[0][0]
    best_score = 0
    window_frames = int(clip_duration / sample_rate)

    for i in range(len(scores) - window_frames):
        window_score = sum(s[1] for s in scores[i:i+window_frames]) / window_frames
        if window_score > best_score:
            best_score = window_score
            best_start = scores[i][0]

    print(f"    Best window: {best_start:.1f}s (score {best_score:.0f}) | "
          f"brightness={scores[int((best_start)/sample_rate)][2]:.0f} "
          f"contrast={scores[int((best_start)/sample_rate)][3]:.0f}")
    return best_start, best_score

# ── BEAT DETECTION ────────────────────────────────────────────────
AUDIO_SRC = f"{GOLD}/bp_masterx_7589676463071268118.mp4"
print("▶ Detecting beats...")
wav = f"{TMP}/audio.wav"
subprocess.run(["ffmpeg","-y","-i",AUDIO_SRC,"-ac","1","-ar","22050","-vn",wav], capture_output=True)
import librosa
y, sr = librosa.load(wav, sr=22050)
tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
bpm = float(tempo)
beat_times = [t for t in beat_times if t <= 22.0]
# Cuts every 4 beats = cinematic pacing (~2.6s per clip)
cut_points = [beat_times[i] for i in range(0, len(beat_times)-1, 4)][:7]
clip_dur = cut_points[1] - cut_points[0]
print(f"  BPM: {bpm:.1f} | clip duration: {clip_dur:.2f}s | {len(cut_points)} cuts")

# ── FIND BEST WINDOWS IN EACH SOURCE ─────────────────────────────
sources = [
    f"{GOLD}/bp_masterx_7589676463071268118.mp4",
    f"{GOLD}/bp4ever.ae_7542208986699844869.mp4",
    f"{GOLD}/morphedmanlet_7540813336661740855.mp4",
    f"{GOLD}/bp.pilled_7580194747986283790.mp4",
    f"{GOLD}/roge.editz_7588525512478166294.mp4",
    f"{GOLD}/black.pill.city_7600958206642326806.mp4",
]

source_names = ["bp_masterx","bp4ever","morphedmanlet","bp.pilled","roge.editz","black.pill.city"]

print("\n▶ Scoring frames to find best clip windows...")
best_windows = []
for i, (src, name) in enumerate(zip(sources, source_names)):
    print(f"  Scanning {name}...")
    start, score = find_best_window(src, clip_duration=clip_dur, sample_rate=0.5)
    best_windows.append((src, start, name))

# ── RENDER SEGMENTS ───────────────────────────────────────────────
print("\n▶ Rendering beat-timed segments with warm cinematic grade...")
segments = []
for i in range(len(cut_points) - 1):
    dur = cut_points[i+1] - cut_points[i]
    src, src_start, name = best_windows[i % len(best_windows)]

    # Detect if source is already 9:16
    probe = subprocess.run([
        "ffprobe","-v","quiet","-select_streams","v:0",
        "-show_entries","stream=width,height","-of","csv=p=0",src
    ], capture_output=True, text=True)
    try:
        w, h = map(int, probe.stdout.strip().split(','))
        is_916 = abs((w/h)-(9/16)) < 0.05
    except:
        is_916 = False

    crop = SCALE_ONLY if is_916 else CROP_16_9
    # Gentle push-in zoom — cinematic, not aggressive
    zoom = "zoompan=z='min(zoom+0.0003,1.025)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"
    vf   = f"{crop},{WARM_CINEMATIC},{zoom}"

    seg = f"{TMP}/seg_{i:02d}.mp4"
    r = subprocess.run([
        "ffmpeg","-y","-ss",str(src_start),"-t",str(dur+0.05),"-i",src,
        "-vf",vf,"-c:v","libx264","-preset","fast","-crf","16","-an",seg
    ], capture_output=True)

    ok = "✅" if r.returncode==0 else "❌"
    kb = os.path.getsize(seg)//1024 if r.returncode==0 else 0
    print(f"  {ok} seg {i+1}: {name} @{src_start:.1f}s | {dur:.2f}s | {kb}KB")
    if r.returncode == 0:
        segments.append(seg)

# ── CONCAT + WATERMARK + AUDIO ────────────────────────────────────
print("\n▶ Assembling final edit...")
with open(f"{TMP}/list.txt","w") as f:
    for s in segments: f.write(f"file '{s}'\n")
concat = f"{TMP}/concat.mp4"
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",f"{TMP}/list.txt","-c","copy",concat], capture_output=True)

total = cut_points[len(segments)]
final = f"{OUT}/smart_edit_v1.mp4"

r = subprocess.run([
    "ffmpeg","-y",
    "-i",concat,
    "-i",f"{ASSETS}/watermark.png",
    "-ss","0","-t",str(total),"-i",AUDIO_SRC,
    "-filter_complex","[0:v][1:v]overlay=0:0[vout]",
    "-map","[vout]","-map","2:a",
    "-c:v","libx264","-preset","fast","-crf","16",
    "-c:a","aac","-b:a","192k","-shortest",final
], capture_output=True)

if r.returncode == 0:
    mb = os.path.getsize(final)/(1024*1024)
    print(f"\n✅ Smart edit done: {final}")
    print(f"   {mb:.1f}MB | {total:.1f}s | {len(segments)} clips | {bpm:.0f}BPM")
    print(f"\n   Cut schedule:")
    for i in range(len(segments)):
        src, start, name = best_windows[i % len(best_windows)]
        print(f"   [{cut_points[i]:.2f}→{cut_points[i+1]:.2f}s] {name} @{start:.1f}s (scored best window)")
else:
    print(f"❌ {r.stderr[-400:]}")

# ── EXTRACT REVIEW FRAMES ─────────────────────────────────────────
print("\n▶ Extracting review frames for QC...")
review_dir = "/Users/sebastiandysart/Desktop/ascension-engine/review"
subprocess.run([
    "ffmpeg","-y","-i",final,"-vf","fps=1",f"{review_dir}/smart_%02d.png"
], capture_output=True)
print("   Frames saved to review/smart_*.png")
