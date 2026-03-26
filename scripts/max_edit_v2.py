#!/usr/bin/env python3
"""
MAX TIER EDIT v2
Key fixes from v1 review:
1. ONLY use clean sources — skip morphedmanlet/roge.editz/black.pill.city (baked-in captions)
2. Subtitle detection: skip windows where bottom 25% has heavy white pixel density
3. Sample MULTIPLE windows per source → find best non-overlapping picks
4. Minimal color grade (gold refs are already cinematic; don't over-process)
5. MOGGED pill capsule graphic matching gold reference style
"""
import subprocess, os, io, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GOLD   = "/Users/sebastiandysart/Desktop/ascension-engine/input/gold"
OUT    = "/Users/sebastiandysart/Desktop/ascension-engine/out"
TMP    = "/Users/sebastiandysart/Desktop/ascension-engine/tmp_v2"
ASSETS = "/Users/sebastiandysart/Desktop/ascension-engine/assets"
REVIEW = "/Users/sebastiandysart/Desktop/ascension-engine/review"
os.makedirs(TMP, exist_ok=True)
os.makedirs(REVIEW, exist_ok=True)

# ── GRADE: Light touch — gold refs are already cinematic ─────────
# Minimal: slight warmth lift, subtle contrast. No crushing blacks.
LIGHT_GRADE = (
    "eq=brightness=-0.02:contrast=1.10:saturation=0.90,"
    "colorbalance=rs=0.06:gs=0.01:bs=-0.09:"
    "rm=0.03:gm=0.0:bm=-0.04:"
    "rh=0.0:gh=0.0:bh=-0.02"
)
CROP_VERTICAL = "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos"
SCALE_ONLY    = "scale=1080:1920:flags=lanczos"

def get_duration(path):
    r = subprocess.run([
        "ffprobe","-v","quiet","-show_entries","format=duration",
        "-of","csv=p=0",path
    ], capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except: return 0.0

def sample_frame(video_path, timestamp):
    """Extract single PNG frame at timestamp, return bytes or None."""
    r = subprocess.run([
        "ffmpeg","-y","-ss",str(timestamp),"-i",video_path,
        "-frames:v","1","-f","image2pipe","-vcodec","png","pipe:1"
    ], capture_output=True)
    return r.stdout if r.returncode == 0 and r.stdout else None

def has_subtitle_overlay(png_bytes, threshold=0.12):
    """
    Detect baked-in subtitle/caption overlays.
    Checks if bottom 25% of frame has >threshold fraction of near-white pixels.
    Near-white = all channels > 200 (typical for TikTok caption text).
    Returns True if subtitles detected (bad window), False if clean.
    """
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        w, h = img.size
        # Bottom 25% — where subtitles/captions live
        bottom = img.crop((0, int(h * 0.75), w, h))
        arr = np.array(bottom, dtype=float)
        # Near-white: all channels > 200
        white_mask = (arr[:,:,0] > 200) & (arr[:,:,1] > 200) & (arr[:,:,2] > 200)
        white_frac = white_mask.mean()
        return white_frac > threshold
    except:
        return False

def score_frame_quality(png_bytes):
    """
    Score face region quality.
    Returns (score, brightness, contrast).
    Best: mean brightness 70-160, contrast std > 30.
    """
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        w, h = img.size
        # Face region: center-upper third
        face = img.crop((w//4, h//8, 3*w//4, h//2))
        arr = np.array(face, dtype=float)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        mean_b = gray.mean()
        std_b  = gray.std()
        brightness_score = 100 - abs(mean_b - 115) * 0.7
        contrast_score   = min(std_b * 1.2, 40)
        return max(0, brightness_score + contrast_score), mean_b, std_b
    except:
        return 0, 0, 0

def find_clean_windows(video_path, clip_dur=2.5, sample_step=0.5, n_picks=3):
    """
    Find the top N non-overlapping clean clip windows in a video.
    'Clean' = no subtitle overlay detected + good face brightness.
    Returns list of (start_time, score) tuples.
    """
    total = get_duration(video_path)
    if total < clip_dur + 1.0:
        return []

    scored = []  # (timestamp, quality_score)
    t = 1.0
    while t < total - clip_dur:
        png = sample_frame(video_path, t)
        if png:
            if has_subtitle_overlay(png):
                scored.append((t, 0.0))  # penalize subtitle windows
            else:
                q, b, c = score_frame_quality(png)
                scored.append((t, q))
        t += sample_step

    if not scored:
        return []

    # Sliding window average
    w_frames = int(clip_dur / sample_step)
    window_scores = []
    for i in range(len(scored) - w_frames):
        avg = sum(s[1] for s in scored[i:i+w_frames]) / w_frames
        window_scores.append((scored[i][0], avg))

    # Pick top N non-overlapping windows
    window_scores.sort(key=lambda x: -x[1])
    picks = []
    for start, score in window_scores:
        if score < 10:  # too many subtitle frames in window
            continue
        # Check no overlap with already-picked windows
        overlap = any(abs(start - p[0]) < clip_dur for p in picks)
        if not overlap:
            picks.append((start, score))
        if len(picks) >= n_picks:
            break

    return picks

# ── SOURCES: CLEAN ONLY ───────────────────────────────────────────
# bp_masterx = clean selfie-style HVM (CONFIRMED GOOD)
# Others: probe and use if they pass subtitle filter
# Excluded: morphedmanlet (baked captions), roge.editz (interview subtitles),
#           black.pill.city (TikTok comment overlays)
CLEAN_SOURCES = [
    (f"{GOLD}/bp_masterx_7589676463071268118.mp4",   "bp_masterx",  4),
    (f"{GOLD}/haifluke_7618372975192018189.mp4",      "haifluke",    2),
    (f"{GOLD}/saffyro__7608533082261474582.mp4",      "saffyro",     2),
    (f"{GOLD}/unc_iiris_7537476213275757846.mp4",     "unc_iiris",   2),
    (f"{GOLD}/bp.editz04_7596127531955244318.mp4",    "bp.editz04",  2),
]

# ── BEAT DETECTION ────────────────────────────────────────────────
AUDIO_SRC = f"{GOLD}/bp_masterx_7589676463071268118.mp4"
print("▶ Detecting beats from bp_masterx audio...")
wav = f"{TMP}/audio.wav"
subprocess.run(["ffmpeg","-y","-i",AUDIO_SRC,"-ac","1","-ar","22050","-vn",wav],
               capture_output=True)
import librosa
y, sr = librosa.load(wav, sr=22050)
tempo, beat_times_raw = librosa.beat.beat_track(y=y, sr=sr, units='time')
bpm = float(tempo)

# Filter beat times to first 22 seconds
beat_times = [float(t) for t in beat_times_raw if float(t) <= 22.0]

# Cuts every 2 beats = energetic pacing (~1.3s per clip at 92BPM)
cut_every = 2
cut_points = [beat_times[i] for i in range(0, len(beat_times)-1, cut_every)][:10]
if len(cut_points) < 3:
    cut_points = [0.0, 1.3, 2.6, 3.9, 5.2, 6.5, 7.8, 9.1, 10.4, 11.7]

clip_dur = cut_points[1] - cut_points[0] if len(cut_points) > 1 else 1.3
print(f"  BPM: {bpm:.1f} | clip_dur: {clip_dur:.2f}s | {len(cut_points)-1} cuts planned")

# ── SCAN SOURCES FOR CLEAN WINDOWS ───────────────────────────────
print("\n▶ Scanning sources for clean clip windows (subtitle filter ON)...")
all_clips = []  # (src_path, start_time, score, name)

for src_path, name, max_picks in CLEAN_SOURCES:
    if not os.path.exists(src_path):
        print(f"  ⚠ {name}: not found, skipping")
        continue
    print(f"  Scanning {name}...", end=" ", flush=True)
    wins = find_clean_windows(src_path, clip_dur=clip_dur, sample_step=0.5, n_picks=max_picks)
    print(f"{len(wins)} clean windows found")
    for start, score in wins:
        all_clips.append((src_path, start, score, name))
        print(f"      @{start:.1f}s score={score:.0f}")

if not all_clips:
    print("❌ No clean clips found! Falling back to bp_masterx no-filter mode...")
    src = f"{GOLD}/bp_masterx_7589676463071268118.mp4"
    dur = get_duration(src)
    step = dur / 5
    all_clips = [(src, step*i, 50, "bp_masterx") for i in range(5)]

# Sort by score (best first), then interleave sources for variety
all_clips.sort(key=lambda x: -x[2])
print(f"\n  Total clean clips available: {len(all_clips)}")

# ── ASSIGN CLIPS TO CUT SLOTS ─────────────────────────────────────
# Interleave sources for variety (don't use same source back-to-back)
def assign_clips(all_clips, n_slots):
    """Assign clips to slots, avoiding same source consecutively."""
    used = set()
    result = []
    remaining = list(all_clips)

    for slot in range(n_slots):
        last_name = result[-1][3] if result else None
        # Try to find a different source than last
        chosen = None
        for clip in remaining:
            if clip[0] not in used and clip[3] != last_name:
                chosen = clip
                break
        if not chosen:
            for clip in remaining:
                if clip[0] not in used:
                    chosen = clip
                    break
        if not chosen and remaining:
            chosen = remaining[0]
        if chosen:
            result.append(chosen)
            used.add(chosen[0] + str(chosen[1]))
            remaining = [c for c in remaining if not (c[0]==chosen[0] and c[1]==chosen[1])]
        else:
            # Reuse from beginning (cycle)
            result.append(all_clips[slot % len(all_clips)])

    return result

n_cuts = len(cut_points) - 1
assigned = assign_clips(all_clips, n_cuts)
print(f"\n▶ Cut plan ({n_cuts} slots):")
for i, (src, start, score, name) in enumerate(assigned):
    dur = cut_points[i+1] - cut_points[i]
    print(f"  [{cut_points[i]:.2f}→{cut_points[i+1]:.2f}s] {name} @{start:.1f}s (score={score:.0f})")

# ── RENDER SEGMENTS ───────────────────────────────────────────────
print("\n▶ Rendering segments...")
segments = []
for i, (src, src_start, score, name) in enumerate(assigned):
    dur = cut_points[i+1] - cut_points[i]

    # Detect aspect ratio
    probe = subprocess.run([
        "ffprobe","-v","quiet","-select_streams","v:0",
        "-show_entries","stream=width,height","-of","csv=p=0",src
    ], capture_output=True, text=True)
    try:
        w, h = map(int, probe.stdout.strip().split(','))
        is_916 = abs((w/h) - (9/16)) < 0.06
    except:
        is_916 = False

    crop_filt = SCALE_ONLY if is_916 else CROP_VERTICAL
    # Subtle zoom — adds life without being distracting
    zoom = "zoompan=z='min(zoom+0.0002,1.015)':d=1:x='iw/2-(iw/zoom/2)':y='ih/4-(ih/zoom/4)':s=1080x1920"
    vf = f"{crop_filt},{LIGHT_GRADE},{zoom}"

    seg = f"{TMP}/seg_{i:02d}.mp4"
    r = subprocess.run([
        "ffmpeg","-y","-ss",str(src_start),"-t",str(dur+0.1),"-i",src,
        "-vf",vf,"-c:v","libx264","-preset","fast","-crf","15","-an",seg
    ], capture_output=True)

    ok = "✅" if r.returncode == 0 else "❌"
    kb = os.path.getsize(seg)//1024 if r.returncode == 0 else 0
    print(f"  {ok} seg {i+1}: {name} @{src_start:.1f}s | {dur:.2f}s | {kb}KB")
    if r.returncode == 0:
        segments.append(seg)
    else:
        print(f"     stderr: {r.stderr[-200:].decode('utf-8','replace')}")

# ── GENERATE MOGGED PILL OVERLAY ──────────────────────────────────
# Black pill capsule with "MOGGED" inside (matching gold reference style)
def make_pill_overlay(text, width=1080, height=1920, out_path=None):
    img = Image.new("RGBA", (width, height), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    # Pill dimensions
    pw, ph = 820, 130
    px = (width - pw) // 2
    py = int(height * 0.58)

    # Draw pill background: dark/black with slight gloss
    r = ph // 2
    # Fill rounded rect manually
    draw.ellipse([px, py, px+ph, py+ph], fill=(20,20,20,230))
    draw.ellipse([px+pw-ph, py, px+pw, py+ph], fill=(20,20,20,230))
    draw.rectangle([px+r, py, px+pw-r, py+ph], fill=(20,20,20,230))

    # Gloss highlight (top strip)
    draw.rectangle([px+r, py+4, px+pw-r, py+ph//3], fill=(60,60,60,120))
    draw.ellipse([px+4, py+4, px+ph-4, py+ph//3+4], fill=(60,60,60,120))

    # Pill divider line (center vertical)
    mid_x = width // 2
    draw.rectangle([mid_x-2, py+8, mid_x+2, py+ph-8], fill=(40,40,40,200))

    # Text inside pill
    try:
        font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 82)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 82)
        except:
            font = ImageFont.load_default()

    # Get text bbox for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = py + (ph - th) // 2 - 4

    # Bold white text
    draw.text((tx, ty), text, font=font, fill=(255,255,255,255))

    if out_path:
        img.save(out_path)
    return img

# Generate overlays for MOGGED reveal sequence
print("\n▶ Generating MOGGED pill overlay assets...")
pill_path = f"{ASSETS}/pill_mogged.png"
make_pill_overlay("MOGGED", out_path=pill_path)
print(f"  ✅ {pill_path}")

# Also make partial reveals
pill_m_path  = f"{ASSETS}/pill_M.png"
pill_mo_path = f"{ASSETS}/pill_MO.png"
pill_mog_path = f"{ASSETS}/pill_MOG.png"
make_pill_overlay("M",    out_path=pill_m_path)
make_pill_overlay("MO",   out_path=pill_mo_path)
make_pill_overlay("MOG",  out_path=pill_mog_path)
print("  ✅ Partial reveal pills: M / MO / MOG / MOGGED")

# ── CONCAT + WATERMARK + MOGGED REVEAL + AUDIO ────────────────────
print("\n▶ Assembling final edit...")
with open(f"{TMP}/list.txt","w") as f:
    for s in segments:
        f.write(f"file '{s}'\n")

concat = f"{TMP}/concat.mp4"
subprocess.run([
    "ffmpeg","-y","-f","concat","-safe","0",
    "-i",f"{TMP}/list.txt","-c","copy",concat
], capture_output=True)

total_dur = cut_points[len(segments)] if len(segments) < len(cut_points) else cut_points[-1]

# MOGGED reveal timing — appears around halfway through, letter by letter
# Timed to beat: reveal M, MO, MOG, then MOGGED over 4 consecutive beats
mid_beat = len(cut_points) // 2
t_m   = cut_points[mid_beat]       if mid_beat < len(cut_points) else total_dur * 0.45
t_mo  = cut_points[mid_beat+1]     if mid_beat+1 < len(cut_points) else t_m + clip_dur
t_mog = cut_points[mid_beat+2]     if mid_beat+2 < len(cut_points) else t_mo + clip_dur
t_end = cut_points[mid_beat+3]     if mid_beat+3 < len(cut_points) else total_dur

print(f"  MOGGED reveal: M@{t_m:.2f}s → MO@{t_mo:.2f}s → MOG@{t_mog:.2f}s → MOGGED@{t_end:.2f}s")

wm   = f"{ASSETS}/watermark.png"

# Build filter_complex: watermark always on + timed MOGGED pill reveals
filter_complex = (
    f"[0:v][1:v]overlay=0:0[wm];"
    f"[wm][2:v]overlay=0:0:enable='between(t,{t_m:.2f},{t_mo:.2f})'[v1];"
    f"[v1][3:v]overlay=0:0:enable='between(t,{t_mo:.2f},{t_mog:.2f})'[v2];"
    f"[v2][4:v]overlay=0:0:enable='between(t,{t_mog:.2f},{t_end:.2f})'[v3];"
    f"[v3][5:v]overlay=0:0:enable='gte(t,{t_end:.2f})'[vout]"
)

final = f"{OUT}/max_edit_v2.mp4"
r = subprocess.run([
    "ffmpeg","-y",
    "-i",concat,
    "-i",wm,
    "-i",pill_m_path,
    "-i",pill_mo_path,
    "-i",pill_mog_path,
    "-i",pill_path,
    "-ss","0","-t",str(total_dur),"-i",AUDIO_SRC,
    "-filter_complex",filter_complex,
    "-map","[vout]","-map","6:a",
    "-c:v","libx264","-preset","fast","-crf","15",
    "-c:a","aac","-b:a","192k","-shortest",final
], capture_output=True)

if r.returncode == 0:
    mb = os.path.getsize(final) / (1024*1024)
    print(f"\n✅ MAX EDIT v2 DONE: {final}")
    print(f"   {mb:.1f}MB | {total_dur:.1f}s | {len(segments)} clips | {bpm:.0f}BPM")
else:
    err = r.stderr[-600:].decode('utf-8','replace')
    print(f"\n❌ Final render failed:\n{err}")
    # Fallback: simpler overlay (just watermark, no pill)
    print("   Trying fallback (watermark only)...")
    r2 = subprocess.run([
        "ffmpeg","-y","-i",concat,"-i",wm,
        "-ss","0","-t",str(total_dur),"-i",AUDIO_SRC,
        "-filter_complex","[0:v][1:v]overlay=0:0[vout]",
        "-map","[vout]","-map","2:a",
        "-c:v","libx264","-preset","fast","-crf","15",
        "-c:a","aac","-b:a","192k","-shortest",final
    ], capture_output=True)
    if r2.returncode == 0:
        mb = os.path.getsize(final)/(1024*1024)
        print(f"   ✅ Fallback succeeded: {mb:.1f}MB")
    else:
        print(f"   ❌ Fallback also failed: {r2.stderr[-300:].decode('utf-8','replace')}")

# ── EXTRACT REVIEW FRAMES ─────────────────────────────────────────
if os.path.exists(final) and os.path.getsize(final) > 10000:
    print("\n▶ Extracting review frames...")
    # Remove old v2 review frames
    for f in os.listdir(REVIEW):
        if f.startswith("v2_"):
            os.remove(os.path.join(REVIEW, f))

    subprocess.run([
        "ffmpeg","-y","-i",final,"-vf","fps=1",
        f"{REVIEW}/v2_%02d.png"
    ], capture_output=True)

    frames = sorted([f for f in os.listdir(REVIEW) if f.startswith("v2_")])
    print(f"   Saved {len(frames)} frames to review/v2_*.png")
    print("\n▶ QC: Checking for subtitle leakage in output frames...")
    issues = 0
    for fname in frames:
        with open(f"{REVIEW}/{fname}","rb") as fh:
            png = fh.read()
        if has_subtitle_overlay(png, threshold=0.08):
            print(f"   ⚠ {fname}: subtitle/caption detected (check this frame)")
            issues += 1
    if issues == 0:
        print(f"   ✅ All {len(frames)} frames clean — no subtitle leakage detected")
    else:
        print(f"   ⚠ {issues} frames have potential subtitle issues")

print("\n" + "="*50)
print("MAX EDIT v2 COMPLETE")
print("="*50)
print(f"Output: {final}")
print("Review frames: review/v2_*.png")
print("\nSource breakdown:")
for i, (src, start, score, name) in enumerate(assigned[:len(segments)]):
    dur = cut_points[i+1] - cut_points[i] if i+1 < len(cut_points) else 0
    print(f"  Clip {i+1}: {name} @{start:.1f}s ({dur:.2f}s, clean score={score:.0f})")
