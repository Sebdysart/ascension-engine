#!/usr/bin/env python3
"""
Ascension Engine v2.1 — Anti-Slop Comparison Gate
──────────────────────────────────────────────────────────────────────────────
Compares a generated edit against gold reference videos on multiple axes:
  1. SSIM (structural similarity) on sampled frames
  2. Color histogram match (RGB channel correlation)
  3. Cut pacing fidelity (scene rhythm vs reference)
  4. Beat alignment accuracy (cut-on-beat scoring)

Usage:
  python3 data/compare.py --generated out/gen_GlowUp.mp4 --reference input/gold/vid_007.mp4
  python3 data/compare.py --generated out/gen_GlowUp.mp4 --gold-dir input/gold/ --top 3
  python3 data/compare.py --batch out/last-batch.json --gold-dir input/gold/
  python3 data/compare.py --generated out/gen.mp4 --reference ref.mp4 --pass-threshold 0.85
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare")

ROOT = Path(__file__).resolve().parent.parent

# ── Optional dependencies ────────────────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from skimage.metrics import structural_similarity as ssim
    from PIL import Image
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: Path, tmpdir: Path, fps: float = 2.0) -> list[Path]:
    """Extract frames from a video at a given fps rate."""
    out_pattern = tmpdir / f"{video_path.stem}_frame_%04d.png"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(out_pattern),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return sorted(tmpdir.glob(f"{video_path.stem}_frame_*.png"))


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


# ── SSIM comparison ──────────────────────────────────────────────────────────

def compute_ssim_score(gen_frames: list[Path], ref_frames: list[Path]) -> dict:
    """Compute average SSIM between paired frames (resized to match)."""
    if not HAS_SSIM or not HAS_NUMPY:
        log.warning("scikit-image or numpy not installed — SSIM skipped.")
        return {"ssim_mean": 0.0, "ssim_min": 0.0, "ssim_max": 0.0, "ssim_samples": 0, "available": False}

    pair_count = min(len(gen_frames), len(ref_frames))
    if pair_count == 0:
        return {"ssim_mean": 0.0, "ssim_min": 0.0, "ssim_max": 0.0, "ssim_samples": 0, "available": True}

    # Sample evenly if many frames
    sample_indices = list(range(0, pair_count, max(1, pair_count // 20)))[:20]
    scores = []

    for idx in sample_indices:
        try:
            gen_img = np.array(Image.open(gen_frames[idx]).convert("RGB").resize((540, 960)))
            ref_img = np.array(Image.open(ref_frames[idx]).convert("RGB").resize((540, 960)))

            score = ssim(gen_img, ref_img, channel_axis=2, data_range=255)
            scores.append(float(score))
        except Exception as e:
            log.debug("SSIM frame %d error: %s", idx, e)

    if not scores:
        return {"ssim_mean": 0.0, "ssim_min": 0.0, "ssim_max": 0.0, "ssim_samples": 0, "available": True}

    return {
        "ssim_mean": round(float(np.mean(scores)), 4),
        "ssim_min": round(float(np.min(scores)), 4),
        "ssim_max": round(float(np.max(scores)), 4),
        "ssim_samples": len(scores),
        "available": True,
    }


# ── Color histogram comparison ───────────────────────────────────────────────

def compute_color_score(gen_frames: list[Path], ref_frames: list[Path]) -> dict:
    """Compare average RGB histograms using correlation."""
    if not HAS_NUMPY:
        return {"color_correlation": 0.0, "available": False}

    def avg_histogram(frames: list[Path]) -> np.ndarray:
        sample = frames[::max(1, len(frames) // 10)][:10]
        hist_sum = np.zeros(256 * 3)
        count = 0
        for f in sample:
            try:
                img = np.array(Image.open(f).convert("RGB").resize((64, 64)))
                for ch in range(3):
                    h, _ = np.histogram(img[:, :, ch], bins=256, range=(0, 256))
                    hist_sum[ch * 256:(ch + 1) * 256] += h.astype(float)
                count += 1
            except Exception:
                pass
        return hist_sum / max(count, 1)

    gen_hist = avg_histogram(gen_frames)
    ref_hist = avg_histogram(ref_frames)

    if np.linalg.norm(gen_hist) == 0 or np.linalg.norm(ref_hist) == 0:
        return {"color_correlation": 0.0, "available": True}

    correlation = float(np.dot(gen_hist, ref_hist) / (np.linalg.norm(gen_hist) * np.linalg.norm(ref_hist)))

    return {
        "color_correlation": round(correlation, 4),
        "available": True,
    }


# ── Cut pacing analysis ──────────────────────────────────────────────────────

def detect_scene_timestamps(video_path: Path) -> list[float]:
    """Detect scene boundaries using FFmpeg scene filter."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0",
        "-f", "lavfi",
        f"movie={video_path},select=gt(scene\\,0.3)",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        timestamps = []
        for line in result.stdout.strip().splitlines():
            try:
                timestamps.append(float(line.strip()))
            except ValueError:
                pass
        return timestamps
    except Exception:
        return []


def compute_pacing_score(gen_path: Path, ref_path: Path) -> dict:
    """Compare cut pacing rhythm between generated and reference videos."""
    gen_scenes = detect_scene_timestamps(gen_path)
    ref_scenes = detect_scene_timestamps(ref_path)

    if not gen_scenes or not ref_scenes:
        gen_dur = get_video_duration(gen_path)
        ref_dur = get_video_duration(ref_path)

        # Fallback: compare total durations
        dur_ratio = min(gen_dur, ref_dur) / max(gen_dur, ref_dur) if max(gen_dur, ref_dur) > 0 else 0
        return {
            "pacing_fidelity": round(dur_ratio, 4),
            "gen_cuts": len(gen_scenes),
            "ref_cuts": len(ref_scenes),
            "avg_cut_gen": 0.0,
            "avg_cut_ref": 0.0,
            "method": "duration_fallback",
        }

    def cut_lengths(timestamps: list[float], total_dur: float) -> list[float]:
        if not timestamps:
            return [total_dur]
        boundaries = [0.0] + timestamps + [total_dur]
        return [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1) if boundaries[i + 1] - boundaries[i] > 0.1]

    gen_dur = get_video_duration(gen_path)
    ref_dur = get_video_duration(ref_path)
    gen_cuts = cut_lengths(gen_scenes, gen_dur)
    ref_cuts = cut_lengths(ref_scenes, ref_dur)

    avg_gen = sum(gen_cuts) / len(gen_cuts) if gen_cuts else 0
    avg_ref = sum(ref_cuts) / len(ref_cuts) if ref_cuts else 0

    # Pacing fidelity: how close avg cut lengths are (1.0 = identical)
    if max(avg_gen, avg_ref) > 0:
        fidelity = 1.0 - abs(avg_gen - avg_ref) / max(avg_gen, avg_ref)
    else:
        fidelity = 0.0

    # Cut count similarity
    max_cuts = max(len(gen_cuts), len(ref_cuts))
    count_similarity = 1.0 - abs(len(gen_cuts) - len(ref_cuts)) / max(max_cuts, 1)

    combined = (fidelity * 0.6 + count_similarity * 0.4)

    return {
        "pacing_fidelity": round(combined, 4),
        "gen_cuts": len(gen_cuts),
        "ref_cuts": len(ref_cuts),
        "avg_cut_gen": round(avg_gen, 3),
        "avg_cut_ref": round(avg_ref, 3),
        "method": "scene_detect",
    }


# ── Beat alignment score ─────────────────────────────────────────────────────

def compute_beat_alignment(gen_path: Path, ref_path: Path) -> dict:
    """Compare BPM and beat alignment between generated and reference audio."""
    if not HAS_LIBROSA or not HAS_NUMPY:
        return {"bpm_gen": 0, "bpm_ref": 0, "bpm_match": 0.0, "available": False}

    def analyze_beats(video_path: Path) -> tuple[float, list[float]]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path), "-vn",
                 "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", wav_path],
                capture_output=True, check=True,
            )
            y, sr = librosa.load(wav_path, sr=22050, mono=True)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
            return float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]), beat_times
        except Exception:
            return 0.0, []
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    gen_bpm, gen_beats = analyze_beats(gen_path)
    ref_bpm, ref_beats = analyze_beats(ref_path)

    # BPM similarity (1.0 = identical)
    if max(gen_bpm, ref_bpm) > 0:
        bpm_match = 1.0 - abs(gen_bpm - ref_bpm) / max(gen_bpm, ref_bpm)
    else:
        bpm_match = 0.0

    return {
        "bpm_gen": round(gen_bpm),
        "bpm_ref": round(ref_bpm),
        "bpm_match": round(bpm_match, 4),
        "beats_gen": len(gen_beats),
        "beats_ref": len(ref_beats),
        "available": True,
    }


# ── Composite DNA fidelity score ─────────────────────────────────────────────

def compute_dna_score(ssim_data: dict, color_data: dict, pacing_data: dict, beat_data: dict) -> dict:
    """
    Weighted composite score from all comparison axes.
    Uses a DNA fidelity approach rather than raw pixel SSIM.
    """
    weights = {
        "pacing": 0.35,   # cut rhythm is the strongest DNA signal
        "color": 0.25,    # color grade consistency
        "ssim": 0.20,     # structural frame similarity
        "beat": 0.20,     # audio/beat alignment
    }

    scores = {
        "ssim": ssim_data.get("ssim_mean", 0.0) if ssim_data.get("available", False) else 0.5,
        "color": color_data.get("color_correlation", 0.0) if color_data.get("available", False) else 0.5,
        "pacing": pacing_data.get("pacing_fidelity", 0.0),
        "beat": beat_data.get("bpm_match", 0.0) if beat_data.get("available", False) else 0.5,
    }

    # Adjust weights if some metrics are unavailable
    active_weights = {}
    for key, w in weights.items():
        if key == "ssim" and not ssim_data.get("available", False):
            continue
        if key == "beat" and not beat_data.get("available", False):
            continue
        if key == "color" and not color_data.get("available", False):
            continue
        active_weights[key] = w

    total_weight = sum(active_weights.values())
    if total_weight == 0:
        return {"composite_score": 0.0, "weights": weights, "component_scores": scores}

    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
    composite = sum(scores[k] * normalized_weights.get(k, 0) for k in scores)

    return {
        "composite_score": round(composite, 4),
        "weights_used": {k: round(v, 3) for k, v in normalized_weights.items()},
        "component_scores": scores,
    }


# ── Full comparison ──────────────────────────────────────────────────────────

def compare_videos(gen_path: Path, ref_path: Path, pass_threshold: float = 0.80) -> dict:
    """Run full comparison between a generated video and a reference."""
    log.info("Comparing: %s vs %s", gen_path.name, ref_path.name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Extract frames
        log.info("  Extracting frames...")
        gen_frames = extract_frames(gen_path, tmpdir_path, fps=2.0)
        ref_frames = extract_frames(ref_path, tmpdir_path, fps=2.0)
        log.info("  Frames: gen=%d, ref=%d", len(gen_frames), len(ref_frames))

        # SSIM
        log.info("  Computing SSIM...")
        ssim_data = compute_ssim_score(gen_frames, ref_frames)
        log.info("  SSIM: mean=%.4f", ssim_data.get("ssim_mean", 0))

        # Color
        log.info("  Computing color histogram...")
        color_data = compute_color_score(gen_frames, ref_frames)
        log.info("  Color correlation: %.4f", color_data.get("color_correlation", 0))

    # Pacing
    log.info("  Computing pacing score...")
    pacing_data = compute_pacing_score(gen_path, ref_path)
    log.info("  Pacing fidelity: %.4f (gen=%d cuts, ref=%d cuts)",
             pacing_data["pacing_fidelity"], pacing_data["gen_cuts"], pacing_data["ref_cuts"])

    # Beat alignment
    log.info("  Computing beat alignment...")
    beat_data = compute_beat_alignment(gen_path, ref_path)
    log.info("  BPM: gen=%d, ref=%d, match=%.4f",
             beat_data.get("bpm_gen", 0), beat_data.get("bpm_ref", 0), beat_data.get("bpm_match", 0))

    # Composite
    dna = compute_dna_score(ssim_data, color_data, pacing_data, beat_data)

    passed = dna["composite_score"] >= pass_threshold
    verdict = "PASS" if passed else "FAIL"

    result = {
        "generated": str(gen_path),
        "reference": str(ref_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pass_threshold": pass_threshold,
        "verdict": verdict,
        "passed": passed,
        "dna_score": dna,
        "ssim": ssim_data,
        "color": color_data,
        "pacing": pacing_data,
        "beat": beat_data,
    }

    return result


def compare_against_best(gen_path: Path, gold_dir: Path, top_n: int = 3, pass_threshold: float = 0.80) -> dict:
    """Compare a generated video against the top N gold references in a directory."""
    gold_files = sorted(gold_dir.glob("*.mp4"))
    if not gold_files:
        log.warning("No gold reference files found in %s", gold_dir)
        return {"error": "no_gold_files", "generated": str(gen_path)}

    # Use the first top_n (in a real system you'd rank by analytics)
    refs = gold_files[:top_n]
    log.info("Comparing against %d gold references", len(refs))

    results = []
    for ref in refs:
        result = compare_videos(gen_path, ref, pass_threshold)
        results.append(result)

    best = max(results, key=lambda r: r["dna_score"]["composite_score"])

    return {
        "generated": str(gen_path),
        "gold_dir": str(gold_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparisons": len(results),
        "best_match": {
            "reference": best["reference"],
            "dna_score": best["dna_score"]["composite_score"],
            "verdict": best["verdict"],
        },
        "all_results": results,
        "overall_verdict": "PASS" if best["passed"] else "FAIL",
    }


# ── Report printer ───────────────────────────────────────────────────────────

def print_report(result: dict) -> None:
    """Print a formatted comparison report."""
    sep = "═" * 60

    if "all_results" in result:
        # Multi-reference comparison
        print(f"\n{sep}")
        print(f"  ANTI-SLOP COMPARISON REPORT")
        print(sep)
        print(f"  Generated : {Path(result['generated']).name}")
        print(f"  Gold dir  : {result['gold_dir']}")
        print(f"  Compared  : {result['comparisons']} references")
        print(sep)

        for r in result["all_results"]:
            dna = r["dna_score"]
            icon = "✅" if r["passed"] else "❌"
            print(f"  {icon} vs {Path(r['reference']).name}")
            print(f"     DNA score: {dna['composite_score']:.4f}  |  Verdict: {r['verdict']}")
            for k, v in dna["component_scores"].items():
                print(f"       {k:8s}: {v:.4f}")

        best = result["best_match"]
        print(sep)
        overall_icon = "✅" if result["overall_verdict"] == "PASS" else "❌"
        print(f"  {overall_icon} OVERALL: {result['overall_verdict']}  (best: {best['dna_score']:.4f} vs {Path(best['reference']).name})")
        print(f"{sep}\n")
    else:
        # Single comparison
        dna = result["dna_score"]
        icon = "✅" if result["passed"] else "❌"

        print(f"\n{sep}")
        print(f"  ANTI-SLOP COMPARISON REPORT")
        print(sep)
        print(f"  Generated : {Path(result['generated']).name}")
        print(f"  Reference : {Path(result['reference']).name}")
        print(f"  Threshold : {result['pass_threshold']}")
        print(sep)
        print(f"  DNA FIDELITY SCORE: {dna['composite_score']:.4f}")
        print(f"  Components:")
        for k, v in dna["component_scores"].items():
            w = dna["weights_used"].get(k, 0)
            print(f"    {k:8s}: {v:.4f}  (weight: {w:.2f})")
        print(sep)
        print(f"  SSIM    : mean={result['ssim'].get('ssim_mean', 0):.4f}  min={result['ssim'].get('ssim_min', 0):.4f}  max={result['ssim'].get('ssim_max', 0):.4f}")
        print(f"  Color   : correlation={result['color'].get('color_correlation', 0):.4f}")
        print(f"  Pacing  : fidelity={result['pacing']['pacing_fidelity']:.4f}  (gen={result['pacing']['gen_cuts']} cuts, ref={result['pacing']['ref_cuts']} cuts)")
        if result["beat"].get("available"):
            print(f"  Beat    : gen={result['beat']['bpm_gen']} BPM, ref={result['beat']['bpm_ref']} BPM, match={result['beat']['bpm_match']:.4f}")
        print(sep)
        print(f"  {icon} VERDICT: {result['verdict']}")
        print(f"{sep}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ascension Engine v2.1 — Anti-Slop Comparison Gate",
    )
    parser.add_argument("--generated", "-g", required=True, help="Path to generated video")
    parser.add_argument("--reference", "-r", help="Path to single gold reference video")
    parser.add_argument("--gold-dir", help="Directory of gold reference videos")
    parser.add_argument("--top", type=int, default=3, help="Top N gold refs to compare against")
    parser.add_argument("--pass-threshold", type=float, default=0.80, help="Minimum DNA score to pass")
    parser.add_argument("--output-json", "-o", help="Write results to JSON file")
    args = parser.parse_args()

    gen_path = Path(args.generated)
    if not gen_path.exists():
        log.error("Generated video not found: %s", gen_path)
        sys.exit(1)

    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            log.error("Reference video not found: %s", ref_path)
            sys.exit(1)
        result = compare_videos(gen_path, ref_path, args.pass_threshold)
    elif args.gold_dir:
        gold_dir = Path(args.gold_dir)
        if not gold_dir.is_dir():
            log.error("Gold directory not found: %s", gold_dir)
            sys.exit(1)
        result = compare_against_best(gen_path, gold_dir, args.top, args.pass_threshold)
    else:
        log.error("Provide either --reference or --gold-dir")
        sys.exit(1)

    print_report(result)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(result, indent=2))
        log.info("Results written to %s", out_path)

    # Exit code based on verdict
    if result.get("passed", result.get("overall_verdict") == "PASS"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
