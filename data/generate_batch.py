#!/usr/bin/env python3
"""
Ascension Engine — generate_batch.py (v2 orchestrator)
Full impact-beat alignment pipeline replacing the old FFmpeg-only generator.

Pipeline per edit:
  1. score_clip_frames() in parallel across all available clips
  2. analyze_phonk_track() on the audio source
  3. assemble_edit() — EDL from scored clips + audio + templates
  4. Remotion render (npx remotion render) → temp.mp4
  5. FFmpeg post-process:
       lut3d + alimiter + loudnorm + libx264 crf18 → final.mp4

FFmpeg post-process (spec-exact):
  ffmpeg -i temp.mp4 -i mog_lut.cube \\
    -af "alimiter=limit=-1dB,loudnorm=I=-14:TP=-1.5:LRA=11" \\
    -vf "lut3d=file=mog_lut.cube:interp=tetrahedral" \\
    -c:v libx264 -crf 18 -preset slow \\
    -pix_fmt yuv420p -c:a aac -b:a 320k -movflags +faststart \\
    final.mp4

Usage:
    python3 data/generate_batch.py
    python3 data/generate_batch.py --clips library/clips/ --audio input/gold/bp_masterx.mp4
    python3 data/generate_batch.py --edl-only   # skip render, just output EDL JSON
    python3 data/generate_batch.py --dry-run    # print plan, no execution
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("generate_batch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

ROOT      = Path(__file__).resolve().parent.parent

# Ensure ROOT is on sys.path so sibling data.* modules resolve correctly
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GOLD_DIR  = ROOT / "input" / "gold"
CLIPS_DIR = ROOT / "library" / "clips"
SEQ_DIR   = ROOT / "library" / "sequence_templates"
LUTS_DIR  = ROOT / "luts"
OUT_DIR   = ROOT / "out"
TMP_DIR   = ROOT / "tmp_genbatch_v2"

OUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

# LUT filenames by color grade (from sequence templates)
LUT_FOR_GRADE: dict[str, str] = {
    "TealOrange":  "teal_orange",
    "WarmGold":    "warm_gold",
    "Desaturated": "neutral",
    "unknown":     "neutral",
}

# Audio sources ranked by phonk energy
AUDIO_PRIORITY = [
    GOLD_DIR / "bp_masterx_7589676463071268118.mp4",
    GOLD_DIR / "bp4ever.ae_7542208986699844869.mp4",
    GOLD_DIR / "black.pill.city_7600958206642326806.mp4",
]


# ── Step 1: parallel clip scoring ────────────────────────────────────────────

def score_clips_parallel(clip_paths: list[Path], max_workers: int = 4) -> list[dict]:
    """Score all clips in parallel using ThreadPoolExecutor."""
    from data.impact_scorer import score_clip_frames  # type: ignore

    log.info("Scoring %d clips (max_workers=%d) ...", len(clip_paths), max_workers)
    results: list[dict] = [{}] * len(clip_paths)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(score_clip_frames, str(p)): i
            for i, p in enumerate(clip_paths)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.error("Scoring failed for %s: %s", clip_paths[idx].name, e)
                results[idx] = {"video_path": str(clip_paths[idx]), "error": str(e)}

    return [r for r in results if r]


# ── Step 2: audio analysis ────────────────────────────────────────────────────

def pick_audio_source(preferred: Path | None = None) -> Path:
    if preferred and preferred.exists():
        return preferred
    for p in AUDIO_PRIORITY:
        if p.exists():
            return p
    raise FileNotFoundError("No audio source found in input/gold/")


# ── Step 3: EDL assembly ──────────────────────────────────────────────────────

def build_edl(
    clip_scores: list[dict],
    audio_path: Path,
    templates_dir: Path,
    target_duration: float = 15.0,
) -> dict:
    from data.audio_analyzer import analyze_phonk_track  # type: ignore
    from data.sequencer import assemble_edit, load_scored_clips, load_templates  # type: ignore

    audio_analysis = analyze_phonk_track(str(audio_path), target_duration)
    templates      = load_templates(str(templates_dir))
    scored         = load_scored_clips(clip_scores)
    edl            = assemble_edit(scored, audio_analysis, templates, target_duration)
    return edl


# ── Step 4: Remotion clip staging ────────────────────────────────────────────

def stage_clips_for_remotion(edl: dict) -> dict:
    """
    Copy EDL clips into remotion/public/clips/ and rewrite clip_path to
    /clips/clip_N.mp4 so Remotion's dev server can serve them via staticFile().

    Returns a modified copy of the EDL with updated clip_paths.
    Does not mutate the original.
    """
    import copy

    # Remotion resolves publicDir relative to where node_modules/remotion is found.
    # The worktree has no node_modules of its own, so Remotion uses the nearest
    # ancestor that does.  Walk up from ROOT to find that project root.
    remotion_root = ROOT
    while remotion_root != remotion_root.parent:
        if (remotion_root / "node_modules" / ".bin" / "remotion").exists():
            break
        remotion_root = remotion_root.parent

    public_clips = remotion_root / "public" / "clips"
    public_clips.mkdir(parents=True, exist_ok=True)

    edl_copy = copy.deepcopy(edl)
    for entry in edl_copy["edl"]:
        src = Path(entry["clip_path"])
        if not src.is_absolute():
            # Try ROOT-relative first, then CWD-relative
            for candidate in (ROOT / src, Path.cwd() / src):
                if candidate.exists():
                    src = candidate
                    break
        if not src.exists():
            log.warning("stage_clips: source not found: %s", src)
            continue
        dest_name = f"clip_{entry['slot']:03d}{src.suffix}"
        dest = public_clips / dest_name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        entry["clip_path"] = f"/clips/{dest_name}"

    log.info("Staged %d clips → %s", len(edl_copy["edl"]), public_clips)
    return edl_copy, public_clips


def write_edl_props(edl: dict, out_path: Path) -> None:
    """Write EDL as JSON props file for Remotion CLI injection."""
    out_path.write_text(json.dumps({"edl": edl, "audioPath": ""}, indent=2))


def remotion_render(
    edl: dict,
    audio_path: Path,
    temp_output: Path,
    public_dir: Path | None = None,
    composition_id: str = "BrutalBeatMontage",
    dry_run: bool = False,
) -> bool:
    """
    Render via Remotion CLI: npx remotion render <root> <id> <out> --props='...'

    Returns True on success (or dry_run). Falls back gracefully if Remotion
    is not installed or render fails (pipeline continues to FFmpeg fallback).
    """
    fps = edl.get("fps", 30)
    total_frames = sum(e["duration_frames"] for e in edl["edl"])

    props = json.dumps({
        "edl":       edl,
        "audioPath": str(audio_path),
    })

    cmd = [
        "npx", "remotion", "render",
        "remotion/index.tsx",
        composition_id,
        str(temp_output),
        f"--props={props}",
        f"--frames=0-{total_frames - 1}",
        "--codec=h264",
        "--crf=28",  # fast pass — FFmpeg will re-encode at crf18
        "--overwrite",
    ]
    log.info("Remotion render: %s → %s  (%d frames)", composition_id, temp_output.name, total_frames)

    if dry_run:
        log.info("[dry-run] would run: %s", " ".join(cmd[:6]) + " ...")
        return True

    # Run from the Remotion project root (where node_modules lives) so that
    # the dev server's publicDir resolves correctly.
    render_cwd = public_dir.parent.parent if public_dir is not None else ROOT
    r = subprocess.run(cmd, capture_output=False, cwd=str(render_cwd))
    if r.returncode != 0:
        log.warning("Remotion render failed (exit %d) — using FFmpeg fallback", r.returncode)
        return False
    return True


# ── Step 4 fallback: FFmpeg concat from EDL ───────────────────────────────────

def ffmpeg_concat_from_edl(edl: dict, audio_path: Path, temp_output: Path) -> bool:
    """
    Pure-FFmpeg fallback when Remotion is unavailable.
    Concatenates clips per EDL entry, then muxes audio.
    """
    fps = edl.get("fps", 30)
    work_dir = TMP_DIR / "concat"
    work_dir.mkdir(exist_ok=True)
    clip_files: list[Path] = []

    # Color grade FFmpeg filter (from last used template grade)
    GRADE_FILTERS: dict[str, str] = {
        "TealOrange": (
            "eq=brightness=-0.04:contrast=1.18:saturation=0.82,"
            "colorbalance=rs=0.08:gs=0.02:bs=-0.12:rm=0.04:gm=0.01:bm=-0.06:rh=-0.02:gh=0.0:bh=-0.04"
        ),
        "WarmGold": (
            "eq=brightness=0.02:contrast=1.12:saturation=1.05,"
            "colorbalance=rs=0.10:gs=0.04:bs=-0.15:rm=0.06:gm=0.02:bm=-0.08:rh=0.0:gh=0.0:bh=-0.05"
        ),
        "Desaturated": "eq=brightness=-0.06:contrast=1.22:saturation=0.55",
    }
    grade = edl.get("template_grade", "TealOrange")
    grade_filt = GRADE_FILTERS.get(grade, "")

    for entry in edl["edl"]:
        clip_path = Path(entry["clip_path"])
        if not clip_path.exists():
            # Try relative to ROOT
            clip_path = ROOT / entry["clip_path"]
        if not clip_path.exists():
            log.warning("Clip not found: %s", entry["clip_path"])
            continue

        out_clip = work_dir / f"slot_{entry['slot']:03d}.mp4"
        start_sec = entry["start_frame"] / fps
        dur_sec   = entry["duration_frames"] / fps

        vf_parts = ["crop=ih*9/16:ih", "scale=1080:1920:flags=lanczos"]
        if grade_filt:
            vf_parts.append(grade_filt)

        r = subprocess.run(
            ["ffmpeg", "-y",
             "-ss", str(start_sec), "-i", str(clip_path),
             "-t", str(dur_sec),
             "-vf", ",".join(vf_parts),
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an",
             str(out_clip)],
            capture_output=True,
        )
        if r.returncode == 0 and out_clip.exists():
            clip_files.append(out_clip)

    if not clip_files:
        log.error("FFmpeg fallback: no clips extracted")
        return False

    # Concat
    list_file = work_dir / "list.txt"
    list_file.write_text("\n".join(f"file '{p.absolute()}'" for p in clip_files))
    raw = work_dir / "raw.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(raw)],
        capture_output=True,
    )
    if r.returncode != 0:
        log.error("FFmpeg concat failed")
        return False

    # Mux audio
    total_dur = sum(e["duration_frames"] / fps for e in edl["edl"])
    r2 = subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(raw),
         "-ss", "0", "-t", str(total_dur), "-i", str(audio_path),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
         str(temp_output)],
        capture_output=True,
    )
    return r2.returncode == 0 and temp_output.exists()


# ── Step 5: FFmpeg post-process (spec-exact) ──────────────────────────────────

def ffmpeg_postprocess(
    temp_input: Path,
    final_output: Path,
    lut_name: str = "teal_orange",
    dry_run: bool = False,
) -> bool:
    """
    Apply LUT + audio loudnorm + libx264 crf18 to the rendered video.

    FFmpeg command (spec-exact):
      ffmpeg -i temp.mp4 -i mog_lut.cube
        -af "alimiter=limit=-1dB,loudnorm=I=-14:TP=-1.5:LRA=11"
        -vf "lut3d=file=mog_lut.cube:interp=tetrahedral"
        -c:v libx264 -crf 18 -preset slow
        -pix_fmt yuv420p -c:a aac -b:a 320k -movflags +faststart
        final.mp4
    """
    lut_path = LUTS_DIR / f"{lut_name}.cube"
    if not lut_path.exists():
        log.warning("LUT not found: %s — skipping colour grade", lut_path)
        shutil.copy2(temp_input, final_output)
        return True

    cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_input),
        "-af", "alimiter=limit=-1dB,loudnorm=I=-14:TP=-1.5:LRA=11",
        "-vf", f"lut3d=file='{lut_path}':interp=tetrahedral",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart",
        str(final_output),
    ]

    log.info("FFmpeg post-process: %s → %s  LUT=%s", temp_input.name, final_output.name, lut_name)

    if dry_run:
        log.info("[dry-run] %s", " ".join(cmd))
        return True

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log.error("FFmpeg post-process failed:\n%s", r.stderr.decode()[-500:])
        shutil.copy2(temp_input, final_output)
        return False
    return True


# ── Top-level orchestrator ────────────────────────────────────────────────────

def generate_mog_edit(
    clip_library_paths: list[Path],
    audio_path: Path,
    templates_dir: Path,
    output_path: Path,
    target_duration: float = 15.0,
    dry_run: bool = False,
    edl_only: bool = False,
    lut_name: str = "teal_orange",
    max_score_workers: int = 4,
) -> dict:
    """
    Full pipeline: score → analyze → assemble EDL → render → post-process.

    Returns result dict with keys: output, edl_path, edl, timings, success.
    """
    import time
    timings: dict[str, float] = {}
    result:  dict             = {"output": str(output_path), "success": False}

    name = output_path.stem
    edl_path = OUT_DIR / f"{name}_edl.json"

    # ── 1. Score clips ────────────────────────────────────────────────────────
    log.info("── Step 1: scoring %d clips ──", len(clip_library_paths))
    t0 = time.perf_counter()
    clip_scores = score_clips_parallel(clip_library_paths, max_workers=max_score_workers)
    timings["score_clips_sec"] = round(time.perf_counter() - t0, 2)
    log.info("   Scored %d clips in %.1fs", len(clip_scores), timings["score_clips_sec"])

    # ── 2. Analyze audio ─────────────────────────────────────────────────────
    log.info("── Step 2: audio analysis ──")
    t0 = time.perf_counter()
    from data.audio_analyzer import analyze_phonk_track  # type: ignore
    audio_analysis = analyze_phonk_track(str(audio_path), target_duration)
    timings["audio_analysis_sec"] = round(time.perf_counter() - t0, 2)
    log.info("   BPM=%.1f  beats=%d  sections=%d", audio_analysis["tempo"],
             audio_analysis["stats"]["beat_count"], len(audio_analysis["sections"]))

    # ── 3. Assemble EDL ──────────────────────────────────────────────────────
    log.info("── Step 3: assemble EDL ──")
    t0 = time.perf_counter()
    from data.sequencer import assemble_edit, load_scored_clips, load_templates  # type: ignore
    templates  = load_templates(str(templates_dir))
    scored     = load_scored_clips(clip_scores)
    edl        = assemble_edit(scored, audio_analysis, templates, target_duration)
    # Attach grade for fallback renderer
    edl["template_grade"] = next(
        (t.get("color_grade", "TealOrange") for t in templates
         if t.get("_name") == edl["template_id"]),
        "TealOrange",
    )
    timings["edl_assembly_sec"] = round(time.perf_counter() - t0, 2)

    # Write EDL
    edl_path.write_text(json.dumps(edl, indent=2))
    log.info("   EDL: %d slots  %.2fs  → %s", edl["stats"]["slot_count"],
             edl["total_duration"], edl_path.name)
    result["edl_path"] = str(edl_path)
    result["edl"]      = edl

    if edl_only:
        result["success"] = True
        result["timings"] = timings
        log.info("   (edl-only mode — stopping before render)")
        return result

    # ── 4. Render ─────────────────────────────────────────────────────────────
    log.info("── Step 4: render ──")
    t0 = time.perf_counter()
    temp_mp4 = TMP_DIR / f"{name}_temp.mp4"

    # Stage clips so Remotion's dev server can serve them via staticFile()
    edl_for_remotion, staged_public_clips = stage_clips_for_remotion(edl)
    rendered = remotion_render(edl_for_remotion, audio_path, temp_mp4,
                               public_dir=staged_public_clips, dry_run=dry_run)
    if not rendered or not temp_mp4.exists():
        log.info("   Remotion unavailable — using FFmpeg concat fallback")
        rendered = ffmpeg_concat_from_edl(edl, audio_path, temp_mp4)

    timings["render_sec"] = round(time.perf_counter() - t0, 2)

    if not rendered or not temp_mp4.exists():
        log.error("Render failed — cannot continue")
        result["timings"] = timings
        return result

    log.info("   Render done in %.1fs → %s (%.1fMB)", timings["render_sec"],
             temp_mp4.name, temp_mp4.stat().st_size / 1024 / 1024)

    # ── 5. FFmpeg post-process ────────────────────────────────────────────────
    log.info("── Step 5: FFmpeg post-process ──")
    t0 = time.perf_counter()
    ok = ffmpeg_postprocess(temp_mp4, output_path, lut_name=lut_name, dry_run=dry_run)
    timings["postprocess_sec"] = round(time.perf_counter() - t0, 2)
    temp_mp4.unlink(missing_ok=True)

    if ok and (dry_run or output_path.exists()):
        size_mb = output_path.stat().st_size / 1024 / 1024 if output_path.exists() else 0
        log.info("   Post-process done in %.1fs → %s (%.1fMB)",
                 timings["postprocess_sec"], output_path.name, size_mb)
        result["success"]  = True
        result["size_mb"]  = round(size_mb, 2)

    result["timings"] = timings
    return result


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    clips_dir: Path,
    audio_path: Path,
    templates_dir: Path,
    target_duration: float = 15.0,
    dry_run: bool = False,
    edl_only: bool = False,
) -> list[dict]:
    """
    Generate 3 edits: one per top-BPM template, rotating audio/LUT per grade.
    """
    clip_paths = sorted(clips_dir.glob("*.mp4"))
    if not clip_paths:
        log.error("No clips found in %s", clips_dir)
        return []

    # Load templates to pick top 3 by BPM
    import json as _json
    raw_templates = []
    for f in templates_dir.glob("*.json"):
        try:
            d = _json.loads(f.read_text())
            d["_name"] = f.stem
            raw_templates.append(d)
        except Exception:
            pass

    top3 = sorted(
        [t for t in raw_templates if t.get("bpm", 0) > 0],
        key=lambda t: t["bpm"],
        reverse=True,
    )[:3]

    if not top3:
        log.error("No templates with BPM found")
        return []

    log.info("Top-3 templates: %s", [f"{t['_name']}@{t['bpm']}BPM" for t in top3])

    results = []
    for i, tpl in enumerate(top3):
        grade    = tpl.get("color_grade", "TealOrange")
        lut_name = LUT_FOR_GRADE.get(grade, "neutral")
        slug     = tpl["_name"].split("_")[0].replace(".", "")[:12]
        out_name = f"v2_{slug}_{tpl['bpm']}bpm_graded.mp4"
        out_path = OUT_DIR / out_name

        log.info("\n%s\n  BATCH EDIT %d/%d: %s\n%s", "═"*56, i+1, len(top3), out_name, "═"*56)

        result = generate_mog_edit(
            clip_library_paths=clip_paths,
            audio_path=audio_path,
            templates_dir=templates_dir,
            output_path=out_path,
            target_duration=target_duration,
            dry_run=dry_run,
            edl_only=edl_only,
            lut_name=lut_name,
        )
        result["template"] = tpl["_name"]
        result["bpm"]      = tpl["bpm"]
        result["grade"]    = grade
        results.append(result)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ascension Engine — generate_batch.py v2")
    parser.add_argument("--clips",    default=str(CLIPS_DIR), help="Dir of clip .mp4 files")
    parser.add_argument("--audio",    default=None, help="Audio source video/file")
    parser.add_argument("--templates",default=str(SEQ_DIR), help="Sequence templates dir")
    parser.add_argument("--target-duration", "-t", type=float, default=15.0)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--edl-only", action="store_true", help="Skip render, output EDL JSON only")
    parser.add_argument("--workers",  type=int, default=4, help="Parallel scoring workers")
    parser.add_argument("--verbose",  "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    clips_dir     = Path(args.clips)
    templates_dir = Path(args.templates)
    audio_path    = pick_audio_source(Path(args.audio) if args.audio else None)

    log.info("Clips dir  : %s  (%d clips)", clips_dir, len(list(clips_dir.glob("*.mp4"))))
    log.info("Audio      : %s", audio_path.name)
    log.info("Templates  : %s  (%d found)", templates_dir, len(list(templates_dir.glob("*.json"))))
    log.info("Duration   : %.1fs", args.target_duration)

    results = run_batch(
        clips_dir=clips_dir,
        audio_path=audio_path,
        templates_dir=templates_dir,
        target_duration=args.target_duration,
        dry_run=args.dry_run,
        edl_only=args.edl_only,
    )

    # Summary
    manifest = OUT_DIR / "v2_batch_manifest.json"
    manifest.write_text(json.dumps(results, indent=2))

    print(f"\n{'═'*58}")
    print("  BATCH RESULTS")
    print(f"{'═'*58}")
    for r in results:
        ok = "✓" if r.get("success") else "✗"
        name = Path(r.get("output", "?")).name
        if r.get("success"):
            t = r.get("timings", {})
            print(f"  {ok}  {name}")
            print(f"       template={r.get('template','?')}  BPM={r.get('bpm','?')}  grade={r.get('grade','?')}")
            if not args.edl_only:
                print(f"       size={r.get('size_mb',0):.1f}MB  "
                      f"score_t={t.get('score_clips_sec',0):.1f}s  "
                      f"render_t={t.get('render_sec',0):.1f}s  "
                      f"post_t={t.get('postprocess_sec',0):.1f}s")
            if r.get("edl_path"):
                edl = r.get("edl", {})
                print(f"       edl={Path(r['edl_path']).name}  "
                      f"slots={edl.get('stats',{}).get('slot_count','?')}  "
                      f"avg_cut={edl.get('stats',{}).get('avg_cut_sec','?')}s")
        else:
            print(f"  {ok}  {name}  — FAILED")
    print(f"\n  Manifest: {manifest}\n")


if __name__ == "__main__":
    main()
