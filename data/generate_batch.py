#!/usr/bin/env python3
"""
Ascension Engine — generate_batch.py
Reads top 3 sequence templates by BPM, selects matching clips from
input/gold/, concatenates via FFmpeg, runs lut_processor on each output.

Usage:
    python3 data/generate_batch.py
    python3 data/generate_batch.py --dry-run
"""

import argparse, io, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

# ── Narrative engine lazy import ──────────────────────────────────────────────
def _import_narrative_engine():
    """Import narrative modules — returns tuple of functions or None on failure."""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from narrative_engine import build_narrative, describe_narrative, GRADE_SPEC
        from hook_generator import generate_hook_spec
        from sequencer import build_narrative_sequence
        return build_narrative, describe_narrative, GRADE_SPEC, generate_hook_spec, build_narrative_sequence
    except ImportError as e:
        return None

ROOT    = Path(__file__).resolve().parent.parent
GOLD    = ROOT / "input" / "gold"
OUT     = ROOT / "out"
TMP     = ROOT / "tmp_genbatch"
SEQ_DIR = ROOT / "library" / "sequence_templates"
LUTS    = ROOT / "luts"

OUT.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)

LUT_MAP = {
    "TealOrange":   "teal_orange",
    "WarmGold":     "warm_gold",
    "Desaturated":  "desaturated",
    "unknown":      "neutral",
}

GRADE_FILTER = {
    "TealOrange": (
        "eq=brightness=-0.04:contrast=1.18:saturation=0.82,"
        "colorbalance=rs=0.08:gs=0.02:bs=-0.12:"
        "rm=0.04:gm=0.01:bm=-0.06:rh=-0.02:gh=0.0:bh=-0.04"
    ),
    "WarmGold": (
        "eq=brightness=0.02:contrast=1.12:saturation=1.05,"
        "colorbalance=rs=0.10:gs=0.04:bs=-0.15:"
        "rm=0.06:gm=0.02:bm=-0.08:rh=0.0:gh=0.0:bh=-0.05"
    ),
    "Desaturated": (
        "eq=brightness=-0.06:contrast=1.22:saturation=0.55,"
        "colorbalance=rs=-0.02:gs=0.0:bs=0.04"
    ),
    "unknown": "",
}

# Audio with strong beat for each grade
BEAT_AUDIO = {
    "TealOrange":  GOLD / "bp_masterx_7589676463071268118.mp4",
    "WarmGold":    GOLD / "bp4ever.ae_7542208986699844869.mp4",
    "Desaturated": GOLD / "black.pill.city_7600958206642326806.mp4",
}


def score_frame(png_bytes: bytes) -> float:
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        arr = np.array(img.crop((w//4, h//6, 3*w//4, h//2)), dtype=float)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        return max(0.0, (100 - abs(gray.mean() - 110)*0.8) + min(gray.std()*1.5, 50))
    except Exception:
        return 0.0


def best_window(src: Path, dur: float) -> float:
    probe = subprocess.run(
        ["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0",str(src)],
        capture_output=True, text=True)
    try:
        total = float(probe.stdout.strip())
    except Exception:
        return 0.0
    if total <= dur + 0.5:
        return 0.0
    step = max(0.3, dur * 0.4)
    t, best_t, best_s = 0.5, 0.5, -1.0
    while t < total - dur:
        r = subprocess.run(
            ["ffmpeg","-y","-ss",str(t),"-i",str(src),"-frames:v","1",
             "-f","image2pipe","-vcodec","png","pipe:1"],
            capture_output=True)
        if r.returncode == 0 and r.stdout:
            s = score_frame(r.stdout)
            if s > best_s:
                best_s, best_t = s, t
        t += step
    return best_t


def detect_beats(src: Path, max_sec: float = 22.0) -> tuple[float, list[float]]:
    try:
        import librosa
    except ImportError:
        return 108.0, []
    wav = TMP / "beats.wav"
    subprocess.run(
        ["ffmpeg","-y","-i",str(src),"-ac","1","-ar","22050","-vn",str(wav)],
        capture_output=True)
    y, sr = librosa.load(str(wav), sr=22050, duration=max_sec)
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(tempo) if isinstance(tempo,(int,float)) else float(tempo[0])
    return bpm, [float(t) for t in frames if float(t) <= max_sec]


def extract_clip(src: Path, start: float, dur: float, out: Path, grade_filt: str) -> bool:
    vf = ["crop=ih*9/16:ih", "scale=1080:1920:flags=lanczos"]
    if grade_filt:
        vf.append(grade_filt)
    r = subprocess.run(
        ["ffmpeg","-y","-ss",str(start),"-i",str(src),"-t",str(dur),
         "-vf",",".join(vf),"-c:v","libx264","-preset","fast","-crf","20","-an",str(out)],
        capture_output=True)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 500


def concat_and_mux(clips: list[Path], audio_src: Path, out: Path, dur: float) -> bool:
    lst = TMP / "list.txt"
    lst.write_text("\n".join(f"file '{p.absolute()}'" for p in clips))
    raw = TMP / "raw.mp4"
    r = subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy",str(raw)],
        capture_output=True)
    if r.returncode != 0:
        return False
    r2 = subprocess.run(
        ["ffmpeg","-y","-i",str(raw),"-ss","0","-t",str(dur),"-i",str(audio_src),
         "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(out)],
        capture_output=True)
    raw.unlink(missing_ok=True)
    return r2.returncode == 0 and out.exists()


def load_top_templates(n: int = 3) -> list[dict]:
    templates = []
    for f in SEQ_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            d["_file"] = f
            templates.append(d)
        except Exception:
            pass
    # Sort by BPM descending (highest energy first), skip 0-BPM
    templates = [t for t in templates if t.get("bpm", 0) > 0]
    templates.sort(key=lambda t: t.get("bpm", 0), reverse=True)
    return templates[:n]


def sources_for_template(template: dict) -> list[Path]:
    """Pick gold source videos relevant to the template's clip tags."""
    # Gather all tags used in this template
    tag_set: set[str] = set()
    for c in template.get("clip_order", []):
        tag_set.update(c.get("tags", []))

    # Score each gold video by name / tag heuristics
    candidates = [f for f in GOLD.glob("*.mp4") if f.exists()]
    if not candidates:
        return []

    # Prefer bp-coded videos for mog edits; fallback to all
    bp_pref = [f for f in candidates if any(k in f.name for k in ["bp","gold","morph","saffyro","haifluke"])]
    dark_pref = [f for f in candidates if any(k in f.name for k in ["black","roge","pilled","editz"])]

    grade = template.get("color_grade","")
    if grade == "Desaturated":
        pool = dark_pref + [f for f in candidates if f not in dark_pref]
    else:
        pool = bp_pref + [f for f in candidates if f not in bp_pref]

    # Remove duplicates, keep order
    seen = set()
    out = []
    for f in pool:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _assign_clips_to_narrative_slots(
    slots: list,
    sources: list,
    gf: str,
    tmp_dir: "Path",
    dry_run: bool = False,
) -> list[dict]:
    """
    Assign source clips to narrative slots based on each slot's preferred_pool.

    Pool → source heuristic:
      victim_contrast  → prefer dark/roge-style sources
      good_parts       → prefer bp-coded sources
      mid_tier         → any sources
    """
    bp_pref   = [f for f in sources if any(k in f.name for k in ["bp", "gold", "morph", "saffyro"])]
    dark_pref = [f for f in sources if any(k in f.name for k in ["black", "roge", "pilled", "editz"])]

    pool_sources = {
        "good_parts":      bp_pref or sources,
        "victim_contrast": dark_pref or sources,
        "mid_tier":        sources,
        "low_cinematic":   sources,
    }

    assignments: list[dict] = []
    for slot in slots:
        pool     = getattr(slot, "preferred_pool", "mid_tier")
        src_list = pool_sources.get(pool, sources)
        src      = src_list[len(assignments) % len(src_list)]
        dur      = slot.duration

        clip_out = tmp_dir / f"c{slot.clip_index:03d}.mp4"

        if not dry_run:
            start = best_window(src, dur)
            ok    = extract_clip(src, start, dur, clip_out, gf)
            if not ok:
                ok = extract_clip(src, 0.5, dur, clip_out, gf)
        else:
            ok = True

        assignments.append({
            "slot_index":   slot.clip_index,
            "act":          getattr(slot, "act", "unknown"),
            "is_victim":    getattr(slot, "is_victim_slot", False),
            "pre_drop_sil": getattr(slot, "pre_drop_silence_sec", 0.0),
            "zoom_pulse":   getattr(slot, "zoom_pulse", False),
            "pool":         pool,
            "source":       src.name,
            "duration":     dur,
            "extracted":    ok,
            "path":         str(clip_out) if ok else None,
        })

    return assignments


def generate_from_template(template: dict, dry_run: bool = False) -> dict:
    vid_id   = template.get("vid_id") or template.get("_file").stem
    bpm      = template.get("bpm", 108)
    grade    = template.get("color_grade", "TealOrange")
    n_clips  = max(template.get("total_clips", 8), 6)
    total_dur = min(template.get("total_duration_sec", 18.0), 22.0)

    # Slug output name
    slug = vid_id.split("_")[0].replace(".","")[:12]
    out_name = f"gen_{slug}_{bpm}bpm"
    out_raw  = OUT / f"{out_name}_raw.mp4"
    out_final = OUT / f"{out_name}_graded.mp4"

    print(f"\n{'─'*58}")
    print(f"  [{bpm} BPM | {grade}] {vid_id[:45]}")
    print(f"  Output: {out_final.name}  clips={n_clips}  dur={total_dur:.1f}s")
    print(f"{'─'*58}")

    if dry_run:
        print("  (dry-run — skipping)")
        return {"name": out_name, "dry_run": True, "bpm": bpm, "grade": grade}

    sources = sources_for_template(template)
    if not sources:
        print("  [error] no sources")
        return {"name": out_name, "error": "no_sources"}

    # ── Beat detection ─────────────────────────────────────────────────────────
    audio_src = BEAT_AUDIO.get(grade, sources[0])
    if not audio_src.exists():
        audio_src = sources[0]

    print(f"  ▶ Beat detection ({audio_src.name[:40]}) ...", flush=True)
    detected_bpm, beats = detect_beats(audio_src, max_sec=total_dur + 5)
    actual_bpm = detected_bpm if beats else float(bpm)

    narrative_report: dict = {}
    tmp_dir = TMP / out_name
    tmp_dir.mkdir(exist_ok=True)
    gf = GRADE_FILTER.get(grade, "")

    narrative_mods = _import_narrative_engine()
    if narrative_mods is not None:
        # ── Narrative arc mode ──────────────────────────────────────────────────
        _build_narrative, _describe_narrative, _GRADE_SPEC, _generate_hook_spec, _build_narrative_sequence = narrative_mods

        print(f"  ▶ Building MogNarrative arc (BPM={actual_bpm:.1f}) ...", flush=True)
        nslots = _build_narrative_sequence(bpm=actual_bpm, total_sec=total_dur, beats=beats)
        hook_spec = _generate_hook_spec(beats=beats, dry_run=dry_run)

        assignments = _assign_clips_to_narrative_slots(nslots, sources, gf, tmp_dir, dry_run)
        extracted   = [Path(a["path"]) for a in assignments if a["extracted"] and a["path"]]
        final_durs  = [a["duration"] for a in assignments if a["extracted"]]
        used        = sum(final_durs)

        # Build narrative report
        act_counts: dict[str, int] = {"victim": 0, "awakening": 0, "ascension": 0}
        victim_placements: list[int] = []
        pre_drop_silence_slots: list[dict] = []
        for a in assignments:
            act_counts[a["act"]] = act_counts.get(a["act"], 0) + 1
            if a["is_victim"]:
                victim_placements.append(a["slot_index"])
            if a["pre_drop_sil"] > 0:
                pre_drop_silence_slots.append({"slot": a["slot_index"], "sec": a["pre_drop_sil"]})

        narrative_report = {
            "mode": "narrative_arc",
            "act_cuts": act_counts,
            "victim_placements": victim_placements,
            "pre_drop_silence": pre_drop_silence_slots,
            "total_cuts": len(assignments),
        }

        print(
            f"  ▶ Narrative: Act1={act_counts['victim']} Act2={act_counts['awakening']} "
            f"Act3={act_counts['ascension']}  victims@{victim_placements}",
            flush=True,
        )
        for a in assignments:
            status      = "✓" if a["extracted"] else "✗"
            victim_flag = " [VICTIM]" if a["is_victim"] else ""
            predrop     = f" [sil={a['pre_drop_sil']:.3f}s]" if a["pre_drop_sil"] > 0 else ""
            print(
                f"    [{a['slot_index']:2d}] {a['act']:10s}  {a['source'][:35]:38} "
                f"{a['duration']:.2f}s {status}{victim_flag}{predrop}",
                flush=True,
            )
    else:
        # ── Fallback: flat beat-grid (narrative_engine not available) ───────────
        print("  ▶ [fallback] narrative_engine not found — using flat beat grid", flush=True)
        beat_interval  = 60.0 / actual_bpm
        target_cut     = total_dur / n_clips
        beats_per_cut  = max(1, round(target_cut / beat_interval))
        cut_durs: list[float] = []
        prev = 0.0
        if beats:
            for i, bt in enumerate(beats):
                if i % beats_per_cut == 0 and i > 0:
                    d = bt - prev
                    if 0.3 <= d <= 5.0:
                        cut_durs.append(d)
                        prev = bt
        else:
            target_cut_fb = total_dur / n_clips
            cut_durs = [target_cut_fb] * n_clips

        used, final_durs = 0.0, []
        for d in cut_durs:
            if used + d > total_dur:
                break
            final_durs.append(d)
            used += d
        if not final_durs:
            final_durs = [total_dur / n_clips] * n_clips
            used = total_dur

        _avg_cut_fb = used / len(final_durs)
        print(f"  ▶ {len(final_durs)} cuts  avg={_avg_cut_fb:.2f}s  total={used:.1f}s  BPM={actual_bpm:.1f}", flush=True)

        extracted = []
        for i, dur in enumerate(final_durs):
            src      = sources[i % len(sources)]
            clip_out = tmp_dir / f"c{i:03d}.mp4"
            start    = best_window(src, dur)
            ok       = extract_clip(src, start, dur, clip_out, gf)
            if not ok:
                ok = extract_clip(src, 0.5, dur, clip_out, gf)
            if ok:
                extracted.append(clip_out)
                print(f"    [{i+1:2d}/{len(final_durs)}] {src.name[:35]:38} {dur:.2f}s ✓", flush=True)
            else:
                print(f"    [{i+1:2d}/{len(final_durs)}] {src.name[:35]:38} {dur:.2f}s ✗", flush=True)

    if len(extracted) < 2:
        return {"name": out_name, "error": "insufficient_clips"}

    avg_cut = (used / len(final_durs)) if final_durs else 0.0

    # Concat + mux
    print(f"  ▶ Concatenating {len(extracted)} clips ...", flush=True)
    ok = concat_and_mux(extracted, audio_src, out_raw, used)
    if not ok:
        return {"name": out_name, "error": "concat_failed"}

    # LUT + grain via lut_processor.py
    lut_name = LUT_NAMES = LUT_MAP.get(grade, "neutral")
    lut_path = LUTS / f"{lut_name}.cube"
    print(f"  ▶ lut_processor: {lut_name}.cube + grain ...", flush=True)
    lut_args = [sys.executable, "data/lut_processor.py", "--lut", str(lut_path), str(out_raw)]
    r = subprocess.run(lut_args, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"  [warn] lut_processor failed:\n{r.stderr[-300:]}")
        shutil.copy2(out_raw, out_final)
    else:
        # lut_processor writes to out/<name>_graded.mp4
        expected = out_raw.parent / (out_raw.stem + "_graded.mp4")
        if expected.exists():
            shutil.move(str(expected), str(out_final))
        else:
            shutil.copy2(out_raw, out_final)
    out_raw.unlink(missing_ok=True)

    size_mb = out_final.stat().st_size / (1024*1024) if out_final.exists() else 0
    result = {
        "name": out_name,
        "output": str(out_final),
        "template_dna": vid_id,
        "bpm": round(actual_bpm, 1),
        "color_grade": grade,
        "lut": lut_name,
        "clips_used": len(extracted),
        "avg_cut_sec": round(avg_cut, 3),
        "total_duration_sec": round(used, 2),
        "file_size_mb": round(size_mb, 2),
        "narrative": narrative_report,
    }
    print(f"  ✅  {out_final.name}  ({size_mb:.1f} MB)")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'═'*58}")
    print("  ASCENSION ENGINE — generate_batch.py")
    print(f"  Top {args.top_n} templates by BPM → 3 output edits")
    print(f"{'═'*58}")

    templates = load_top_templates(args.top_n)
    if not templates:
        print("[error] No templates found in", SEQ_DIR)
        sys.exit(1)

    print(f"\n  Templates selected:")
    for t in templates:
        print(f"    {t.get('vid_id','?')[:48]:50} BPM={t.get('bpm',0):3}  grade={t.get('color_grade','?')}")

    results = []
    for t in templates:
        r = generate_from_template(t, dry_run=args.dry_run)
        results.append(r)

    manifest = OUT / "generate_batch_manifest.json"
    manifest.write_text(json.dumps(results, indent=2))

    print(f"\n{'═'*58}")
    print("  RESULTS")
    print(f"{'═'*58}")
    for r in results:
        if "error" in r:
            print(f"  ✗  {r['name']} — {r.get('error')}")
        elif r.get("dry_run"):
            print(f"  ~  {r['name']} (dry-run)")
        else:
            print(f"  ✓  {r['name']}")
            print(f"       BPM={r['bpm']}  grade={r['color_grade']}  cuts={r['clips_used']}  avg={r['avg_cut_sec']}s  dur={r['total_duration_sec']}s  {r['file_size_mb']}MB")
            nr = r.get("narrative", {})
            if nr.get("mode") == "narrative_arc":
                ac = nr.get("act_cuts", {})
                vp = nr.get("victim_placements", [])
                ps = len(nr.get("pre_drop_silence", []))
                print(f"       Narrative: Act1={ac.get('victim',0)} Act2={ac.get('awakening',0)} Act3={ac.get('ascension',0)}")
                print(f"       Victims@slots={vp}  pre-drop-silence={ps}")
    print(f"\n  Manifest: {manifest}")


if __name__ == "__main__":
    main()
