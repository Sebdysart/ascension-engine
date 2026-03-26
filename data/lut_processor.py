#!/usr/bin/env python3
"""
Ascension Engine — FFmpeg LUT Post-Processor
Applies .cube LUT files + grain overlay to rendered MP4s in out/.

Usage:
    python data/lut_processor.py                    # process all MP4s in out/
    python data/lut_processor.py out/myvideo.mp4    # process a single file
    python data/lut_processor.py --lut luts/teal_orange.cube out/myvideo.mp4
    python data/lut_processor.py --list-luts        # show available LUTs
    python data/lut_processor.py --dry-run          # preview without writing

Called via npm:
    npm run postprocess
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
LUTS_DIR = _ROOT / "luts"
OUT_DIR = _ROOT / "out"
PROCESSED_SUFFIX = "_graded"

log = logging.getLogger("lut_processor")

# ── Sample LUT generators ─────────────────────────────────────────────────────

def _write_identity_lut(path: Path, size: int = 33) -> None:
    """Write a 33³ identity .cube LUT (no colour change — baseline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ascension Engine — Identity LUT",
        f"LUT_3D_SIZE {size}",
        "",
    ]
    step = 1.0 / (size - 1)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rv = round(r * step, 6)
                gv = round(g * step, 6)
                bv = round(b * step, 6)
                lines.append(f"{rv} {gv} {bv}")
    path.write_text("\n".join(lines) + "\n")


def _write_teal_orange_lut(path: Path, size: int = 33) -> None:
    """Write a teal-orange stylised .cube LUT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ascension Engine — Teal Orange LUT",
        f"LUT_3D_SIZE {size}",
        "",
    ]
    step = 1.0 / (size - 1)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rv = r * step
                gv = g * step
                bv = b * step

                # Lift shadows toward teal, push highlights toward orange
                luma = 0.299 * rv + 0.587 * gv + 0.114 * bv

                # Shadow teal push: boost blue/green in darks
                shadow_mask = max(0.0, 1.0 - luma * 3)
                bv += shadow_mask * 0.06
                gv += shadow_mask * 0.03

                # Highlight orange push: boost red/green in brights
                hi_mask = max(0.0, luma - 0.6) * 2.5
                rv += hi_mask * 0.08
                gv += hi_mask * 0.03

                # Slight saturation lift
                avg = (rv + gv + bv) / 3
                rv = avg + (rv - avg) * 1.15
                gv = avg + (gv - avg) * 1.15
                bv = avg + (bv - avg) * 1.15

                rv = round(max(0.0, min(1.0, rv)), 6)
                gv = round(max(0.0, min(1.0, gv)), 6)
                bv = round(max(0.0, min(1.0, bv)), 6)
                lines.append(f"{rv} {gv} {bv}")
    path.write_text("\n".join(lines) + "\n")


def _write_cold_blue_lut(path: Path, size: int = 33) -> None:
    """Write a cold-blue desaturated .cube LUT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ascension Engine — Cold Blue LUT",
        f"LUT_3D_SIZE {size}",
        "",
    ]
    step = 1.0 / (size - 1)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rv = r * step
                gv = g * step
                bv = b * step

                # Desaturate
                luma = 0.299 * rv + 0.587 * gv + 0.114 * bv
                rv = luma + (rv - luma) * 0.8
                gv = luma + (gv - luma) * 0.8
                bv = luma + (bv - luma) * 0.8

                # Blue channel boost
                bv = min(1.0, bv * 1.12 + 0.03)
                # Slight red reduction
                rv = max(0.0, rv * 0.93)

                rv = round(max(0.0, min(1.0, rv)), 6)
                gv = round(max(0.0, min(1.0, gv)), 6)
                bv = round(max(0.0, min(1.0, bv)), 6)
                lines.append(f"{rv} {gv} {bv}")
    path.write_text("\n".join(lines) + "\n")


def _write_warm_gold_lut(path: Path, size: int = 33) -> None:
    """Write a warm-gold sunrise .cube LUT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ascension Engine — Warm Gold LUT",
        f"LUT_3D_SIZE {size}",
        "",
    ]
    step = 1.0 / (size - 1)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rv = r * step
                gv = g * step
                bv = b * step

                # Warm lift: boost reds and greens, reduce blues
                rv = min(1.0, rv * 1.1 + 0.04)
                gv = min(1.0, gv * 1.05 + 0.02)
                bv = max(0.0, bv * 0.85)

                # Slight sepia in midtones
                luma = 0.299 * rv + 0.587 * gv + 0.114 * bv
                mid_mask = 1.0 - abs(luma - 0.5) * 2
                rv += mid_mask * 0.03
                gv += mid_mask * 0.015

                rv = round(max(0.0, min(1.0, rv)), 6)
                gv = round(max(0.0, min(1.0, gv)), 6)
                bv = round(max(0.0, min(1.0, bv)), 6)
                lines.append(f"{rv} {gv} {bv}")
    path.write_text("\n".join(lines) + "\n")


def ensure_sample_luts() -> None:
    """Create sample LUT files if none exist in luts/."""
    LUTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = {
        "neutral.cube": _write_identity_lut,
        "teal_orange.cube": _write_teal_orange_lut,
        "cold_blue.cube": _write_cold_blue_lut,
        "warm_gold.cube": _write_warm_gold_lut,
    }
    for name, writer in samples.items():
        lut_path = LUTS_DIR / name
        if not lut_path.exists():
            log.info("Creating sample LUT: %s", name)
            writer(lut_path)


def list_luts() -> list[Path]:
    """Return all .cube files in the luts/ directory."""
    return sorted(LUTS_DIR.glob("*.cube"))


# ── FFmpeg pipeline ───────────────────────────────────────────────────────────

def _pick_lut(lut_arg: str | None) -> Path | None:
    """Return a LUT path — explicit arg, or auto-detect from out/ filename, or default."""
    if lut_arg:
        p = Path(lut_arg)
        if not p.exists():
            p = LUTS_DIR / lut_arg
        if p.exists():
            return p
        log.warning("LUT not found: %s", lut_arg)
        return None

    # Default to teal_orange if available
    default = LUTS_DIR / "teal_orange.cube"
    return default if default.exists() else next(iter(list_luts()), None)


def _build_grain_filter(strength: float = 12.0) -> str:
    """Return an FFmpeg filter string that adds film grain."""
    # geq generates per-pixel noise; overlay=blend adds grain on top
    return (
        f"split=2[main][grain_src],"
        f"[grain_src]geq="
        f"'p(X,Y)+({strength}*(random(1)-0.5))':"
        f"'p(X,Y)+({strength}*(random(2)-0.5))':"
        f"'p(X,Y)+({strength}*(random(3)-0.5))'[grain],"
        f"[main][grain]blend=all_mode=overlay:all_opacity=0.18"
    )


def apply_lut_and_grain(
    input_path: Path,
    output_path: Path,
    lut_path: Path | None,
    grain_strength: float = 12.0,
    dry_run: bool = False,
) -> bool:
    """
    Apply LUT + grain overlay to a video file using FFmpeg.
    Returns True on success.
    """
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found")
        return False

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return False

    # Build filter chain
    filters: list[str] = []

    if lut_path and lut_path.exists():
        # haldclut needs a HALD image generated from the cube file
        # We use the lut3d filter instead (direct .cube support in FFmpeg)
        filters.append(f"lut3d='{lut_path}'")
        log.info("   Applying LUT: %s", lut_path.name)
    else:
        log.info("   No LUT applied (file missing or none specified)")

    if grain_strength > 0:
        filters.append(_build_grain_filter(grain_strength))
        log.info("   Grain overlay: strength=%.1f", grain_strength)

    filter_str = ",".join(filters) if filters else "null"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]

    if dry_run:
        log.info("[dry-run] Would run: %s", " ".join(cmd))
        return True

    log.info("   → %s", output_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg failed: %s", e.stderr[-500:] if e.stderr else e)
        return False


def process_output_dir(
    lut_arg: str | None = None,
    grain_strength: float = 12.0,
    dry_run: bool = False,
) -> list[Path]:
    """
    Process all MP4s in out/ that don't already have the graded suffix.
    Returns list of output paths.
    """
    ensure_sample_luts()
    lut_path = _pick_lut(lut_arg)

    mp4s = [
        p for p in sorted(OUT_DIR.glob("*.mp4"))
        if PROCESSED_SUFFIX not in p.stem
    ]

    if not mp4s:
        log.info("No ungraded MP4s found in %s", OUT_DIR)
        return []

    outputs: list[Path] = []
    for mp4 in mp4s:
        out_path = mp4.parent / f"{mp4.stem}{PROCESSED_SUFFIX}.mp4"
        success = apply_lut_and_grain(mp4, out_path, lut_path, grain_strength, dry_run)
        if success:
            outputs.append(out_path)

    return outputs


def process_single(
    input_path: Path,
    lut_arg: str | None = None,
    grain_strength: float = 12.0,
    dry_run: bool = False,
) -> Path | None:
    """Process a single MP4 file. Returns output path or None."""
    ensure_sample_luts()
    lut_path = _pick_lut(lut_arg)
    out_path = input_path.parent / f"{input_path.stem}{PROCESSED_SUFFIX}.mp4"
    success = apply_lut_and_grain(input_path, out_path, lut_path, grain_strength, dry_run)
    return out_path if success else None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Ascension Engine — LUT + grain post-processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="MP4 to process (default: all in out/)")
    parser.add_argument("--lut", help="LUT name or path (default: teal_orange.cube)")
    parser.add_argument("--grain", type=float, default=12.0, help="Grain strength (default: 12)")
    parser.add_argument("--list-luts", action="store_true", help="List available LUTs and exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_sample_luts()

    if args.list_luts:
        luts = list_luts()
        if luts:
            print("Available LUTs:")
            for lut in luts:
                print(f"  • {lut.name}")
        else:
            print("No LUT files found in luts/")
        return

    if args.input:
        out = process_single(Path(args.input).resolve(), args.lut, args.grain, args.dry_run)
        if out:
            print(f"Graded: {out}")
    else:
        outputs = process_output_dir(args.lut, args.grain, args.dry_run)
        if outputs:
            print(f"Graded {len(outputs)} file(s):")
            for o in outputs:
                print(f"  • {o.name}")
        else:
            print("Nothing to process.")


if __name__ == "__main__":
    main()
