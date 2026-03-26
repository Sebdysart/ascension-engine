#!/usr/bin/env bash
# Ascension Engine — full-fidelity setup
# Run once to install all Python dependencies.
#
# Usage:
#   bash scripts/setup.sh
#   bash scripts/setup.sh --upgrade   # re-install / upgrade all packages

set -euo pipefail

UPGRADE=""
for arg in "$@"; do
  [[ "$arg" == "--upgrade" ]] && UPGRADE="--upgrade"
done

echo "════════════════════════════════════════════════════════════"
echo "  Ascension Engine — dependency setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Prefer python3; fall back to python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON="python"
fi

echo "Python : $($PYTHON --version)"
echo "pip    : $($PYTHON -m pip --version | awk '{print $1,$2}')"
echo ""

echo "── Installing requirements.txt ──────────────────────────────"
$PYTHON -m pip install $UPGRADE -r requirements.txt
echo ""

echo "── Verifying key imports ────────────────────────────────────"
$PYTHON - <<'PYCHECK'
import importlib, sys

checks = [
    ("scenedetect", "scenedetect"),
    ("cv2",         "opencv-python"),
    ("PIL",         "Pillow"),
    ("imagehash",   "imagehash"),
    ("watchdog",    "watchdog"),
    ("librosa",     "librosa"),
    ("numpy",       "numpy"),
    ("easyocr",     "easyocr"),
    ("anthropic",   "anthropic"),
    ("dotenv",      "python-dotenv"),
]

all_ok = True
for mod, pkg in checks:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "")
        print(f"  ✅  {pkg:<22} ({ver})")
    except ImportError:
        print(f"  ❌  {pkg:<22} — import failed")
        all_ok = False

if not all_ok:
    print("\nSome imports failed. Re-run: bash scripts/setup.sh --upgrade")
    sys.exit(1)
PYCHECK

echo ""

# Check claude CLI
echo "── Checking claude CLI ──────────────────────────────────────"
if command -v claude &>/dev/null; then
  echo "  ✅  claude CLI found ($(claude --version 2>/dev/null || echo 'version unknown'))"
else
  echo "  ⚠️   claude CLI not found"
  echo "       Vision tagging requires Claude Code: https://claude.ai/code"
fi

# Check ffmpeg
echo ""
echo "── Checking ffmpeg ──────────────────────────────────────────"
if command -v ffmpeg &>/dev/null; then
  echo "  ✅  ffmpeg found ($(ffmpeg -version 2>&1 | head -1 | awk '{print $3}'))"
else
  echo "  ❌  ffmpeg not found — required for ingest, LUT post-processing"
  echo "       macOS: brew install ffmpeg"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete. Run the pipeline:"
echo "    python data/ingest.py --once input/gold/<video>.mp4"
echo "════════════════════════════════════════════════════════════"
