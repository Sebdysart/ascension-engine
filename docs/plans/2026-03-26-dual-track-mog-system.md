# Dual-Track Mogger vs Victim System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a mog-score-driven dual-track routing system that classifies every clip as S-Tier (good_parts/), Victim Contrast (victim_contrast/), or Mid-Tier (mid_tier/), and surfaces this in the TypeScript library.

**Architecture:** The vision tagger gains a second Claude call that scores faces 0.0–1.0 (jawline, hunter eyes, symmetry, bone structure). Ingest routes and grades clips by score. clipLibrary.ts adds three new query helpers and a `SequenceTemplate` interface with optional `contrast_slot` support. Maintenance CLI flags let the operator re-classify the whole library.

**Tech Stack:** Python + FFmpeg (routing + victim grade), TypeScript (clipLibrary.ts), Claude CLI (scoring), JSON manifests.

---

## Existing file map (read before editing)

| File | Role |
|------|------|
| `data/vision_tagger.py` | Calls `claude` CLI to tag keyframes; `tag_clips_for_video()` entry point |
| `data/ingest.py` | Full ingest pipeline, 9 steps; `ingest_video()` orchestrates everything |
| `data/lut_processor.py` | FFmpeg LUT + grain pipeline; `apply_lut_and_grain()` is the workhorse |
| `src/lib/clipLibrary.ts` | TS library queries; `Clip` interface; `remixSequence()`, `getBrutalTimeline()` |
| `library/clips/` | Scene-segmented MP4s + manifests written by `export_clip_segments()` |
| `clip-manifest.json` | Master flat clip list read by clipLibrary.ts |

---

## Task 1: Add mog scoring to `data/vision_tagger.py`

**Files:**
- Modify: `data/vision_tagger.py`

### Step 1: Add mog-score prompt constant

After line 64 (end of `_PROMPT_TEMPLATE`), add:

```python
_MOG_SCORE_PROMPT = """\
Use the Read tool to read each of these image files:
{paths}

You are scoring a male face for physical dominance / looksmax potential on a 0.0–1.0 scale.
Evaluate ONLY what is visible. If no clear face, return 0.5.

Scoring criteria:
- Jawline definition and sharpness (0–0.25)
- Hunter eyes / orbital rim / canthal tilt (0–0.25)
- Facial symmetry (0–0.20)
- Bone structure / cheekbones / brow ridge (0–0.20)
- Overall mog potential vs average male (0–0.10)

Output ONLY a valid JSON object, nothing else:
{"mog_score": <float 0.0-1.0>, "dominant_trait": "<one word>", "notes": "<10 words max>"}

Examples:
{"mog_score": 0.92, "dominant_trait": "jawline", "notes": "sharp jaw, hunter eyes, exceptional bone structure"}
{"mog_score": 0.41, "dominant_trait": "recessed", "notes": "weak chin, flat face, poor bone projection"}
{"mog_score": 0.5, "dominant_trait": "average", "notes": "no clear face visible or average features"}
"""
```

### Step 2: Add `_call_claude_mog_score()` function

After `_call_claude_cli()` (after line 160), add:

```python
def _call_claude_mog_score(frame_paths: list[Path]) -> dict:
    """
    Score face clips 0.0–1.0 for mog potential via Claude vision.
    Returns dict with mog_score, dominant_trait, notes.
    """
    paths_str = "\n".join(f"  {p.resolve()}" for p in frame_paths)
    prompt = _MOG_SCORE_PROMPT.format(paths=paths_str)

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = [
        "claude", "-p", prompt,
        "--allowedTools", "Read",
        "--output-format", "json",
        "--no-session-persistence",
        "--model", VISION_MODEL,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "scoring failed"}

    if result.returncode != 0:
        return {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "scoring failed"}

    try:
        wrapper = json.loads(result.stdout)
        raw = wrapper.get("result", "").strip()
    except (json.JSONDecodeError, AttributeError):
        raw = result.stdout.strip()

    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.splitlines() if not line.startswith("```")).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "mog_score" in data:
            score = float(data["mog_score"])
            return {
                "mog_score": round(max(0.0, min(1.0, score)), 3),
                "dominant_trait": str(data.get("dominant_trait", "unknown")),
                "notes": str(data.get("notes", "")),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return {"mog_score": 0.5, "dominant_trait": "unknown", "notes": "parse failed"}
```

### Step 3: Add `classify_mog_track()` helper

After `_call_claude_mog_score()`, add:

```python
# Thresholds
MOG_S_TIER = 0.88    # → good_parts/
MOG_VICTIM = 0.75    # below → victim_contrast/
# between → mid_tier/

def classify_mog_track(mog_score: float) -> str:
    """Return 'good_parts', 'victim_contrast', or 'mid_tier'."""
    if mog_score >= MOG_S_TIER:
        return "good_parts"
    if mog_score < MOG_VICTIM:
        return "victim_contrast"
    return "mid_tier"
```

### Step 4: Add mog-specific tags to prompt vocabulary

In `_PROMPT_TEMPLATE`, extend the tag vocabulary examples line to include:

```
  # S-tier tags (use when mog_score >= 0.88):
  "mog_face_closeup_push", "jawline_pop", "hunter_eyes", "bone_structure_reveal",
  # Victim-contrast tags (use when mog_score < 0.75):
  "victim_mogged_by", "pre_glowup_sideprofile", "average_guy_stare",
  "recessed_chin_reveal", "flat_lighting_cope",
```

Edit the `_PROMPT_TEMPLATE` string to add these after the existing examples.

### Step 5: Update `tag_clips_for_video()` to attach mog_score

In `tag_clips_for_video()` (around line 199), after the line `tags = tag_clip_keyframes(keyframes_dir, dry_run)`, add:

```python
    # Mog scoring — use first batch of scene frames
    mog_result = {"mog_score": 0.5, "dominant_trait": "unknown", "notes": ""}
    if not dry_run:
        scene_frames = _gather_frames(keyframes_dir, max_frames=3)
        if scene_frames:
            mog_result = _call_claude_mog_score(scene_frames)
            log.info("   Mog score: %.3f (%s) — track: %s",
                     mog_result["mog_score"],
                     mog_result["dominant_trait"],
                     classify_mog_track(mog_result["mog_score"]))
```

Then in the `for clip in clips:` loop, also attach:

```python
        clip["mog_score"] = mog_result["mog_score"]
        clip["mog_track"] = classify_mog_track(mog_result["mog_score"])
        clip["mog_notes"] = mog_result.get("notes", "")
```

### Step 6: Verify manually (no automated test needed for CLI subprocess)

Run dry-run to confirm no import errors:
```bash
cd /Users/sebastiandysart/Desktop/ascension-engine/.claude/worktrees/agitated-easley
python -c "import sys; sys.path.insert(0,'data'); from vision_tagger import classify_mog_track; print(classify_mog_track(0.9), classify_mog_track(0.5), classify_mog_track(0.3))"
```
Expected: `good_parts mid_tier victim_contrast`

### Step 7: Commit

```bash
git add data/vision_tagger.py
git commit -m "feat(vision_tagger): add mog scoring (0.0-1.0) + dual-track routing"
```

---

## Task 2: Add routing dirs + victim grade to `data/ingest.py`

**Files:**
- Modify: `data/ingest.py`

### Step 1: Add new path constants

After line 83 (`CLIP_MANIFEST = ROOT / "clip-manifest.json"`), add:

```python
GOOD_PARTS_DIR    = LIBRARY_DIR / "good_parts"
VICTIM_CONTRAST_DIR = LIBRARY_DIR / "victim_contrast"
MID_TIER_DIR      = LIBRARY_DIR / "mid_tier"
```

### Step 2: Add `_import_lut_processor()` lazy import

After `_import_text_processor()` (around line 48), add:

```python
def _import_lut_processor():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from lut_processor import apply_victim_grade
        return apply_victim_grade
    except Exception as e:
        log.warning("lut_processor unavailable: %s", e)
        return None
```

### Step 3: Add `route_and_grade_clips()` function

After `update_tags_index()` (around line 572), add:

```python
def route_and_grade_clips(clips: list[dict], vid_id: str, dry_run: bool) -> dict:
    """
    Copy each clip to its mog-track subdirectory and apply victim grade if needed.
    Returns dict with track distribution counts.
    """
    ensure_dirs(GOOD_PARTS_DIR, VICTIM_CONTRAST_DIR, MID_TIER_DIR)
    _apply_victim_grade = _import_lut_processor()

    counts = {"good_parts": 0, "victim_contrast": 0, "mid_tier": 0}

    for clip in clips:
        track = clip.get("mog_track", "mid_tier")
        src = ROOT / clip["file"]

        if track == "good_parts":
            dest_dir = GOOD_PARTS_DIR
        elif track == "victim_contrast":
            dest_dir = VICTIM_CONTRAST_DIR
        else:
            dest_dir = MID_TIER_DIR

        dest = dest_dir / src.name
        counts[track] = counts.get(track, 0) + 1

        if dry_run:
            log.info("[dry-run] Would copy %s → %s/", src.name, dest_dir.name)
            continue

        if not src.exists():
            log.warning("Clip file missing, skipping route: %s", src)
            continue

        shutil.copy2(src, dest)
        clip["routed_file"] = str(dest.relative_to(ROOT))

        # Apply harsh victim grade
        if track == "victim_contrast" and _apply_victim_grade:
            graded = dest_dir / f"{src.stem}_victim_graded.mp4"
            try:
                _apply_victim_grade(dest, graded, dry_run=False)
                clip["victim_graded_file"] = str(graded.relative_to(ROOT))
                log.info("   Victim grade applied → %s", graded.name)
            except Exception as e:
                log.warning("Victim grade failed for %s: %s", src.name, e)

    log.info("   Mog routing: %d S-tier | %d victim | %d mid",
             counts["good_parts"], counts["victim_contrast"], counts["mid_tier"])
    return counts


def build_mog_track_summary(clips: list[dict]) -> dict:
    """Return per-track counts and clip IDs for the summary report."""
    tracks: dict[str, list[str]] = {"good_parts": [], "victim_contrast": [], "mid_tier": []}
    for clip in clips:
        t = clip.get("mog_track", "mid_tier")
        tracks.setdefault(t, []).append(clip["clip_id"])
    return tracks
```

### Step 4: Wire routing into `ingest_video()` after step 8

After the vision tagging block (after the `log.info("   [8/9]...")` block), add a new step 8b:

```python
    # 8b. Mog-track routing + victim grade
    log.info("   [8b] Routing clips to mog tracks …")
    mog_counts = route_and_grade_clips(clips, vid_id, dry_run)
```

Also update `ensure_dirs()` call at the top of `ingest_video()` to include the new dirs:
```python
    ensure_dirs(CLIPS_DIR, RAW_DIR, ASSETS_VIDEO, ASSETS_AUDIO,
                ASSETS_THUMB, STYLE_PROFILES_DIR, SEQUENCE_TEMPLATES_DIR,
                GOOD_PARTS_DIR, VICTIM_CONTRAST_DIR, MID_TIER_DIR)
```

### Step 5: Update `print_summary()` to show mog track split

In `print_summary()`, add after the `Onsets` line:

```python
    # Mog track distribution (if available)
    tracks = build_mog_track_summary(clips)
    print(f"  S-Tier Clips  : {len(tracks['good_parts'])} → good_parts/")
    print(f"  Victim Clips  : {len(tracks['victim_contrast'])} → victim_contrast/")
    print(f"  Mid-Tier      : {len(tracks['mid_tier'])} → mid_tier/")
```

The function signature needs `clips` — it already receives it, so no change needed there.

### Step 6: Add maintenance commands

In `main()`, add new CLI arguments after `--once`:

```python
    parser.add_argument(
        "--maintenance", metavar="COMMAND",
        choices=["audit-mog", "re-audit-mog", "deduplicate", "prune", "audit"],
        help="Maintenance command: audit-mog | re-audit-mog | deduplicate | prune | audit"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.88,
        help="Mog score threshold for re-audit-mog (default: 0.88)"
    )
    parser.add_argument(
        "--min-rank", type=float, default=0.4,
        help="Minimum rank for prune maintenance command"
    )
    parser.add_argument(
        "--preview-victim", metavar="PATH",
        help="Preview a victim contrast clip (open with ffplay)"
    )
```

Then in the main execution block, before `if args.once:`, add:

```python
    if args.preview_victim:
        _preview_victim(Path(args.preview_victim).expanduser().resolve())
        sys.exit(0)

    if args.maintenance:
        _run_maintenance(args.maintenance, args.threshold, getattr(args, 'min_rank', 0.4))
        sys.exit(0)
```

### Step 7: Add `_run_maintenance()` and `_preview_victim()` functions

Before `main()`, add:

```python
def _run_maintenance(command: str, threshold: float = 0.88, min_rank: float = 0.4) -> None:
    """Execute a library maintenance command."""
    manifest = load_json(CLIP_MANIFEST)
    clips = manifest.get("clips", [])

    if command == "audit-mog":
        tracks: dict[str, list[str]] = {"good_parts": [], "victim_contrast": [], "mid_tier": [], "unscored": []}
        for clip in clips:
            score = clip.get("mog_score")
            if score is None:
                tracks["unscored"].append(clip["clip_id"])
            elif score >= MOG_S_TIER:
                tracks["good_parts"].append(clip["clip_id"])
            elif score < MOG_VICTIM:
                tracks["victim_contrast"].append(clip["clip_id"])
            else:
                tracks["mid_tier"].append(clip["clip_id"])

        print("\n" + "═" * 60)
        print("  MOG TRACK AUDIT")
        print("═" * 60)
        print(f"  Total clips   : {len(clips)}")
        print(f"  S-Tier (≥{MOG_S_TIER}) : {len(tracks['good_parts'])} clips → good_parts/")
        print(f"  Mid   ({MOG_VICTIM}–{MOG_S_TIER}) : {len(tracks['mid_tier'])} clips → mid_tier/")
        print(f"  Victim (<{MOG_VICTIM}) : {len(tracks['victim_contrast'])} clips → victim_contrast/")
        print(f"  Unscored      : {len(tracks['unscored'])} clips (run re-audit-mog)")
        print("═" * 60 + "\n")

    elif command == "re-audit-mog":
        _tagger = _import_vision_tagger()
        if not _tagger:
            log.error("Vision tagger unavailable — cannot re-audit.")
            return
        MOG_S_TIER_local = threshold
        log.info("Re-auditing %d clips with threshold %.2f …", len(clips), threshold)
        changed = 0
        for clip in clips:
            vid_id = clip.get("source_video_id", "")
            kf_dir = ASSETS_THUMB / vid_id
            if not kf_dir.exists():
                continue
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from vision_tagger import _gather_frames, _call_claude_mog_score, classify_mog_track
            frames = _gather_frames(kf_dir, max_frames=3)
            if not frames:
                continue
            result = _call_claude_mog_score(frames)
            old_track = clip.get("mog_track", "unscored")
            new_track = classify_mog_track(result["mog_score"])
            clip["mog_score"] = result["mog_score"]
            clip["mog_track"] = new_track
            clip["mog_notes"] = result.get("notes", "")
            if old_track != new_track:
                changed += 1
                log.info("  %s: %s → %s (%.3f)", clip["clip_id"], old_track, new_track, result["mog_score"])
        manifest["clips"] = clips
        manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_json(CLIP_MANIFEST, manifest)
        log.info("Re-audit complete: %d clips reclassified.", changed)
        _run_maintenance("audit-mog")

    elif command in ("deduplicate", "prune", "audit"):
        log.info("Maintenance command '%s' not yet implemented.", command)


def _preview_victim(path: Path) -> None:
    """Open a victim contrast clip with ffplay for preview."""
    if not path.exists():
        log.error("File not found: %s", path)
        return
    ffplay = shutil.which("ffplay")
    if not ffplay:
        log.error("ffplay not found (install ffmpeg).")
        return
    log.info("Previewing victim clip: %s", path.name)
    subprocess.run([ffplay, "-autoexit", str(path)], check=False)
```

Also add imports at top of `_run_maintenance`:
```python
# (MOG_S_TIER and MOG_VICTIM are imported from vision_tagger at call time —
# define local fallbacks at module level)
MOG_S_TIER = 0.88
MOG_VICTIM = 0.75
```
Add these two constants after the `CLIP_MANIFEST` path constant (before the logging setup).

### Step 8: Verify no import errors

```bash
python -c "
import sys; sys.path.insert(0,'data')
# Just parse imports
import ast, pathlib
src = pathlib.Path('data/ingest.py').read_text()
ast.parse(src)
print('Parse OK')
"
```
Expected: `Parse OK`

### Step 9: Commit

```bash
git add data/ingest.py
git commit -m "feat(ingest): add mog-track routing to good_parts/ victim_contrast/ mid_tier/"
```

---

## Task 3: Add `apply_victim_grade()` to `data/lut_processor.py`

**Files:**
- Modify: `data/lut_processor.py`

### Step 1: Add `apply_victim_grade()` function

After `process_single()` (before the `# ── CLI` comment), add:

```python
def apply_victim_grade(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> bool:
    """
    Apply harsh victim-contrast grade via FFmpeg:
    - Crush blacks further (-40% brightness in shadows)
    - Desaturate skin tones (-25% saturation)
    - Harsher shadows (contrast boost)

    Uses FFmpeg eq + hue filters (no external LUT needed).
    Returns True on success.
    """
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found")
        return False

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return False

    # eq: contrast=1.35 (crushes blacks), brightness=-0.07 (darker overall),
    #     saturation=0.72 (desaturated skin, -28%)
    # hue: s=0.75 (additional skin desaturation via hue filter)
    victim_filter = (
        "eq=contrast=1.35:brightness=-0.07:saturation=0.72,"
        "hue=s=0.75"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", victim_filter,
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]

    if dry_run:
        log.info("[dry-run] Would apply victim grade: %s → %s", input_path.name, output_path.name)
        return True

    log.info("   Applying victim grade: %s → %s", input_path.name, output_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg victim grade failed: %s", e.stderr[-500:] if e.stderr else e)
        return False
```

### Step 2: Verify import

```bash
python -c "
import sys; sys.path.insert(0,'data')
from lut_processor import apply_victim_grade
print('import OK:', apply_victim_grade)
"
```
Expected: `import OK: <function apply_victim_grade ...>`

### Step 3: Commit

```bash
git add data/lut_processor.py
git commit -m "feat(lut_processor): add apply_victim_grade() — crushed blacks + desaturated skin"
```

---

## Task 4: Update `src/lib/clipLibrary.ts`

**Files:**
- Modify: `src/lib/clipLibrary.ts`

### Step 1: Extend `Clip` interface

In the `Clip` interface (around line 22), add after `ingest_timestamp`:

```ts
  /** Mog score [0.0–1.0]: face dominance rating. 0.5 = unscored/no face */
  mog_score?: number;
  /** Routing track: 'good_parts' | 'victim_contrast' | 'mid_tier' */
  mog_track?: 'good_parts' | 'victim_contrast' | 'mid_tier';
  /** Path to mog-track-routed copy of the clip file */
  routed_file?: string;
  /** Path to victim-graded version (only on victim_contrast clips) */
  victim_graded_file?: string;
```

### Step 2: Add `SequenceTemplate` interface and `ContrastSlot` type

After the `TimelineBlueprint` interface (after line 67), add:

```ts
/** A single slot in a sequence template timeline */
export interface SequenceSlot {
  index: number;
  start_sec: number;
  duration_sec: number;
  on_beat: boolean;
  is_impact: boolean;
  original_clip_id: string;
  preferred_tags: string[];
}

/** Optional victim-contrast insertion point within a sequence */
export interface ContrastSlot {
  type: 'victim_contrast';
  /** Maximum clip duration to use from victim_contrast/ library */
  max_duration: number;
  /** Where in the sequence this contrast moment lands */
  position: 'before_phonk_drop' | 'after_hook' | 'mid_sequence' | string;
  /** Beat-anchor in seconds (snap to nearest beat when inserting) */
  beat_anchor_sec?: number;
}

/** A full sequence template — reusable timeline DNA extracted from a gold video */
export interface SequenceTemplate {
  template_id: string;
  source_video_id: string;
  source_creator: string;
  bpm: number;
  total_duration_sec: number;
  total_slots: number;
  avg_cut_sec: number;
  cuts_on_beat_pct: number;
  color_grade: string;
  slots: SequenceSlot[];
  /** Optional victim-contrast insertion points */
  contrast_slots?: ContrastSlot[];
  created_at: string;
}
```

### Step 3: Add path constants for new dirs

After `const SEQUENCE_DIR` (around line 346), add:

```ts
const GOOD_PARTS_DIR     = path.join(ROOT, 'library', 'good_parts');
const VICTIM_CONTRAST_DIR = path.join(ROOT, 'library', 'victim_contrast');
```

### Step 4: Add `getMogClips()`

After `getTopClips()` (after line 151), add:

```ts
/**
 * Return S-Tier mog clips (mog_score ≥ minMogScore) matching any of the given tags.
 * Falls back to all good_parts-track clips if no tag match is found.
 *
 * @param tags        - Semantic tags to filter by (OR logic).
 * @param count       - Maximum clips to return.
 * @param minMogScore - Minimum mog_score threshold (default: 0.88).
 */
export function getMogClips(
  tags: string[],
  count: number,
  minMogScore = 0.88,
): Clip[] {
  const tagSet = new Set(tags);
  const clips = loadManifest().clips.filter(
    (c) =>
      (c.mog_score ?? 0) >= minMogScore &&
      (tagSet.size === 0 || c.tags.some((t) => tagSet.has(t))),
  );

  // Fallback: if no tag match, return all S-tier clips
  const pool = clips.length > 0
    ? clips
    : loadManifest().clips.filter((c) => (c.mog_score ?? 0) >= minMogScore);

  return pool.sort((a, b) => (b.mog_score ?? 0) - (a.mog_score ?? 0)).slice(0, count);
}

/**
 * Return victim-contrast clips (mog_score < 0.75) up to maxDuration.
 * Sorted by mog_score ascending (weakest face first).
 *
 * @param count       - Maximum clips to return.
 * @param maxDuration - Maximum clip duration in seconds (default: 0.8).
 */
export function getVictimContrastClips(
  count: number,
  maxDuration = 0.8,
): Clip[] {
  return loadManifest()
    .clips
    .filter(
      (c) =>
        (c.mog_score ?? 0.5) < 0.75 &&
        c.duration_sec <= maxDuration,
    )
    .sort((a, b) => (a.mog_score ?? 0.5) - (b.mog_score ?? 0.5))
    .slice(0, count);
}

/**
 * Build a remixed timeline that optionally injects a victim-contrast clip
 * before the phonk drop (or any contrast_slot in the template).
 *
 * @param baseTemplate - A SequenceTemplate to remix from.
 * @param victimClip   - Optional victim clip to inject at contrast slots.
 * @param fps          - Frames per second (default: 30).
 */
export function remixWithContrast(
  baseTemplate: SequenceTemplate,
  victimClip?: Clip,
  fps = 30,
): BPTimelineEntry[] {
  const allClips = getAllClips();
  if (allClips.length === 0) return [];

  // Build shuffled pool of S-tier clips (mog_score >= 0.88)
  const mogPool = allClips
    .filter((c) => (c.mog_score ?? 0.5) >= 0.88)
    .sort(() => Math.random() - 0.5);
  const fallbackPool = allClips.sort(() => Math.random() - 0.5);
  const pool = mogPool.length > 0 ? mogPool : fallbackPool;

  // Identify contrast slot positions (by beat_anchor_sec if present)
  const contrastPositions = new Set<number>(
    (baseTemplate.contrast_slots ?? [])
      .map((cs) => cs.beat_anchor_sec ?? -1)
      .filter((t) => t >= 0),
  );

  const timeline: BPTimelineEntry[] = [];

  for (let i = 0; i < baseTemplate.slots.length; i++) {
    const slot = baseTemplate.slots[i];

    // Check if this is a contrast injection point
    const isContrast =
      victimClip !== undefined &&
      (contrastPositions.has(slot.start_sec) ||
        (baseTemplate.contrast_slots ?? []).some(
          (cs) =>
            cs.position === 'before_phonk_drop' &&
            slot.is_impact &&
            i === 0,
        ));

    const clip = isContrast ? victimClip! : pool[i % pool.length];
    const durationFrames = Math.round(slot.duration_sec * fps);
    const clipMaxFrames = Math.round(clip.duration_sec * fps);
    const actualDuration = Math.min(durationFrames, clipMaxFrames);
    const maxTrimStart = Math.max(0, clipMaxFrames - actualDuration);
    const trimStart = maxTrimStart > 0 ? Math.floor(Math.random() * maxTrimStart) : 0;

    timeline.push({
      clip,
      startFrame: Math.round(slot.start_sec * fps),
      durationFrames: actualDuration,
      trimStartFrames: trimStart,
      on_beat: slot.on_beat,
      is_impact: slot.is_impact,
    });
  }

  return timeline;
}
```

### Step 5: Verify TypeScript compiles

```bash
cd /Users/sebastiandysart/Desktop/ascension-engine/.claude/worktrees/agitated-easley
npx tsc --noEmit 2>&1 | head -30
```
Expected: no errors (or only pre-existing errors unrelated to our changes).

### Step 6: Commit

```bash
git add src/lib/clipLibrary.ts
git commit -m "feat(clipLibrary): add getMogClips, getVictimContrastClips, remixWithContrast + SequenceTemplate interface"
```

---

## Task 5: Run re-audit on existing library + report split

**Files:**
- No code changes — run maintenance command

### Step 1: Run audit-mog first (dry-run stats on unscored clips)

```bash
cd /Users/sebastiandysart/Desktop/ascension-engine/.claude/worktrees/agitated-easley
python data/ingest.py --maintenance audit-mog
```
Expected: shows how many clips are unscored.

### Step 2: Run re-audit-mog to classify everything

```bash
python data/ingest.py --maintenance re-audit-mog --threshold 0.88
```
This calls Claude vision on each clip's keyframes. Will take ~2–5 min for 34 clips.

### Step 3: Run audit-mog again to show final split

```bash
python data/ingest.py --maintenance audit-mog
```
Record the output.

### Step 4: Commit manifest with mog scores

```bash
git add clip-manifest.json library/
git commit -m "chore: re-audit mog scores on 34-clip library — dual-track classification complete"
```

---

## Task 6: Update README.md

**Files:**
- Modify: `README.md`

### Step 1: Add dual-track section to README

In `README.md`, after the `### How the DNA Library Works` section and before `### Generation Flow`, add:

```markdown
### Dual-Track System: Mogger vs Victim

Every clip is scored 0.0–1.0 for face dominance during ingest. The library auto-routes:

| Score | Track | Directory | Usage |
|-------|-------|-----------|-------|
| ≥ 0.88 | **S-Tier** | `library/good_parts/` | 90%+ of every sequence |
| 0.75–0.87 | Mid-Tier | `library/mid_tier/` | Rarely used |
| < 0.75 | **Victim Contrast** | `library/victim_contrast/` | Contrast slots only |

**Victim clips** get an extra harsh FFmpeg grade automatically: blacks crushed further, skin desaturated -25%, shadows strengthened.

**Sequence templates** can include optional contrast slot entries:
```json
{ "type": "victim_contrast", "max_duration": 0.8, "position": "before_phonk_drop" }
```

**TypeScript helpers:**
```ts
getMogClips(tags, count, minMogScore?)      // S-tier clips ≥ threshold
getVictimContrastClips(count, maxDuration?) // victim clips ≤ 0.8s
remixWithContrast(template, victimClip?)    // injects victim at contrast slots
```

**Maintenance:**
```bash
python data/ingest.py --maintenance audit-mog                      # show split stats
python data/ingest.py --maintenance re-audit-mog --threshold 0.88  # re-classify whole library
python data/ingest.py --preview-victim library/victim_contrast/x.mp4
```
```

### Step 2: Commit

```bash
git add README.md
git commit -m "docs: document dual-track mogger/victim system in README"
```

---

## Final: Push all commits

```bash
git push origin HEAD:main
git update-ref refs/heads/main HEAD
```

---

## Quick Reference — Key Constants

| Constant | Value | Where |
|----------|-------|-------|
| `MOG_S_TIER` | 0.88 | `vision_tagger.py` + `ingest.py` |
| `MOG_VICTIM` | 0.75 | `vision_tagger.py` + `ingest.py` |
| `GOOD_PARTS_DIR` | `library/good_parts/` | `ingest.py` |
| `VICTIM_CONTRAST_DIR` | `library/victim_contrast/` | `ingest.py` |
| `MID_TIER_DIR` | `library/mid_tier/` | `ingest.py` |
| victim grade filter | `eq=contrast=1.35:brightness=-0.07:saturation=0.72,hue=s=0.75` | `lut_processor.py` |
