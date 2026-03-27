/**
 * Ascension Engine v2.1 — Clip Library
 * ──────────────────────────────────────────────────────────────────────────────
 * Pure TypeScript module (no React). Reads from library/clips/ and
 * clip-manifest.json to provide structured access to the gold clip catalog.
 *
 * Usage:
 *   import { getClipsByTags, getBlueprintFromReference } from './lib/clipLibrary';
 */

import fs from 'fs';
import path from 'path';

// ── Root resolution ──────────────────────────────────────────────────────────
const ROOT = path.resolve(__dirname, '..', '..');
const MANIFEST_PATH = path.join(ROOT, 'clip-manifest.json');
const CLIPS_DIR = path.join(ROOT, 'library', 'clips');

// ── Interfaces ────────────────────────────────────────────────────────────────

/** A single extracted clip from the gold video library. */
export interface Clip {
  /** Unique identifier: {videoId}_scene{index} */
  clip_id: string;
  /** Parent video ID this clip was extracted from */
  source_video_id: string;
  /** Zero-based scene index within the source video */
  scene_index: number;
  /** Start time in the source video (seconds) */
  start_sec: number;
  /** End time in the source video (seconds) */
  end_sec: number;
  /** Duration of this clip (seconds) */
  duration_sec: number;
  /** Relative path to the .mp4 file */
  file: string;
  /** Relative path to the thumbnail .jpg */
  thumbnail: string;
  /** Semantic tags applied during ingest or manual review */
  tags: string[];
  /** Quality/relevance rank [0.0–1.0]; 0.5 = unrated */
  rank: number;
  /** ISO 8601 timestamp of when this clip was ingested */
  ingest_timestamp: string;
}

/** A complete timeline blueprint derived from a reference video's style profile. */
export interface TimelineBlueprint {
  /** Source video ID */
  video_id: string;
  /** Ordered list of recommended clip IDs for this blueprint */
  clip_sequence: string[];
  /** Target cut rate in seconds */
  cut_rate_sec: number;
  /** Detected color grade preset */
  color_grade: string;
  /** Detected BPM */
  bpm: number;
  /** Beat-cut alignment descriptor */
  beat_cut_alignment: string;
  /** Peak energy moments for sync points */
  peak_moments_sec: number[];
  /** Hook window [startSec, endSec] */
  hook_window_sec: [number, number];
  /** Raw style profile for full access */
  style_profile: Record<string, unknown>;
}

// ── Manifest cache ────────────────────────────────────────────────────────────

interface Manifest {
  version: string;
  clips: Clip[];
  last_updated: string;
  total_clips: number;
}

let _cache: Manifest | null = null;

/** Load and cache clip-manifest.json. Re-reads on every process restart. */
function loadManifest(): Manifest {
  if (_cache) return _cache;
  if (!fs.existsSync(MANIFEST_PATH)) {
    _cache = { version: '1.0', clips: [], last_updated: '', total_clips: 0 };
    return _cache;
  }
  _cache = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8')) as Manifest;
  return _cache;
}

/** Invalidate the in-memory cache (call after writing to manifest). */
export function invalidateCache(): void {
  _cache = null;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Return all clips in the library.
 *
 * @returns Full array of Clip objects from clip-manifest.json.
 */
export function getAllClips(): Clip[] {
  return loadManifest().clips;
}

/**
 * Filter clips by one or more tags, optionally requiring a minimum rank.
 *
 * @param tags     - Array of tag strings (OR logic — clip must have at least one).
 * @param count    - Maximum number of clips to return.
 * @param minRank  - Minimum rank threshold [0.0–1.0]. Defaults to 0.
 * @returns Clips sorted by rank descending, up to `count` results.
 *
 * @example
 *   const clips = getClipsByTags(['face_closeup', 'glow_up_reveal'], 6, 0.6);
 */
export function getClipsByTags(
  tags: string[],
  count: number,
  minRank = 0,
): Clip[] {
  const tagSet = new Set(tags);
  return loadManifest()
    .clips
    .filter(
      (c) =>
        c.rank >= minRank &&
        c.tags.some((t) => tagSet.has(t)),
    )
    .sort((a, b) => b.rank - a.rank)
    .slice(0, count);
}

/**
 * Return the top-ranked clips regardless of tag.
 *
 * @param count    - Maximum number of clips to return.
 * @param minRank  - Minimum rank threshold [0.0–1.0]. Defaults to 0.
 * @returns Clips sorted by rank descending.
 *
 * @example
 *   const best = getTopClips(10, 0.7);
 */
export function getTopClips(count: number, minRank = 0): Clip[] {
  return loadManifest()
    .clips
    .filter((c) => c.rank >= minRank)
    .sort((a, b) => b.rank - a.rank)
    .slice(0, count);
}

/**
 * Build a TimelineBlueprint from a reference video's style-profile.json.
 * Assembles clip_sequence from library clips that match the source video.
 *
 * @param videoId - The video_id string (matches style-profiles/{videoId}.json).
 * @returns A TimelineBlueprint ready for use in Remotion compositions.
 * @throws If the style profile file is not found.
 *
 * @example
 *   const blueprint = getBlueprintFromReference('jordan_abc12345');
 */
export function getBlueprintFromReference(videoId: string): TimelineBlueprint {
  const profilePath = path.join(ROOT, 'style-profiles', `${videoId}.json`);
  if (!fs.existsSync(profilePath)) {
    throw new Error(`Style profile not found: ${profilePath}`);
  }

  const profile = JSON.parse(fs.readFileSync(profilePath, 'utf-8'));
  const manifest = loadManifest();

  // Collect all clips from this source video, ordered by scene_index
  const sourceClips = manifest.clips
    .filter((c) => c.source_video_id === videoId)
    .sort((a, b) => a.scene_index - b.scene_index);

  const cutRate: number =
    (profile?.cut_rhythm?.avg_cut_length_sec as number) ?? 2.0;

  return {
    video_id: videoId,
    clip_sequence: sourceClips.map((c) => c.clip_id),
    cut_rate_sec: cutRate,
    color_grade: (profile?.visuals?.color_grade as string) ?? 'Neutral',
    bpm: (profile?.audio?.bpm as number) ?? 0,
    beat_cut_alignment: (profile?.audio?.beat_cut_alignment as string) ?? 'free',
    peak_moments_sec: (profile?.audio?.peak_moments_sec as number[]) ?? [],
    hook_window_sec: (profile?.cut_rhythm?.hook_pacing_sec as [number, number]) ?? [0, 3],
    style_profile: profile as Record<string, unknown>,
  };
}

/**
 * Adjust the rank of a clip by a delta value, clamped to [0.0, 1.0].
 * Persists the change directly to clip-manifest.json.
 *
 * @param clipId    - The clip_id to update.
 * @param rankDelta - Amount to add to current rank (negative to decrease).
 * @throws If clip_id is not found in the manifest.
 *
 * @example
 *   updateClipRank('jordan_abc12345_scene003', +0.1);
 */
export function updateClipRank(clipId: string, rankDelta: number): void {
  const manifest = loadManifest();
  const clip = manifest.clips.find((c) => c.clip_id === clipId);
  if (!clip) {
    throw new Error(`Clip not found: ${clipId}`);
  }
  clip.rank = Math.max(0, Math.min(1, clip.rank + rankDelta));
  manifest.last_updated = new Date().toISOString();

  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
  invalidateCache();
}

// ── Beat-Synced Timeline Builder ─────────────────────────────────────────────

interface BPBlueprintInput {
  bpm: number;
  beat_times_sec: number[];
  cut_rhythm: {
    avg_cut_sec: number;
    cut_points?: { time_sec: number; duration_sec: number; on_beat: boolean }[];
  };
  audio?: {
    peak_moments_sec?: number[];
    onset_times_sec?: number[];
  };
}

interface BPTimelineEntry {
  clip: Clip;
  startFrame: number;
  durationFrames: number;
  trimStartFrames: number;
  on_beat: boolean;
  is_impact: boolean;
}

/**
 * Build a beat-synced timeline from a brutal BP blueprint.
 * Cuts land on beat times from the blueprint. Clips are pulled from the
 * library and interleaved across sources for variety.
 *
 * @param blueprint - A brutal BP blueprint (from style-profiles/*.json)
 * @param fps - Frames per second (default 30)
 * @param totalDurationSec - Target total duration (default 15)
 */
export function getBrutalTimeline(
  blueprint: BPBlueprintInput,
  fps = 30,
  totalDurationSec = 15,
): BPTimelineEntry[] {
  const allClips = getAllClips();
  if (allClips.length === 0) return [];

  const totalFrames = fps * totalDurationSec;
  const beatTimes = blueprint.beat_times_sec.filter(t => t < totalDurationSec);
  const peakMoments = new Set(
    (blueprint.audio?.peak_moments_sec || [])
      .filter(t => t < totalDurationSec)
      .map(t => Math.round(t * 10) / 10)
  );

  // Build cut points from beat times
  let cutTimes: number[];
  if (beatTimes.length >= 4) {
    // Use every Nth beat to get ~avg_cut_sec spacing
    const targetCutSec = blueprint.cut_rhythm.avg_cut_sec || 1.0;
    const beatInterval = beatTimes.length > 1
      ? (beatTimes[beatTimes.length - 1] - beatTimes[0]) / (beatTimes.length - 1)
      : 0.5;
    const beatsPerCut = Math.max(1, Math.round(targetCutSec / beatInterval));
    cutTimes = [0];
    for (let i = beatsPerCut; i < beatTimes.length; i += beatsPerCut) {
      if (beatTimes[i] < totalDurationSec) {
        cutTimes.push(beatTimes[i]);
      }
    }
  } else {
    // Fallback: fixed intervals at avg_cut_sec
    const cutSec = blueprint.cut_rhythm.avg_cut_sec || 1.0;
    cutTimes = [];
    for (let t = 0; t < totalDurationSec; t += cutSec) {
      cutTimes.push(Math.round(t * 1000) / 1000);
    }
  }

  // Interleave clips from different sources
  const bySource = new Map<string, Clip[]>();
  for (const c of allClips) {
    const src = c.source_video_id;
    if (!bySource.has(src)) bySource.set(src, []);
    bySource.get(src)!.push(c);
  }
  const interleavedPool: Clip[] = [];
  const sources = [...bySource.values()];
  const maxLen = Math.max(...sources.map(s => s.length));
  for (let i = 0; i < maxLen; i++) {
    for (const s of sources) {
      if (i < s.length) interleavedPool.push(s[i]);
    }
  }

  // Build timeline entries
  const timeline: BPTimelineEntry[] = [];
  for (let i = 0; i < cutTimes.length; i++) {
    const startSec = cutTimes[i];
    const endSec = i + 1 < cutTimes.length ? cutTimes[i + 1] : totalDurationSec;
    const durSec = endSec - startSec;
    if (durSec < 0.15) continue;

    const startFrame = Math.round(startSec * fps);
    const durationFrames = Math.round(durSec * fps);
    if (startFrame >= totalFrames) break;

    const clip = interleavedPool[i % interleavedPool.length];
    const clipMaxFrames = Math.round(clip.duration_sec * fps);
    // Clamp segment to actual clip length
    const actualDuration = Math.min(durationFrames, clipMaxFrames);
    const maxTrimStart = Math.max(0, clipMaxFrames - actualDuration);
    const trimStart = maxTrimStart > 0 ? Math.floor(Math.random() * maxTrimStart) : 0;

    // Check if this cut is near a peak energy moment (= impact frame)
    const isImpact = [...peakMoments].some(
      p => Math.abs(p - startSec) < 0.2
    );

    timeline.push({
      clip,
      startFrame,
      durationFrames: Math.min(actualDuration, totalFrames - startFrame),
      trimStartFrames: trimStart,
      on_beat: beatTimes.some(bt => Math.abs(bt - startSec) < 0.08),
      is_impact: isImpact,
    });
  }

  return timeline;
}

// ── Sequence Template Remix ──────────────────────────────────────────────────

const SEQUENCE_DIR = path.join(ROOT, 'library', 'sequence_templates');

export function listSequenceTemplates(): string[] {
  if (!fs.existsSync(SEQUENCE_DIR)) return [];
  return fs.readdirSync(SEQUENCE_DIR)
    .filter(f => f.endsWith('.json'))
    .map(f => f.replace('.json', ''));
}

export function remixSequence(templateId: string, fps = 30): BPTimelineEntry[] {
  const filePath = path.join(SEQUENCE_DIR, `${templateId}.json`);
  if (!fs.existsSync(filePath)) throw new Error(`Template not found: ${templateId}`);
  const template = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

  const allClips = getAllClips();
  if (allClips.length === 0) return [];

  const bySource = new Map<string, Clip[]>();
  for (const c of allClips) {
    if (!bySource.has(c.source_video_id)) bySource.set(c.source_video_id, []);
    bySource.get(c.source_video_id)!.push(c);
  }
  const pool: Clip[] = [];
  const sources = [...bySource.values()];
  const maxLen = Math.max(...sources.map(s => s.length));
  for (let i = 0; i < maxLen; i++) {
    for (const s of sources) {
      if (i < s.length) pool.push(s[i]);
    }
  }
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }

  const timeline: BPTimelineEntry[] = [];
  for (let i = 0; i < template.slots.length; i++) {
    const slot = template.slots[i];
    const clip = pool[i % pool.length];
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
