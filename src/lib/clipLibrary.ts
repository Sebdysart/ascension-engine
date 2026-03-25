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
