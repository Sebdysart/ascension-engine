/**
 * Ascension Engine v2.1 — Remotion Presets
 * ──────────────────────────────────────────────────────────────────────────────
 * Central config for all archetype-specific rendering parameters.
 * Import these into your Remotion compositions to ensure DNA-accurate output.
 *
 * Usage:
 *   import { getPresetByArchetype, PRESETS } from '../remotion/remotion-presets';
 *   const preset = getPresetByArchetype('GlowUp');
 */

import fs from 'fs';
import path from 'path';

// ── Color Grade Enum ──────────────────────────────────────────────────────────

export enum ColorGrade {
  TealOrange  = 'TealOrange',
  ColdBlue    = 'ColdBlue',
  WarmGold    = 'WarmGold',
  Desaturated = 'Desaturated',
}

// ── Hook Type Enum ────────────────────────────────────────────────────────────

export enum HookType {
  FaceReveal      = 'FaceReveal',
  TextBomb        = 'TextBomb',
  BeforeAfterCut  = 'BeforeAfterCut',
  SilentStare     = 'SilentStare',
  AggressiveOpen  = 'AggressiveOpen',
  TransformFreeze = 'TransformFreeze',
}

// ── Caption Position ──────────────────────────────────────────────────────────

export enum CaptionPosition {
  BottomThird = 'bottom-third',
  Center      = 'center',
  TopThird    = 'top-third',
}

// ── Archetype Preset Interface ────────────────────────────────────────────────

export interface ArchetypePreset {
  /** Archetype name identifier */
  name: string;
  /** Target color grade to apply in post */
  colorGrade: ColorGrade;
  /** Target avg cut rate in seconds */
  cutRateSec: number;
  /** Font style identifier for captions */
  fontStyle: 'DrukWide' | 'Bebas' | 'Monument' | 'Inter' | 'PPNeueMontreal';
  /** Caption vertical position */
  captionPosition: CaptionPosition;
  /** Acceptable BPM range [min, max] for music matching */
  musicBpmRange: [number, number];
  /** Default hook strategy */
  hookType: HookType;
  /**
   * Exploit weight [0.0–1.0]: how aggressively to lean into platform-exploit
   * patterns (cuts on beat, zoom punches, text animations on peaks).
   */
  exploitWeight: number;
  /** Caption max characters per line */
  captionMaxCharsPerLine: number;
  /** Whether to use aggressive zoom-punch transitions */
  zoomPunch: boolean;
  /** Slow-motion factor for reveal moments (1 = no slo-mo) */
  revealSlowFactor: number;
  /** Recommended clip tags to pull from library */
  preferredTags: string[];
}

// ── Archetype Presets ─────────────────────────────────────────────────────────

export const PRESETS: Record<string, ArchetypePreset> = {

  GlowUp: {
    name:               'GlowUp',
    colorGrade:         ColorGrade.WarmGold,
    cutRateSec:         2.2,
    fontStyle:          'DrukWide',
    captionPosition:    CaptionPosition.BottomThird,
    musicBpmRange:      [120, 145],
    hookType:           HookType.BeforeAfterCut,
    exploitWeight:      0.75,
    captionMaxCharsPerLine: 18,
    zoomPunch:          true,
    revealSlowFactor:   0.5,
    preferredTags:      ['transformation_reveal', 'before_after', 'glow_up_reveal', 'face_closeup'],
  },

  FrameMaxxing: {
    name:               'FrameMaxxing',
    colorGrade:         ColorGrade.TealOrange,
    cutRateSec:         1.6,
    fontStyle:          'Bebas',
    captionPosition:    CaptionPosition.BottomThird,
    musicBpmRange:      [135, 165],
    hookType:           HookType.FaceReveal,
    exploitWeight:      0.85,
    captionMaxCharsPerLine: 16,
    zoomPunch:          true,
    revealSlowFactor:   1.0,
    preferredTags:      ['face_closeup', 'high_energy', 'aggressive_ascension', 'gym_lift'],
  },

  SkinMaxxing: {
    name:               'SkinMaxxing',
    colorGrade:         ColorGrade.ColdBlue,
    cutRateSec:         3.0,
    fontStyle:          'PPNeueMontreal',
    captionPosition:    CaptionPosition.BottomThird,
    musicBpmRange:      [90, 120],
    hookType:           HookType.FaceReveal,
    exploitWeight:      0.6,
    captionMaxCharsPerLine: 20,
    zoomPunch:          false,
    revealSlowFactor:   0.7,
    preferredTags:      ['face_closeup', 'transformation_reveal', 'slow_motion', 'glow_up_reveal'],
  },

  StyleMaxxing: {
    name:               'StyleMaxxing',
    colorGrade:         ColorGrade.Desaturated,
    cutRateSec:         2.0,
    fontStyle:          'Monument',
    captionPosition:    CaptionPosition.BottomThird,
    musicBpmRange:      [105, 135],
    hookType:           HookType.TransformFreeze,
    exploitWeight:      0.7,
    captionMaxCharsPerLine: 18,
    zoomPunch:          true,
    revealSlowFactor:   0.6,
    preferredTags:      ['transformation_reveal', 'before_after', 'text_punch', 'slow_motion'],
  },

};

// ── Runtime Style Profile Override ───────────────────────────────────────────

interface StyleProfileOverride {
  visuals?: { color_grade?: string };
  cut_rhythm?: { avg_cut_length_sec?: number };
  audio?: { bpm?: number };
}

/**
 * Load a style-profile.json override for a given video ID.
 * Returns null if the file doesn't exist.
 */
function loadStyleProfileOverride(videoId: string): StyleProfileOverride | null {
  const profilePath = path.join(__dirname, '..', 'style-profiles', `${videoId}.json`);
  if (!fs.existsSync(profilePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(profilePath, 'utf-8')) as StyleProfileOverride;
  } catch {
    return null;
  }
}

/**
 * Map a style-profile color_grade string to the ColorGrade enum.
 */
function resolveColorGrade(grade: string | undefined): ColorGrade {
  switch (grade) {
    case 'TealOrange':  return ColorGrade.TealOrange;
    case 'ColdBlue':    return ColorGrade.ColdBlue;
    case 'WarmGold':    return ColorGrade.WarmGold;
    case 'Desaturated': return ColorGrade.Desaturated;
    default:            return ColorGrade.TealOrange;
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Return the preset for a given archetype name.
 * Optionally merges overrides from a style-profile.json if videoId is supplied.
 *
 * @param archetype - One of: GlowUp, FrameMaxxing, SkinMaxxing, StyleMaxxing
 * @param videoId   - Optional video ID to load style-profile overrides from.
 * @returns The resolved ArchetypePreset (base + any profile overrides).
 * @throws If archetype is not recognized.
 *
 * @example
 *   const preset = getPresetByArchetype('GlowUp');
 *   const presetWithOverrides = getPresetByArchetype('GlowUp', 'jordan_abc12345');
 */
export function getPresetByArchetype(
  archetype: string,
  videoId?: string,
): ArchetypePreset {
  const base = PRESETS[archetype];
  if (!base) {
    throw new Error(
      `Unknown archetype: "${archetype}". Valid options: ${Object.keys(PRESETS).join(', ')}`,
    );
  }

  if (!videoId) return { ...base };

  const override = loadStyleProfileOverride(videoId);
  if (!override) return { ...base };

  return {
    ...base,
    colorGrade: resolveColorGrade(override.visuals?.color_grade) ?? base.colorGrade,
    cutRateSec: override.cut_rhythm?.avg_cut_length_sec ?? base.cutRateSec,
  };
}

/**
 * Return all available archetype names.
 */
export function listArchetypes(): string[] {
  return Object.keys(PRESETS);
}

/**
 * Return a preset config object formatted for direct use as Remotion props.
 * Flattens the preset into a flat record compatible with Remotion's inputProps.
 */
export function presetToRemotionProps(preset: ArchetypePreset): Record<string, unknown> {
  return {
    colorGrade:             preset.colorGrade,
    cutRateSec:             preset.cutRateSec,
    fontStyle:              preset.fontStyle,
    captionPosition:        preset.captionPosition,
    musicBpmMin:            preset.musicBpmRange[0],
    musicBpmMax:            preset.musicBpmRange[1],
    hookType:               preset.hookType,
    exploitWeight:          preset.exploitWeight,
    captionMaxCharsPerLine: preset.captionMaxCharsPerLine,
    zoomPunch:              preset.zoomPunch,
    revealSlowFactor:       preset.revealSlowFactor,
    preferredTags:          preset.preferredTags,
  };
}
