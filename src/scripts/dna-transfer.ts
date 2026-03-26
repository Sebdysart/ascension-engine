#!/usr/bin/env npx ts-node
/**
 * Ascension Engine v3.0 — DNA Transfer Script
 * ──────────────────────────────────────────────────────────────────────────────
 * Generates beat-synced BP/mog edit compositions from the clip library.
 * Pulls clips from multiple source videos, builds rapid-cut timelines
 * matching the DNA of your gold references.
 *
 * Usage:
 *   npx ts-node src/scripts/dna-transfer.ts --style dark_cinema --cut-rate 1.2 --render
 *   npx ts-node src/scripts/dna-transfer.ts --style desaturated --clip-count 12 --render
 *   npx ts-node src/scripts/dna-transfer.ts --batch 4 --render
 *   npx ts-node src/scripts/dna-transfer.ts --list-styles
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import {
  getAllClips,
  invalidateCache,
} from '../lib/clipLibrary';
import type { Clip } from '../lib/clipLibrary';

// ── Constants ────────────────────────────────────────────────────────────────

const ROOT = path.resolve(__dirname, '..', '..');
const OUTPUT_DIR = path.join(ROOT, 'out');
const FPS = 30;
const TOTAL_DURATION_SEC = 15;
const TOTAL_FRAMES = FPS * TOTAL_DURATION_SEC;

// ── Edit styles (derived from gold DNA) ──────────────────────────────────────

interface EditStyle {
  name: string;
  colorGrade: string;
  cutRateSec: number;
  zoomPunch: boolean;
  description: string;
}

const STYLES: Record<string, EditStyle> = {
  dark_cinema: {
    name: "dark_cinema",
    colorGrade: "dark_cinema",
    cutRateSec: 1.0,
    zoomPunch: true,
    description: "Dark cinema mog edit. Near-black bg, warm skin tones. @roge.editz @morphedmanlet style.",
  },
  desaturated: {
    name: "desaturated",
    colorGrade: "desaturated",
    cutRateSec: 1.2,
    zoomPunch: true,
    description: "Desaturated B&W. @black.pill.city podcast/analysis style.",
  },
  cold_blue: {
    name: "cold_blue",
    colorGrade: "cold_blue",
    cutRateSec: 1.5,
    zoomPunch: true,
    description: "Cold blue skin/frame analysis style.",
  },
  warm_ambient: {
    name: "warm_ambient",
    colorGrade: "warm_ambient",
    cutRateSec: 1.0,
    zoomPunch: true,
    description: "Warm ambient social dynamics. @unc_iiris style.",
  },
  natural: {
    name: "natural",
    colorGrade: "natural",
    cutRateSec: 1.4,
    zoomPunch: false,
    description: "Natural minimal grade. Raw selfie/authentic feel.",
  },
  teal_orange: {
    name: "teal_orange",
    colorGrade: "teal_orange",
    cutRateSec: 0.8,
    zoomPunch: true,
    description: "High contrast teal/orange. Fast aggressive cuts.",
  },
};

// ── Config ───────────────────────────────────────────────────────────────────

interface TransferConfig {
  style: string;
  cutRateSec: number;
  clipCount: number;
  render: boolean;
  batch: number;
  outputPrefix: string;
  dryRun: boolean;
  shuffle: boolean;
  mixSources: boolean;
  overlayText?: string;
}

// ── Clip selection — montage builder ─────────────────────────────────────────

function buildMontageTimeline(
  clips: Clip[],
  cutRateSec: number,
  totalFrames: number,
  mixSources: boolean,
): { clip: Clip; durationFrames: number; trimStartFrames: number }[] {
  const cutFrames = Math.round(cutRateSec * FPS);
  const timeline: { clip: Clip; durationFrames: number; trimStartFrames: number }[] = [];
  let usedFrames = 0;

  // Shuffle clips for variety, optionally ensuring source mixing
  let pool = [...clips];
  if (mixSources) {
    // Interleave clips from different sources
    const bySource = new Map<string, Clip[]>();
    for (const c of pool) {
      const src = c.source_video_id;
      if (!bySource.has(src)) bySource.set(src, []);
      bySource.get(src)!.push(c);
    }
    pool = [];
    const sources = [...bySource.values()];
    const maxLen = Math.max(...sources.map(s => s.length));
    for (let i = 0; i < maxLen; i++) {
      for (const s of sources) {
        if (i < s.length) pool.push(s[i]);
      }
    }
  } else {
    // Fisher-Yates shuffle
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
  }

  let clipIdx = 0;
  while (usedFrames < totalFrames && clipIdx < pool.length * 3) {
    const clip = pool[clipIdx % pool.length];
    const clipMaxFrames = Math.round(clip.duration_sec * FPS);

    // Each cut segment: trim a portion of the clip
    const segmentFrames = Math.min(cutFrames, totalFrames - usedFrames, clipMaxFrames);
    if (segmentFrames < Math.round(0.3 * FPS)) break;

    // Random trim offset within the clip to get variety
    const maxTrimStart = Math.max(0, clipMaxFrames - segmentFrames);
    const trimStart = maxTrimStart > 0 ? Math.floor(Math.random() * maxTrimStart) : 0;

    timeline.push({
      clip,
      durationFrames: segmentFrames,
      trimStartFrames: trimStart,
    });

    usedFrames += segmentFrames;
    clipIdx++;
  }

  return timeline;
}

// ── Props generation ─────────────────────────────────────────────────────────

function generateProps(
  timeline: { clip: Clip; durationFrames: number; trimStartFrames: number }[],
  style: EditStyle,
  overlayText?: string,
): Record<string, unknown> {
  return {
    bodyClips: timeline.map(t => ({
      src: t.clip.file,
      durationFrames: t.durationFrames,
      trimStartFrames: t.trimStartFrames,
    })),
    colorGrade: style.colorGrade,
    zoomPunch: style.zoomPunch,
    showOverlay: !!overlayText,
    overlayText: overlayText || "",
    overlayPosition: "bottom",
    musicVolume: 0.85,
  };
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderEdit(propsFile: string, outputFile: string): boolean {
  const propsData = fs.readFileSync(propsFile, 'utf-8');
  const cmd = [
    'npx', 'remotion', 'render',
    'remotion/index.ts',
    'AscensionVideo',
    outputFile,
    '--props', `'${propsData.replace(/'/g, "'\\''")}'`,
  ].join(' ');

  console.log(`  [RENDER] ${path.basename(outputFile)}`);
  try {
    execSync(cmd, { cwd: ROOT, stdio: 'inherit', timeout: 120_000 });
    return true;
  } catch {
    console.error(`  [ERROR] Render failed`);
    return false;
  }
}

// ── Main generation ──────────────────────────────────────────────────────────

function generateOne(config: TransferConfig, index: number): string | null {
  const style = STYLES[config.style] || STYLES.dark_cinema;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const id = `${config.outputPrefix}_${style.name}_${timestamp}_${index}`;

  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  ${id}`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`  Style     : ${style.name} — ${style.description}`);
  console.log(`  Cut rate  : ${config.cutRateSec}s (${Math.round(TOTAL_DURATION_SEC / config.cutRateSec)} cuts in ${TOTAL_DURATION_SEC}s)`);
  console.log(`  Color     : ${style.colorGrade}`);
  console.log(`  Zoom punch: ${style.zoomPunch}`);

  // Get clips from library
  invalidateCache();
  const allClips = getAllClips();
  if (allClips.length === 0) {
    console.log(`  [SKIP] No clips in library.`);
    return null;
  }

  console.log(`  Library   : ${allClips.length} clips available`);

  // Build rapid-cut montage timeline
  const timeline = buildMontageTimeline(allClips, config.cutRateSec, TOTAL_FRAMES, config.mixSources);
  console.log(`  Timeline  : ${timeline.length} cuts`);

  const sources = new Set(timeline.map(t => t.clip.source_video_id));
  console.log(`  Sources   : ${sources.size} different videos mixed`);

  for (const t of timeline) {
    const trimSec = (t.trimStartFrames / FPS).toFixed(1);
    const durSec = (t.durationFrames / FPS).toFixed(1);
    console.log(`    → ${t.clip.clip_id.slice(0, 50)} | ${durSec}s (trim @${trimSec}s)`);
  }

  // Generate props
  const props = generateProps(timeline, style, config.overlayText);
  const propsDir = path.join(OUTPUT_DIR, 'props');
  fs.mkdirSync(propsDir, { recursive: true });
  const propsFile = path.join(propsDir, `${id}.props.json`);
  fs.writeFileSync(propsFile, JSON.stringify(props, null, 2));

  if (config.render && !config.dryRun) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    const outputFile = path.join(OUTPUT_DIR, `${id}.mp4`);
    const ok = renderEdit(propsFile, outputFile);
    if (ok) {
      console.log(`  ✅ ${path.relative(ROOT, outputFile)} (${(fs.statSync(outputFile).size / 1024 / 1024).toFixed(1)}MB)`);
      return outputFile;
    }
    return null;
  } else if (config.dryRun) {
    console.log(`  [DRY RUN] Would render ${id}.mp4`);
  }
  return propsFile;
}

function generateBatch(config: TransferConfig): void {
  const results: (string | null)[] = [];
  for (let i = 0; i < config.batch; i++) {
    results.push(generateOne(config, i));
  }

  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  BATCH: ${results.filter(Boolean).length}/${config.batch} complete`);
  console.log(`${'═'.repeat(60)}\n`);
}

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs(): TransferConfig {
  const args = process.argv.slice(2);
  const config: TransferConfig = {
    style: 'dark_cinema',
    cutRateSec: 1.0,
    clipCount: 15,
    render: false,
    batch: 1,
    outputPrefix: 'bp_edit',
    dryRun: false,
    shuffle: true,
    mixSources: true,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--style': config.style = args[++i]; break;
      case '--cut-rate': config.cutRateSec = parseFloat(args[++i]); break;
      case '--clip-count': config.clipCount = parseInt(args[++i], 10); break;
      case '--render': config.render = true; break;
      case '--batch': config.batch = parseInt(args[++i], 10); break;
      case '--output-prefix': config.outputPrefix = args[++i]; break;
      case '--dry-run': config.dryRun = true; break;
      case '--no-mix': config.mixSources = false; break;
      case '--overlay': config.overlayText = args[++i]; break;
      case '--list-styles':
        console.log('Available styles:');
        for (const [k, v] of Object.entries(STYLES)) {
          console.log(`  ${k}: ${v.description} (cut: ${v.cutRateSec}s, grade: ${v.colorGrade})`);
        }
        process.exit(0);
        break; // eslint: unreachable but satisfies no-fallthrough
      case '--help':
        console.log(`
Ascension Engine v3.0 — DNA Transfer (BP/Mog Edit Generator)

Usage:
  npx ts-node src/scripts/dna-transfer.ts [options]

Options:
  --style <name>         Edit style: dark_cinema, desaturated, cold_blue, warm_ambient, natural, teal_orange
  --cut-rate <sec>       Seconds per cut (default: style default, typically 0.8-1.5)
  --render               Render to MP4
  --batch <n>            Generate N edits
  --output-prefix <str>  Filename prefix (default: bp_edit)
  --overlay <text>       Optional minimal text overlay
  --no-mix               Don't interleave clips from different source videos
  --dry-run              Preview only
  --list-styles          Show available styles
`);
        process.exit(0);
        break; // eslint: unreachable but satisfies no-fallthrough
      default:
        if (!args[i].startsWith('-')) break;
        console.error(`Unknown: ${args[i]}`);
        process.exit(1);
    }
  }

  // Apply style defaults if cut rate not explicitly set
  const style = STYLES[config.style];
  if (style && !args.includes('--cut-rate')) {
    config.cutRateSec = style.cutRateSec;
  }

  return config;
}

const config = parseArgs();
generateBatch(config);
