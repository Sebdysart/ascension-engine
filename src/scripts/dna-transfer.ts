#!/usr/bin/env npx ts-node
/**
 * Ascension Engine v4.0 — Brutal BP DNA Transfer
 * Generates beat-synced mog edits from gold reference blueprints.
 *
 * Usage:
 *   npx ts-node src/scripts/dna-transfer.ts --render
 *   npx ts-node src/scripts/dna-transfer.ts --style dark_cinema --render
 *   npx ts-node src/scripts/dna-transfer.ts --reference <videoId> --render
 *   npx ts-node src/scripts/dna-transfer.ts --batch 4 --render
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import {
  getAllClips,
  getBrutalTimeline,
  invalidateCache,
  listSequenceTemplates,
  remixSequence,
} from '../lib/clipLibrary';

const ROOT = path.resolve(__dirname, '..', '..');
const OUTPUT_DIR = path.join(ROOT, 'out');
const PROFILES_DIR = path.join(ROOT, 'style-profiles');
const FPS = 30;
const DURATION_SEC = 15;

// ── Style presets ────────────────────────────────────────────────────────────

const STYLE_GRADES: Record<string, string> = {
  dark_cinema: "dark_cinema",
  dark_cinema_hard: "dark_cinema_hard",
  desaturated: "desaturated",
  cold_blue: "cold_blue",
  warm_ambient: "warm_ambient",
  natural: "natural",
  teal_orange: "teal_orange",
  teal_orange_hard: "teal_orange_hard",
};

// ── Load blueprints from style-profiles ──────────────────────────────────────

interface Blueprint {
  video_id: string;
  source_creator: string;
  bpm: number;
  beat_times_sec: number[];
  cut_rhythm: {
    avg_cut_sec: number;
    total_cuts: number;
    cuts_on_beat_pct: number;
    cut_points?: { time_sec: number; duration_sec: number; on_beat: boolean }[];
  };
  visual_grade?: { color_grade?: string };
  audio?: {
    peak_moments_sec?: number[];
    onset_times_sec?: number[];
  };
  motion_fx?: {
    impact_hold_ms?: number;
  };
}

function loadAllBlueprints(): Blueprint[] {
  if (!fs.existsSync(PROFILES_DIR)) return [];
  return fs.readdirSync(PROFILES_DIR)
    .filter(f => f.endsWith('.json'))
    .map(f => {
      try {
        return JSON.parse(fs.readFileSync(path.join(PROFILES_DIR, f), 'utf-8'));
      } catch {
        return null;
      }
    })
    .filter((b): b is Blueprint => b !== null && b.bpm > 0);
}

function pickBlueprint(blueprints: Blueprint[], reference?: string): Blueprint {
  if (reference) {
    const match = blueprints.find(b => b.video_id.includes(reference));
    if (match) return match;
  }
  // Pick the blueprint with the best beat alignment
  return blueprints.sort((a, b) =>
    (b.cut_rhythm.cuts_on_beat_pct || 0) - (a.cut_rhythm.cuts_on_beat_pct || 0)
  )[0];
}

// ── Generate ─────────────────────────────────────────────────────────────────

interface Config {
  style: string;
  reference?: string;
  remix?: string;
  render: boolean;
  batch: number;
  prefix: string;
  dryRun: boolean;
}

function generateOne(config: Config, blueprints: Blueprint[], index: number): string | null {
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const mode = config.remix ? "remix" : "blueprint";
  const id = `${config.prefix}_${config.style}_${mode}_${ts}_${index}`;
  const grade = STYLE_GRADES[config.style] || "dark_cinema";

  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  ${id}`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`  Mode      : ${mode}`);
  console.log(`  Grade     : ${grade}`);

  invalidateCache();
  const allClips = getAllClips();
  console.log(`  Library   : ${allClips.length} clips`);

  let timeline;

  if (config.remix) {
    const templates = listSequenceTemplates();
    const target = config.remix === 'auto'
      ? templates[Math.floor(Math.random() * templates.length)]
      : templates.find(t => t.includes(config.remix!)) || templates[0];
    if (!target) { console.log("  [SKIP] No sequence templates."); return null; }
    console.log(`  Template  : ${target}`);
    timeline = remixSequence(target, FPS);
    console.log(`  Remixed   : ${timeline.length} slots`);
  } else {
    const blueprint = pickBlueprint(blueprints, config.reference);
    if (!blueprint) { console.log("  [SKIP] No blueprints."); return null; }
    console.log(`  Blueprint : ${blueprint.video_id} (${blueprint.source_creator})`);
    console.log(`  BPM       : ${blueprint.bpm}`);
    console.log(`  Avg cut   : ${blueprint.cut_rhythm.avg_cut_sec}s`);
    console.log(`  Beat sync : ${(blueprint.cut_rhythm.cuts_on_beat_pct * 100).toFixed(0)}%`);
    timeline = getBrutalTimeline(blueprint, FPS, DURATION_SEC);
  }
  console.log(`  Timeline  : ${timeline.length} cuts`);

  const sources = new Set(timeline.map(t => t.clip.source_video_id));
  const impacts = timeline.filter(t => t.is_impact).length;
  const onBeat = timeline.filter(t => t.on_beat).length;
  console.log(`  Sources   : ${sources.size} videos mixed`);
  console.log(`  On-beat   : ${onBeat}/${timeline.length} cuts`);
  console.log(`  Impacts   : ${impacts} energy peaks`);

  for (const t of timeline.slice(0, 8)) {
    const durSec = (t.durationFrames / FPS).toFixed(2);
    const beat = t.on_beat ? "♫" : " ";
    const impact = t.is_impact ? "⚡" : " ";
    console.log(`    ${beat}${impact} ${(t.startFrame / FPS).toFixed(2)}s → +${durSec}s | ${t.clip.clip_id.slice(0, 45)}`);
  }
  if (timeline.length > 8) console.log(`    ... +${timeline.length - 8} more`);

  // Build props
  const impactHold = 4;
  const props = {
    bodyClips: timeline.map(t => ({
      src: t.clip.file,
      durationFrames: t.durationFrames,
      trimStartFrames: t.trimStartFrames,
      isImpact: t.is_impact,
    })),
    colorGrade: grade,
    zoomPunch: true,
    impactHoldFrames: impactHold,
    musicVolume: 0.85,
  };

  const propsDir = path.join(OUTPUT_DIR, 'props');
  fs.mkdirSync(propsDir, { recursive: true });
  const propsFile = path.join(propsDir, `${id}.props.json`);
  fs.writeFileSync(propsFile, JSON.stringify(props, null, 2));

  if (config.render && !config.dryRun) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    const outFile = path.join(OUTPUT_DIR, `${id}.mp4`);
    const propsData = fs.readFileSync(propsFile, 'utf-8');
    const cmd = `npx remotion render remotion/index.ts AscensionVideo ${outFile} --props '${propsData.replace(/'/g, "'\\''")}'`;
    console.log(`  [RENDER] ${path.basename(outFile)}`);
    try {
      execSync(cmd, { cwd: ROOT, stdio: 'inherit', timeout: 120_000 });
      const size = (fs.statSync(outFile).size / 1024 / 1024).toFixed(1);
      console.log(`  ✅ ${path.relative(ROOT, outFile)} (${size}MB)`);
      return outFile;
    } catch {
      console.log(`  ❌ Render failed`);
      return null;
    }
  }
  if (config.dryRun) console.log(`  [DRY RUN] Would render ${id}.mp4`);
  return propsFile;
}

// ── CLI ──────────────────────────────────────────────────────────────────────

function main() {
  const args = process.argv.slice(2);
  const config: Config = {
    style: 'dark_cinema',
    render: false,
    batch: 1,
    prefix: 'bp',
    dryRun: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--style': config.style = args[++i]; break;
      case '--reference': config.reference = args[++i]; break;
      case '--remix': config.remix = args[i + 1]?.startsWith('-') ? 'auto' : (args[++i] || 'auto'); break;
      case '--render': config.render = true; break;
      case '--batch': config.batch = parseInt(args[++i], 10); break;
      case '--prefix': config.prefix = args[++i]; break;
      case '--dry-run': config.dryRun = true; break;
      case '--help':
        console.log(`
Ascension Engine v4.0 — Brutal BP DNA Transfer

  --style <grade>    Color grade (dark_cinema, desaturated, teal_orange, cold_blue, warm_ambient, natural)
  --reference <id>   Use specific gold video's blueprint
  --render           Render to MP4
  --batch <n>        Generate N edits
  --prefix <str>     Output prefix (default: bp)
  --dry-run          Preview only
`);
        process.exit(0);
        break; // eslint
      default: break;
    }
  }

  const blueprints = loadAllBlueprints();
  console.log(`Loaded ${blueprints.length} blueprints`);
  if (blueprints.length === 0) {
    console.error("No blueprints found. Run ingest first.");
    process.exit(1);
  }

  const results: (string | null)[] = [];
  for (let i = 0; i < config.batch; i++) {
    results.push(generateOne(config, blueprints, i));
  }
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  DONE: ${results.filter(Boolean).length}/${config.batch}`);
  console.log(`${'═'.repeat(60)}\n`);
}

main();
