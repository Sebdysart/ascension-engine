#!/usr/bin/env npx ts-node
/**
 * Ascension Engine v2.1 — DNA Transfer Script
 * ──────────────────────────────────────────────────────────────────────────────
 * Generates new Remotion compositions from style profiles + clip library.
 * The bridge between "I have ranked clips" and "here's a rendered MP4."
 *
 * Usage:
 *   npx ts-node src/scripts/dna-transfer.ts --archetype GlowUp --clip-count 8
 *   npx ts-node src/scripts/dna-transfer.ts --reference jordan_abc12345 --archetype GlowUp
 *   npx ts-node src/scripts/dna-transfer.ts --archetype FrameMaxxing --min-rank 0.85 --render
 *   npx ts-node src/scripts/dna-transfer.ts --batch 4 --archetype GlowUp --render
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

// ── Imports from engine modules ──────────────────────────────────────────────

import {
  getAllClips,
  getClipsByTags,
  getTopClips,
  getBlueprintFromReference,
  invalidateCache,
} from '../lib/clipLibrary';
import type { Clip, TimelineBlueprint } from '../lib/clipLibrary';

import {
  getPresetByArchetype,
  listArchetypes,
  PRESETS,
} from '../../remotion/remotion-presets';
import type { ArchetypePreset } from '../../remotion/remotion-presets';

// ── Constants ────────────────────────────────────────────────────────────────

const ROOT = path.resolve(__dirname, '..', '..');
const OUTPUT_DIR = path.join(ROOT, 'out');

const FPS = 30;

// ── Color grade mapping (preset enum → VideoTemplate prop) ───────────────────

const COLOR_GRADE_MAP: Record<string, string> = {
  TealOrange: 'teal_orange',
  ColdBlue: 'cold_blue',
  WarmGold: 'warm_gold',
  Desaturated: 'desaturated',
};

const ARCHETYPE_MAP: Record<string, string> = {
  GlowUp: 'glow_up',
  FrameMaxxing: 'frame_maxxing',
  SkinMaxxing: 'skin_maxxing',
  StyleMaxxing: 'style_maxxing',
};

// ── Interfaces ───────────────────────────────────────────────────────────────

interface TransferConfig {
  archetype: string;
  reference?: string;
  clipCount: number;
  minRank: number;
  exploitRatio: number;
  hookText?: string;
  ctaText?: string;
  render: boolean;
  batch: number;
  outputPrefix: string;
  dryRun: boolean;
}

interface GeneratedComposition {
  id: string;
  archetype: string;
  clipIds: string[];
  hookText: string;
  ctaText: string;
  colorGrade: string;
  cutRateSec: number;
  propsFile: string;
  outputFile?: string;
}

// ── Hook text generation ─────────────────────────────────────────────────────

const HOOK_TEMPLATES: Record<string, string[]> = {
  GlowUp: [
    'I CHANGED MY FACE IN 90 DAYS',
    'THIS IS WHAT DISCIPLINE LOOKS LIKE',
    'NOBODY BELIEVED THE TRANSFORMATION',
    'FROM INVISIBLE TO UNSTOPPABLE',
    'THE GLOW UP THEY SAID WAS IMPOSSIBLE',
  ],
  FrameMaxxing: [
    'GENETICS LOADED DIFFERENT',
    'FRAME DOESN\'T LIE',
    'THIS IS WHAT ELITE LOOKS LIKE',
    'BONE STRUCTURE > EVERYTHING',
    'THE MOG IS REAL',
  ],
  SkinMaxxing: [
    'CLEAR SKIN CHANGED EVERYTHING',
    'THE SKINCARE THAT ACTUALLY WORKED',
    'FROM WORST SKIN TO BEST SKIN',
    'THEY ASKED WHAT I DID DIFFERENT',
    '90 DAYS OF CONSISTENCY',
  ],
  StyleMaxxing: [
    'STYLE IS THE MULTIPLIER',
    'FIT CHECK × GLOW UP',
    'DRIP CHANGES PERCEPTION',
    'FROM BASIC TO BASED',
    'THE OUTFIT THAT CHANGED EVERYTHING',
  ],
};

function pickHookText(archetype: string, index: number): string {
  const templates = HOOK_TEMPLATES[archetype] || HOOK_TEMPLATES.GlowUp;
  return templates[index % templates.length];
}

// ── Clip selection logic ─────────────────────────────────────────────────────

function selectClips(
  preset: ArchetypePreset,
  count: number,
  minRank: number,
  exploitRatio: number,
): Clip[] {
  const exploitCount = Math.ceil(count * exploitRatio);
  const exploreCount = count - exploitCount;

  // Exploit: pull from preferred tags with strict rank filter
  const exploitClips = getClipsByTags(preset.preferredTags, exploitCount, minRank);

  // Explore: pull top-ranked clips regardless of tag, lower rank threshold
  const usedIds = new Set(exploitClips.map((c) => c.clip_id));
  const explorePool = getTopClips(exploreCount * 3, Math.max(0, minRank - 0.2))
    .filter((c) => !usedIds.has(c.clip_id));

  // Shuffle explore pool for variety
  for (let i = explorePool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [explorePool[i], explorePool[j]] = [explorePool[j], explorePool[i]];
  }
  const exploreClips = explorePool.slice(0, exploreCount);

  const selected = [...exploitClips, ...exploreClips];

  // If library is empty or insufficient, return what we have
  if (selected.length === 0) {
    console.log(`  [WARN] No clips found matching criteria (tags: ${preset.preferredTags.join(', ')}, minRank: ${minRank})`);
    console.log(`         Using all available clips as fallback.`);
    return getAllClips().slice(0, count);
  }

  return selected;
}

// ── Blueprint-based clip selection ───────────────────────────────────────────

function selectClipsFromBlueprint(
  blueprint: TimelineBlueprint,
  preset: ArchetypePreset,
  count: number,
): Clip[] {
  const allClips = getAllClips();
  const clipMap = new Map(allClips.map((c) => [c.clip_id, c]));

  // Start with clips from the blueprint sequence
  const blueprintClips = blueprint.clip_sequence
    .map((id) => clipMap.get(id))
    .filter((c): c is Clip => c !== undefined)
    .slice(0, count);

  // Fill remaining slots from library
  if (blueprintClips.length < count) {
    const usedIds = new Set(blueprintClips.map((c) => c.clip_id));
    const fill = getClipsByTags(preset.preferredTags, count - blueprintClips.length, 0.3)
      .filter((c) => !usedIds.has(c.clip_id));
    blueprintClips.push(...fill);
  }

  return blueprintClips;
}

// ── Props file generation ────────────────────────────────────────────────────

function generatePropsJson(
  id: string,
  clips: Clip[],
  preset: ArchetypePreset,
  hookText: string,
  ctaText: string,
  cutRateSec: number,
): string {
  const cutRateFrames = Math.round(cutRateSec * FPS);

  const bodyClips = clips.map((clip) => ({
    src: clip.file,
    durationFrames: Math.min(
      cutRateFrames,
      Math.round(clip.duration_sec * FPS),
    ),
    trimStartFrames: 0,
  }));

  const props = {
    hookText,
    bodyClips,
    overlayTexts: [],
    ctaText,
    colorGrade: COLOR_GRADE_MAP[preset.colorGrade] || 'teal_orange',
    cutRateSec,
    archetype: ARCHETYPE_MAP[preset.name] || 'glow_up',
    showBeforeAfter: preset.hookType === 'BeforeAfterCut',
  };

  const propsDir = path.join(ROOT, 'out', 'props');
  if (!fs.existsSync(propsDir)) {
    fs.mkdirSync(propsDir, { recursive: true });
  }

  const propsFile = path.join(propsDir, `${id}.props.json`);
  fs.writeFileSync(propsFile, JSON.stringify(props, null, 2));

  return propsFile;
}

// ── Render via Remotion CLI ──────────────────────────────────────────────────

function renderComposition(
  propsFile: string,
  outputFile: string,
): boolean {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const propsData = fs.readFileSync(propsFile, 'utf-8');

  const cmd = [
    'npx', 'remotion', 'render',
    'remotion/index.ts',
    'AscensionVideo',
    outputFile,
    '--props', `'${propsData.replace(/'/g, "'\\''")}'`,
  ].join(' ');

  console.log(`  [RENDER] ${outputFile}`);
  try {
    execSync(cmd, { cwd: ROOT, stdio: 'inherit', timeout: 120_000 });
    return true;
  } catch (err) {
    console.error(`  [ERROR] Render failed: ${err}`);
    return false;
  }
}

// ── Main transfer function ───────────────────────────────────────────────────

function generateOne(config: TransferConfig, index: number): GeneratedComposition | null {
  const preset = getPresetByArchetype(config.archetype, config.reference);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const id = `${config.outputPrefix}_${config.archetype}_${timestamp}_${index}`;

  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  GENERATING: ${id}`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`  Archetype  : ${config.archetype}`);
  console.log(`  Reference  : ${config.reference || '(preset only)'}`);
  console.log(`  Clip count : ${config.clipCount}`);
  console.log(`  Min rank   : ${config.minRank}`);
  console.log(`  Exploit    : ${(config.exploitRatio * 100).toFixed(0)}%`);

  // Select clips
  let clips: Clip[];
  let cutRateSec = preset.cutRateSec;

  if (config.reference) {
    try {
      const blueprint = getBlueprintFromReference(config.reference);
      clips = selectClipsFromBlueprint(blueprint, preset, config.clipCount);
      cutRateSec = blueprint.cut_rate_sec || preset.cutRateSec;
      console.log(`  Blueprint  : ${blueprint.video_id} (${blueprint.bpm} BPM, ${blueprint.color_grade})`);
    } catch {
      console.log(`  [WARN] Reference not found, falling back to preset-only mode.`);
      clips = selectClips(preset, config.clipCount, config.minRank, config.exploitRatio);
    }
  } else {
    clips = selectClips(preset, config.clipCount, config.minRank, config.exploitRatio);
  }

  console.log(`  Selected   : ${clips.length} clips`);
  for (const c of clips) {
    console.log(`    → ${c.clip_id} (rank: ${c.rank.toFixed(2)}, dur: ${c.duration_sec.toFixed(1)}s, tags: ${c.tags.join(', ') || 'none'})`);
  }

  if (clips.length === 0) {
    console.log(`  [SKIP] No clips available. Skipping generation.`);
    return null;
  }

  const hookText = config.hookText || pickHookText(config.archetype, index);
  const ctaText = config.ctaText || 'DISCIPLINE = RESULTS. DROP YOUR GLOW-UP PROGRESS BELOW.';

  console.log(`  Hook text  : ${hookText}`);
  console.log(`  CTA text   : ${ctaText.slice(0, 40)}...`);
  console.log(`  Color grade: ${preset.colorGrade} → ${COLOR_GRADE_MAP[preset.colorGrade]}`);
  console.log(`  Cut rate   : ${cutRateSec}s`);

  // Generate props file
  const propsFile = generatePropsJson(id, clips, preset, hookText, ctaText, cutRateSec);
  console.log(`  Props file : ${path.relative(ROOT, propsFile)}`);

  const result: GeneratedComposition = {
    id,
    archetype: config.archetype,
    clipIds: clips.map((c) => c.clip_id),
    hookText,
    ctaText,
    colorGrade: COLOR_GRADE_MAP[preset.colorGrade] || 'teal_orange',
    cutRateSec,
    propsFile,
  };

  // Render if requested
  if (config.render && !config.dryRun) {
    const outputFile = path.join(OUTPUT_DIR, `${id}.mp4`);
    const success = renderComposition(propsFile, outputFile);
    if (success) {
      result.outputFile = outputFile;
      console.log(`  ✅ Rendered: ${path.relative(ROOT, outputFile)}`);
    } else {
      console.log(`  ❌ Render failed`);
    }
  } else if (config.dryRun) {
    console.log(`  [DRY RUN] Would render to out/${id}.mp4`);
  }

  return result;
}

// ── Batch generation ─────────────────────────────────────────────────────────

function generateBatch(config: TransferConfig): GeneratedComposition[] {
  invalidateCache();

  const results: GeneratedComposition[] = [];
  for (let i = 0; i < config.batch; i++) {
    const result = generateOne(config, i);
    if (result) results.push(result);
  }

  // Write batch manifest
  const manifestPath = path.join(OUTPUT_DIR, 'last-batch.json');
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  fs.writeFileSync(manifestPath, JSON.stringify({
    generated_at: new Date().toISOString(),
    archetype: config.archetype,
    reference: config.reference || null,
    total: results.length,
    compositions: results.map((r) => ({
      id: r.id,
      clip_ids: r.clipIds,
      hook_text: r.hookText,
      color_grade: r.colorGrade,
      cut_rate_sec: r.cutRateSec,
      props_file: r.propsFile,
      output_file: r.outputFile || null,
    })),
  }, null, 2));

  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  BATCH COMPLETE`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`  Generated  : ${results.length}/${config.batch}`);
  console.log(`  Rendered   : ${results.filter((r) => r.outputFile).length}`);
  console.log(`  Manifest   : ${path.relative(ROOT, manifestPath)}`);
  console.log(`${'═'.repeat(60)}\n`);

  return results;
}

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs(): TransferConfig {
  const args = process.argv.slice(2);
  const config: TransferConfig = {
    archetype: 'GlowUp',
    clipCount: 8,
    minRank: 0.5,
    exploitRatio: 0.7,
    render: false,
    batch: 1,
    outputPrefix: 'gen',
    dryRun: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--archetype':
        config.archetype = args[++i];
        break;
      case '--reference':
        config.reference = args[++i];
        break;
      case '--clip-count':
        config.clipCount = parseInt(args[++i], 10);
        break;
      case '--min-rank':
        config.minRank = parseFloat(args[++i]);
        break;
      case '--exploit-ratio':
        config.exploitRatio = parseFloat(args[++i]);
        break;
      case '--hook-text':
        config.hookText = args[++i];
        break;
      case '--cta-text':
        config.ctaText = args[++i];
        break;
      case '--render':
        config.render = true;
        break;
      case '--batch':
        config.batch = parseInt(args[++i], 10);
        break;
      case '--output-prefix':
        config.outputPrefix = args[++i];
        break;
      case '--dry-run':
        config.dryRun = true;
        break;
      case '--list-archetypes':
        console.log('Available archetypes:', listArchetypes().join(', '));
        process.exit(0);
        break; // eslint: unreachable but satisfies no-fallthrough
      case '--help':
        console.log(`
Ascension Engine v2.1 — DNA Transfer

Usage:
  npx ts-node src/scripts/dna-transfer.ts [options]

Options:
  --archetype <name>      Archetype preset (default: GlowUp)
  --reference <videoId>   Reference video ID for blueprint-based generation
  --clip-count <n>        Number of clips to select (default: 8)
  --min-rank <0.0-1.0>    Minimum clip rank threshold (default: 0.5)
  --exploit-ratio <0-1>   Exploit vs explore ratio (default: 0.7)
  --hook-text <text>      Custom hook text override
  --cta-text <text>       Custom CTA text override
  --render                Render to MP4 after generating
  --batch <n>             Generate multiple compositions (default: 1)
  --output-prefix <str>   Output filename prefix (default: gen)
  --dry-run               Preview only, don't render
  --list-archetypes       Show available archetypes
  --help                  Show this help
`);
        process.exit(0);
        break; // eslint: unreachable but satisfies no-fallthrough
      default:
        console.error(`Unknown argument: ${args[i]}`);
        process.exit(1);
    }
  }

  if (!PRESETS[config.archetype]) {
    console.error(`Unknown archetype: ${config.archetype}. Available: ${listArchetypes().join(', ')}`);
    process.exit(1);
  }

  return config;
}

// ── Entry point ──────────────────────────────────────────────────────────────

const config = parseArgs();
generateBatch(config);
