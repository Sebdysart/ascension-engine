#!/usr/bin/env ts-node
/**
 * Ascension Engine — DNA Transfer Generator
 * Reads a reference video's style-profile + clip manifest and generates a
 * Remotion composition file that transfers its DNA to a new composition.
 *
 * Usage:
 *   npx ts-node src/scripts/dna-transfer.ts \
 *     --reference <video_id> \
 *     --archetype <glow_up|frame_maxxing|skin_maxxing|style_maxxing> \
 *     --clip-count 12 \
 *     --output remotion/compositions/<output_name>.tsx
 *
 *   # Quick preset (no reference video):
 *   npx ts-node src/scripts/dna-transfer.ts \
 *     --archetype glow_up \
 *     --output remotion/compositions/glowup_v1.tsx
 */

import * as fs from "fs";
import * as path from "path";

// ── CLI args ────────────────────────────────────────────────────────────────

function parseArgs(): Record<string, string> {
  const args = process.argv.slice(2);
  const out: Record<string, string> = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--")) {
      const key = args[i].slice(2);
      const val = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : "true";
      out[key] = val;
    }
  }
  return out;
}

// ── Paths ───────────────────────────────────────────────────────────────────

const ROOT = path.resolve(__dirname, "../..");
const STYLE_PROFILES_DIR = path.join(ROOT, "style-profiles");
const CLIP_MANIFEST = path.join(ROOT, "clip-manifest.json");
const SEQUENCE_TEMPLATES_DIR = path.join(ROOT, "library", "sequence_templates");

// ── Archetype presets ───────────────────────────────────────────────────────

type Archetype = "glow_up" | "frame_maxxing" | "skin_maxxing" | "style_maxxing";

const ARCHETYPE_PRESETS: Record<Archetype, {
  colorGrade: string;
  cutRateSec: number;
  hookText: string;
  ctaText: string;
  durationSec: number;
  fps: number;
}> = {
  glow_up: {
    colorGrade: "teal_orange",
    cutRateSec: 1.8,
    hookText: "I CHANGED MY FACE IN 90 DAYS",
    ctaText: "DISCIPLINE = RESULTS. DROP YOUR GLOW-UP PROGRESS BELOW.",
    durationSec: 15,
    fps: 30,
  },
  frame_maxxing: {
    colorGrade: "cold_blue",
    cutRateSec: 2.2,
    hookText: "JAWLINE MAXXING PROTOCOL",
    ctaText: "STRUCTURE IS DESTINY. SAVE THIS AND START TODAY.",
    durationSec: 15,
    fps: 30,
  },
  skin_maxxing: {
    colorGrade: "warm_gold",
    cutRateSec: 2.8,
    hookText: "CLEAR SKIN IN 30 DAYS",
    ctaText: "GLOW IS BUILT, NOT BORN. FOLLOW FOR THE PROTOCOL.",
    durationSec: 15,
    fps: 30,
  },
  style_maxxing: {
    colorGrade: "desaturated",
    cutRateSec: 2.0,
    hookText: "FROM AVERAGE TO AESTHETIC",
    ctaText: "YOUR WARDROBE IS YOUR FIRST IMPRESSION. LEVEL UP.",
    durationSec: 15,
    fps: 30,
  },
};

// ── Helpers ─────────────────────────────────────────────────────────────────

function loadJson<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

function pickClips(manifest: any, referenceVidId: string | null, count: number): any[] {
  const allClips: any[] = manifest?.clips ?? [];
  if (!allClips.length) return [];

  // Prefer clips from reference video; fill remainder from library
  const refClips = referenceVidId
    ? allClips.filter((c: any) => c.source_video_id === referenceVidId)
    : [];
  const otherClips = allClips.filter((c: any) => c.source_video_id !== referenceVidId);

  // Sort by rank descending
  const sort = (cs: any[]) => [...cs].sort((a, b) => (b.rank ?? 0.5) - (a.rank ?? 0.5));

  const picked = [...sort(refClips), ...sort(otherClips)].slice(0, count);
  return picked;
}

function buildCaptionEntries(
  hookText: string,
  fps: number,
  wordsPerCaption: number = 4,
  bodyStartSec: number = 3,
  bodySec: number = 9,
): string {
  // Generate dummy caption entries that cover the body section
  const words = hookText.split(" ");
  const entries: string[] = [];
  const captionDurFrames = Math.round(1.5 * fps);
  const gapFrames = 4;
  let frame = Math.round(bodyStartSec * fps);

  for (let i = 0; i < words.length && frame < Math.round((bodyStartSec + bodySec) * fps); i += wordsPerCaption) {
    const chunk = words.slice(i, i + wordsPerCaption).join(" ");
    entries.push(`    { text: ${JSON.stringify(chunk)}, startFrame: ${frame}, durationFrames: ${captionDurFrames} },`);
    frame += captionDurFrames + gapFrames;
  }
  return entries.join("\n");
}

// ── Composition template ─────────────────────────────────────────────────────

function generateComposition(params: {
  outputName: string;
  archetype: Archetype;
  colorGrade: string;
  cutRateSec: number;
  hookText: string;
  ctaText: string;
  durationInFrames: number;
  fps: number;
  clips: any[];
  sourceVidId: string | null;
  sequenceTemplate: any | null;
}): string {
  const {
    outputName,
    archetype,
    colorGrade,
    cutRateSec,
    hookText,
    ctaText,
    durationInFrames,
    fps,
    clips,
    sourceVidId,
    sequenceTemplate,
  } = params;

  const clipSegments = clips.map((c) => ({
    src: c.file ?? `library/clips/${c.clip_id}.mp4`,
    durationFrames: Math.round((c.duration_sec ?? 2) * fps),
    trimStartFrames: 0,
  }));

  const captionEntries = buildCaptionEntries(hookText, fps);
  const bpm = sequenceTemplate?.bpm ?? 0;
  const beatAlignment = sequenceTemplate?.beat_alignment ?? "unknown";
  const colorGradeComment = sequenceTemplate?.color_grade ?? colorGrade;

  return `/**
 * Ascension Engine — DNA Transfer Composition
 * Generated: ${new Date().toISOString()}
 * Archetype: ${archetype}
 * Source reference: ${sourceVidId ?? "preset (no reference)"}
 * BPM: ${bpm}  Beat alignment: ${beatAlignment}  Color grade: ${colorGradeComment}
 *
 * Register in remotion/index.tsx:
 *   import { ${outputName} } from "./compositions/${outputName}";
 *   <Composition id="${outputName}" component={${outputName}} durationInFrames={${durationInFrames}} fps={${fps}} width={1080} height={1920} defaultProps={{...}} />
 */

import React from "react";
import { Composition } from "remotion";
import { VideoTemplate, CaptionEntry } from "../VideoTemplate";

// ── Clip library ────────────────────────────────────────────────────────────

const BODY_CLIPS = ${JSON.stringify(clipSegments, null, 2)};

// ── Caption track ────────────────────────────────────────────────────────────

const CAPTIONS: CaptionEntry[] = [
${captionEntries}
];

// ── Composition ─────────────────────────────────────────────────────────────

export const ${outputName}: React.FC = () => (
  <Composition
    id="${outputName}"
    component={VideoTemplate}
    durationInFrames={${durationInFrames}}
    fps={${fps}}
    width={1080}
    height={1920}
    defaultProps={{
      hookText: ${JSON.stringify(hookText)},
      bodyClips: BODY_CLIPS,
      captions: CAPTIONS,
      ctaText: ${JSON.stringify(ctaText)},
      colorGrade: ${JSON.stringify(colorGrade)} as any,
      cutRateSec: ${cutRateSec},
      archetype: ${JSON.stringify(archetype)} as any,
      showBeforeAfter: false,
    }}
  />
);

export default ${outputName};
`;
}

// ── Main ─────────────────────────────────────────────────────────────────────

function main(): void {
  const args = parseArgs();

  const referenceId = args["reference"] ?? null;
  const archetypeRaw = (args["archetype"] ?? "glow_up").toLowerCase().replace("-", "_") as Archetype;
  const archetype: Archetype = ARCHETYPE_PRESETS[archetypeRaw] ? archetypeRaw : "glow_up";
  const clipCount = parseInt(args["clip-count"] ?? "8", 10);
  const outputArg = args["output"] ?? `remotion/compositions/${archetype}_dna_${Date.now()}.tsx`;

  const preset = ARCHETYPE_PRESETS[archetype];

  // Load style profile if reference given
  let profile: any = null;
  let sequenceTemplate: any = null;
  if (referenceId) {
    profile = loadJson(path.join(STYLE_PROFILES_DIR, `${referenceId}.json`));
    sequenceTemplate = loadJson(path.join(SEQUENCE_TEMPLATES_DIR, `${referenceId}.json`));
    if (!profile) {
      console.warn(`[warn] style-profile not found for ${referenceId} — using preset defaults`);
    }
  }

  // Derive params from profile (or preset fallback)
  const colorGrade = profile?.visuals?.color_grade?.toLowerCase().replace(/\s+/g, "_") ?? preset.colorGrade;
  const cutRateSec = profile?.cut_rhythm?.avg_cut_length_sec || preset.cutRateSec;
  const fps = preset.fps;
  const durationInFrames = preset.durationSec * fps;

  // Load clip manifest
  const manifest = loadJson<any>(CLIP_MANIFEST);
  const clips = pickClips(manifest, referenceId, clipCount);

  if (!clips.length) {
    console.warn("[warn] No clips found in manifest — body clips will be empty.");
  }

  // Derive output component name from file path
  const outputFile = path.resolve(ROOT, outputArg);
  const outputBasename = path.basename(outputFile, ".tsx");
  const outputName = outputBasename
    .split(/[-_]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join("");

  const source = generateComposition({
    outputName,
    archetype,
    colorGrade,
    cutRateSec,
    hookText: preset.hookText,
    ctaText: preset.ctaText,
    durationInFrames,
    fps,
    clips,
    sourceVidId: referenceId,
    sequenceTemplate,
  });

  // Write output
  fs.mkdirSync(path.dirname(outputFile), { recursive: true });
  fs.writeFileSync(outputFile, source, "utf-8");

  console.log(`\n✅  DNA Transfer complete`);
  console.log(`   Archetype    : ${archetype}`);
  console.log(`   Reference    : ${referenceId ?? "(preset)"}`);
  console.log(`   Color grade  : ${colorGrade}`);
  console.log(`   Cut rate     : ${cutRateSec}s`);
  console.log(`   Clips        : ${clips.length}`);
  console.log(`   Output       : ${outputFile}\n`);
}

main();
