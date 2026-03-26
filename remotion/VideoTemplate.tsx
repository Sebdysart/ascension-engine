/**
 * Ascension Engine v4.0 — BrutalBeatMontage
 * Beat-synced rapid-cut BP/mog edit composition.
 * Zero text. Zero CTA. Just clips cut on beats with dark grading.
 */

import {
  AbsoluteFill,
  Sequence,
  Video,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Audio,
  staticFile,
} from "remotion";

// ── Types ──────────────────────────────────────────────────────────────────

export interface ClipSegment {
  src: string;
  durationFrames: number;
  trimStartFrames?: number;
  isImpact?: boolean;
}

export interface VideoTemplateProps {
  bodyClips: ClipSegment[];
  musicPath?: string;
  colorGrade?: string;
  zoomPunch?: boolean;
  impactHoldFrames?: number;
  musicVolume?: number;
}

// ── Color Grades (BP-accurate from gold DNA) ─────────────────────────────

const COLOR_GRADES: Record<string, string> = {
  dark_cinema:      "brightness(0.75) contrast(1.35) saturate(0.85) sepia(0.15)",
  dark_cinema_hard: "brightness(0.68) contrast(1.45) saturate(0.75) sepia(0.12)",
  desaturated:      "saturate(0.15) contrast(1.25) brightness(0.95)",
  cold_blue:        "saturate(0.9) hue-rotate(20deg) brightness(0.85) contrast(1.2)",
  warm_ambient:     "saturate(1.1) sepia(0.2) brightness(0.85) contrast(1.2) hue-rotate(-5deg)",
  natural:          "saturate(1.05) contrast(1.08) brightness(1.0)",
  teal_orange:      "saturate(1.3) hue-rotate(-10deg) contrast(1.15)",
  teal_orange_hard: "saturate(1.4) hue-rotate(-12deg) contrast(1.25) brightness(0.9)",
  none:             "none",
};

// ── Micro-zoom with cubic easing ─────────────────────────────────────────

function cubicEaseIn(t: number): number {
  return t * t;
}

// ── Main Composition ─────────────────────────────────────────────────────

export const VideoTemplate: React.FC<VideoTemplateProps> = ({
  bodyClips = [],
  musicPath,
  colorGrade = "dark_cinema",
  zoomPunch = true,
  impactHoldFrames = 4,
  musicVolume = 0.85,
}) => {
  const { fps, durationInFrames } = useVideoConfig();
  const frame = useCurrentFrame();

  const cssFilter = COLOR_GRADES[colorGrade] || COLOR_GRADES.dark_cinema;

  // Build sequential timeline
  let currentFrame = 0;
  const timeline = bodyClips.map((clip, i) => {
    const start = currentFrame;
    const dur = Math.min(clip.durationFrames, durationInFrames - start);
    currentFrame += dur;
    return { ...clip, startFrame: start, renderDuration: Math.max(1, dur), index: i };
  }).filter(c => c.startFrame < durationInFrames);

  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a0a", overflow: "hidden" }}>

      {/* Vignette overlay */}
      <div style={{
        position: "absolute", inset: 0, zIndex: 20, pointerEvents: "none",
        background: "radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.45) 100%)",
      }} />

      {/* Color graded clip layer */}
      <AbsoluteFill style={{ filter: cssFilter }}>
        {timeline.map((clip) => {
          const clipLocalFrame = frame - clip.startFrame;
          const clipProgress = clipLocalFrame / Math.max(clip.renderDuration, 1);

          // Micro-zoom: 1.0 → 1.06 with cubic ease
          const zoom = zoomPunch
            ? interpolate(
                cubicEaseIn(Math.min(1, clipProgress)),
                [0, 1],
                [1.0, 1.06],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              )
            : 1.0;

          // Impact frames: brief brightness flash on first 2 frames
          const impactFlash = clip.isImpact && clipLocalFrame < 2
            ? "brightness(1.3)"
            : "";

          return (
            <Sequence
              key={clip.index}
              from={clip.startFrame}
              durationInFrames={clip.renderDuration}
            >
              <AbsoluteFill style={{
                transform: `scale(${zoom})`,
                filter: impactFlash,
              }}>
                <Video
                  src={staticFile(clip.src)}
                  startFrom={clip.trimStartFrames || 0}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              </AbsoluteFill>
            </Sequence>
          );
        })}
      </AbsoluteFill>

      {/* Audio */}
      {musicPath && (
        <Audio src={staticFile(musicPath)} volume={musicVolume} />
      )}

    </AbsoluteFill>
  );
};

export default VideoTemplate;
