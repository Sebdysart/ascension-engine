/**
 * Ascension Engine v3.0 — VideoTemplate.tsx
 * Beat-synced BP/mog edit composition.
 *
 * Format: rapid-cut clip montage with dark cinema grading,
 * no CTA cards, zero or minimal text. The clips ARE the edit.
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
}

export interface VideoTemplateProps {
  bodyClips: ClipSegment[];
  musicPath?: string;
  colorGrade?: "dark_cinema" | "desaturated" | "cold_blue" | "warm_ambient" | "natural" | "teal_orange" | "none";
  zoomPunch?: boolean;
  overlayText?: string;
  overlayPosition?: "top" | "center" | "bottom";
  showOverlay?: boolean;
  musicVolume?: number;
}

// ── Color Grade CSS Filters ─────────────────────────────────────────────

const COLOR_GRADES: Record<string, string> = {
  dark_cinema:   "brightness(0.75) contrast(1.35) saturate(0.85) sepia(0.15)",
  desaturated:   "saturate(0.15) contrast(1.25) brightness(0.95)",
  cold_blue:     "saturate(0.9) hue-rotate(20deg) brightness(0.85) contrast(1.2)",
  warm_ambient:  "saturate(1.1) sepia(0.2) brightness(0.85) contrast(1.2) hue-rotate(-5deg)",
  natural:       "saturate(1.05) contrast(1.05) brightness(1.0)",
  teal_orange:   "saturate(1.3) hue-rotate(-10deg) contrast(1.15)",
  none:          "none",
};

// ── Main Composition ───────────────────────────────────────────────────────

export const VideoTemplate: React.FC<VideoTemplateProps> = ({
  bodyClips = [],
  musicPath,
  colorGrade = "dark_cinema",
  zoomPunch = true,
  overlayText,
  overlayPosition = "bottom",
  showOverlay = false,
  musicVolume = 0.85,
}) => {
  const { fps, durationInFrames } = useVideoConfig();
  const frame = useCurrentFrame();

  const cssFilter = COLOR_GRADES[colorGrade] || COLOR_GRADES.dark_cinema;

  // Build timeline: each clip plays sequentially
  let currentFrame = 0;
  const timeline = bodyClips.map((clip, i) => {
    const start = currentFrame;
    const dur = Math.min(clip.durationFrames, durationInFrames - start);
    currentFrame += dur;
    return { ...clip, startFrame: start, renderDuration: Math.max(1, dur), index: i };
  }).filter(c => c.startFrame < durationInFrames);

  return (
    <AbsoluteFill style={{ backgroundColor: "#000", overflow: "hidden" }}>

      {/* Color graded clip layer */}
      <AbsoluteFill style={{ filter: cssFilter }}>
        {timeline.map((clip) => {
          // Per-clip zoom punch: subtle push-in from 1.0 → 1.06
          const clipLocalFrame = frame - clip.startFrame;
          const zoom = zoomPunch
            ? interpolate(
                clipLocalFrame,
                [0, clip.renderDuration],
                [1.0, 1.06],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              )
            : 1.0;

          return (
            <Sequence
              key={clip.index}
              from={clip.startFrame}
              durationInFrames={clip.renderDuration}
            >
              <AbsoluteFill style={{ transform: `scale(${zoom})` }}>
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

      {/* Optional minimal text overlay (only if explicitly set) */}
      {showOverlay && overlayText && (
        <Sequence from={0} durationInFrames={Math.round(2.5 * fps)}>
          <div
            style={{
              position: "absolute",
              ...(overlayPosition === "top" ? { top: 80 } : {}),
              ...(overlayPosition === "center" ? { top: "50%", transform: "translateY(-50%)" } : {}),
              ...(overlayPosition === "bottom" ? { bottom: 200 } : {}),
              left: 20,
              right: 20,
              textAlign: "center",
              fontFamily: "Impact, Arial Black, sans-serif",
              fontSize: "2.4rem",
              fontWeight: "bold",
              color: "#FFFFFF",
              textTransform: "uppercase",
              letterSpacing: "0.03em",
              lineHeight: 1.1,
              textShadow: "3px 3px 0 #000, -1px -1px 0 #000",
              opacity: interpolate(frame, [0, 4, Math.round(2 * fps), Math.round(2.5 * fps)], [0, 1, 1, 0], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
              zIndex: 10,
            }}
          >
            {overlayText}
          </div>
        </Sequence>
      )}

      {/* Audio track */}
      {musicPath && (
        <Audio src={staticFile(musicPath)} volume={musicVolume} />
      )}

    </AbsoluteFill>
  );
};

export default VideoTemplate;
