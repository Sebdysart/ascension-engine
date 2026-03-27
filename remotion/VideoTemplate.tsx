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

/** A single timed caption entry matching editing-rules.md spec. */
export interface CaptionEntry {
  /** Raw caption text — will be auto-broken at 18 chars. */
  text: string;
  /** Frame at which this caption appears. */
  startFrame: number;
  /** How many frames the caption stays visible (spoken duration + 4 frame buffer). */
  durationFrames: number;
}

export interface VideoTemplateProps {
  bodyClips: ClipSegment[];
  /** Structured caption track (preferred — use instead of overlayTexts). */
  captions?: CaptionEntry[];
  /** Legacy overlay texts — still supported for backwards compatibility. */
  overlayTexts?: { text: string; startFrame: number; durationFrames: number }[];
  ctaText?: string;
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

// ── Caption utilities ───────────────────────────────────────────────────────

const CAPTION_MAX_CHARS = 18;

/**
 * Break text at natural word boundaries so no line exceeds CAPTION_MAX_CHARS.
 * Matches editing-rules.md §4: "Break at natural speech pause, not mid-word."
 */
function breakCaptionLines(text: string, maxChars: number = CAPTION_MAX_CHARS): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxChars) {
      current = candidate;
    } else {
      if (current) lines.push(current);
      // If a single word exceeds limit, hard-break it
      if (word.length > maxChars) {
        for (let i = 0; i < word.length; i += maxChars) {
          lines.push(word.slice(i, i + maxChars));
        }
        current = "";
      } else {
        current = word;
      }
    }
  }
  if (current) lines.push(current);
  return lines;
}

// ── GlowText — single caption block ────────────────────────────────────────

const GlowText: React.FC<{
  text: string;
  fontSize?: string;
  bottom?: number;
  opacity?: number;
}> = ({ text, fontSize = "3.2rem", bottom = 180, opacity = 1 }) => {
  const frame = useCurrentFrame();

  // Punch-in: scale 1.2 → 1.0 over 6 frames (editing-rules.md §4)
  const scale = interpolate(frame, [0, 6], [1.2, 1.0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const lines = breakCaptionLines(text.toUpperCase());

  return (
    <div
      style={{
        position: "absolute",
        bottom,
        left: 20,
        right: 20,
        textAlign: "center",
        fontFamily: "Impact, Arial Black, sans-serif",
        fontSize,
        fontWeight: "bold",
        color: "#FFFFFF",
        textTransform: "uppercase",
        letterSpacing: "0.04em",
        lineHeight: 1.1,
        // Glow effect: white glow 6px blur + black stroke 3px (editing-rules.md §4)
        textShadow: `
          0 0 6px rgba(255,255,255,0.8),
          0 0 12px rgba(255,255,255,0.4),
          3px 3px 0 #000,
          -3px -3px 0 #000,
          3px -3px 0 #000,
          -3px 3px 0 #000
        `,
        transform: `scale(${scale})`,
        opacity,
        zIndex: 10,
      }}
    >
      {lines.map((line, i) => (
        <div key={i}>{line}</div>
      ))}
    </div>
  );
};

// ── CaptionTrack — full timed caption system ────────────────────────────────

/**
 * Renders a full caption track from an array of CaptionEntry objects.
 * Each entry gets its own <Sequence> wrapper so punch-in resets per caption.
 * Positioning: lower third (bottom 180px) matching editing-rules.md §4.
 */
const CaptionTrack: React.FC<{
  captions: CaptionEntry[];
  fontSize?: string;
}> = ({ captions, fontSize = "3.0rem" }) => {
  return (
    <>
      {captions.map((caption, i) => (
        <Sequence
          key={`caption-${i}`}
          from={caption.startFrame}
          durationInFrames={caption.durationFrames}
        >
          <GlowText
            text={caption.text}
            fontSize={fontSize}
            bottom={180}
          />
        </Sequence>
      ))}
    </>
  );
};

const BeforeAfterReveal: React.FC<{
  beforeSrc: string;
  afterSrc: string;
  durationFrames: number;
}> = ({ beforeSrc, afterSrc, durationFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Wipe from left to right over 30 frames, starting at frame 20
  const wipeProgress = interpolate(
    frame,
    [20, 50],
    [0, 100],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Seam line opacity: visible during wipe
  const seamOpacity = interpolate(
    frame,
    [18, 22, 48, 52],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill>
      {/* Before image (full frame) */}
      <Img
        src={beforeSrc}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
      {/* After image (revealed via clip-path wipe) */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          clipPath: `inset(0 ${100 - wipeProgress}% 0 0)`,
        }}
      >
        <Img
          src={afterSrc}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </div>
      {/* Seam line */}
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${wipeProgress}%`,
          width: 3,
          background: "rgba(255,255,255,0.9)",
          opacity: seamOpacity,
          zIndex: 5,
        }}
      />
      {/* BEFORE label */}
      <div style={{
        position: "absolute", top: 60, left: 30,
        fontFamily: "Impact, sans-serif", fontSize: "1.6rem",
        color: "#fff", textShadow: "2px 2px 0 #000",
        opacity: interpolate(frame, [0, 5, 18, 22], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
      }}>BEFORE</div>
      {/* AFTER label */}
      <div style={{
        position: "absolute", top: 60, right: 30,
        fontFamily: "Impact, sans-serif", fontSize: "1.6rem",
        color: "#fff", textShadow: "2px 2px 0 #000",
        opacity: interpolate(frame, [48, 52], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
      }}>AFTER</div>
    </AbsoluteFill>
  );
};

const CTACard: React.FC<{ text: string }> = ({ text }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = spring({ frame, fps, config: { damping: 12 } });
  const translateY = interpolate(opacity, [0, 1], [30, 0]);

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "rgba(0,0,0,0.75)",
        justifyContent: "center",
        alignItems: "center",
        padding: "0 40px",
      }}
    >
      <div
        style={{
          transform: `translateY(${translateY}px)`,
          opacity,
          textAlign: "center",
          fontFamily: "Impact, Arial Black, sans-serif",
          fontSize: "2.2rem",
          color: "#FFFFFF",
          textTransform: "uppercase",
          lineHeight: 1.25,
          textShadow: `
            0 0 8px rgba(255,255,255,0.6),
            3px 3px 0 #000
          `,
          maxWidth: 900,
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  );
};

// ── Main Composition ───────────────────────────────────────────────────────

export const VideoTemplate: React.FC<VideoTemplateProps> = ({
  bodyClips = [],
  captions = [],
  overlayTexts = [],
  ctaText = "DISCIPLINE = RESULTS. DROP YOUR GLOW-UP PROGRESS BELOW.",
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

      {/* ── CAPTION TRACK (structured, editing-rules.md compliant) ── */}
      {captions.length > 0 && <CaptionTrack captions={captions} />}

      {/* ── OVERLAY TEXTS (legacy — rendered above color grade) ── */}
      {captions.length === 0 && overlayTexts.map((ot, i) => (
        <Sequence key={`ot-${i}`} from={ot.startFrame} durationInFrames={ot.durationFrames}>
          <GlowText text={ot.text} fontSize="2.8rem" bottom={160} />
        </Sequence>
      ))}

      {/* ── SEQUENCE 4: CTA ── */}
      <Sequence
        from={durationInFrames - ctaDurationFrames}
        durationInFrames={ctaDurationFrames}
      >
        <CTACard text={ctaText} />
      </Sequence>

      {/* ── AUDIO ── */}
      {musicPath && (
        <Audio src={staticFile(musicPath)} volume={musicVolume} />
      )}

    </AbsoluteFill>
  );
};

export default VideoTemplate;
