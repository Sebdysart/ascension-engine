/**
 * Ascension Engine — BrutalBeatMontage.tsx
 * EDL-driven Remotion composition for beat-aligned BP/mog edits.
 *
 * Accepts an EditDecisionList produced by data/sequencer.py and renders
 * each clip segment at its EDL-specified cut time with section-specific
 * transition effects.
 *
 * Transitions:
 *   glitch      (drop)    — chromatic aberration flash over 4 frames
 *   zoom_in_1.2x (buildup) — scale 1.0→1.2 over the clip duration
 *   hard_cut    (verse)   — instant cut, no effect
 *
 * Register in remotion/index.tsx:
 *   import { BrutalBeatMontage } from "./BrutalBeatMontage";
 *   <Composition
 *     id="BrutalBeatMontage"
 *     component={BrutalBeatMontage}
 *     durationInFrames={450}
 *     fps={30}
 *     width={1080}
 *     height={1920}
 *     defaultProps={{ edl: SAMPLE_EDL, audioPath: "" }}
 *   />
 */

import React from "react";
import {
  AbsoluteFill,
  Audio,
  Sequence,
  Video,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  staticFile,
} from "remotion";

// ── EDL Types ─────────────────────────────────────────────────────────────────

export type TransitionType = "glitch" | "zoom_in_1.2x" | "hard_cut";
export type SectionType    = "drop" | "buildup" | "verse";

export interface EDLEntry {
  /** Zero-based slot index. */
  slot: number;
  /** Absolute path or staticFile-relative path to the clip. */
  clip_path: string;
  /** Frame offset into the source clip to start playback from. */
  start_frame: number;
  /** How many frames this slot spans in the output timeline. */
  duration_frames: number;
  /** Adjusted beat time this slot starts at (seconds). */
  cut_time: number;
  /** Audio section this slot belongs to. */
  section: SectionType;
  /** Transition applied at the start of this slot. */
  transition: TransitionType;
  /** Composite impact score [0–1]. */
  impact_score: number;
  /** Face yaw angle in degrees, or null if unavailable. */
  face_angle: number | null;
}

export interface EditDecisionList {
  tempo: number;
  beat_grid: Array<{
    time: number;
    beat_num: number;
    section_type: SectionType;
    micro_offset_ms: number;
    adjusted_time: number;
  }>;
  template_id: string;
  target_duration: number;
  total_duration: number;
  fps: number;
  edl: EDLEntry[];
  stats: {
    slot_count: number;
    beats_per_cut: number;
    avg_cut_sec: number;
    transitions: Record<string, number>;
  };
}

export interface MontageProps {
  edl: EditDecisionList;
  /** Path to audio file (relative to /public or absolute). Empty = no audio. */
  audioPath: string;
}

// ── Glitch overlay ────────────────────────────────────────────────────────────

/**
 * Chromatic aberration flash: three colour-shifted copies of the frame
 * converge to normal over `durationFrames` frames.
 * Applied as a CSS filter/transform overlay — does not affect the source video.
 */
const GlitchOverlay: React.FC<{ durationFrames?: number }> = ({
  durationFrames = 4,
}) => {
  const frame = useCurrentFrame();
  if (frame >= durationFrames) return null;

  const progress = frame / durationFrames;
  const strength = interpolate(progress, [0, 1], [8, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ pointerEvents: "none", mixBlendMode: "screen" }}>
      {/* Red channel — shift left */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(255,0,0,0.25)",
          transform: `translateX(${-strength}px)`,
          opacity: 1 - progress,
        }}
      />
      {/* Blue channel — shift right */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(0,0,255,0.25)",
          transform: `translateX(${strength}px)`,
          opacity: 1 - progress,
        }}
      />
      {/* White flash on first frame */}
      {frame === 0 && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(255,255,255,0.35)",
          }}
        />
      )}
    </AbsoluteFill>
  );
};

// ── Zoom effect ───────────────────────────────────────────────────────────────

/**
 * Gradual 1.0→1.2 scale spring over the clip duration.
 * Wraps the clip in a scaled container.
 */
const ZoomEffect: React.FC<{
  durationFrames: number;
  children: React.ReactNode;
}> = ({ durationFrames, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame,
    fps,
    config: { damping: 200, stiffness: 20, mass: 1 },
    from: 1.0,
    to: 1.2,
    durationInFrames: durationFrames,
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        transform: `scale(${scale})`,
        transformOrigin: "50% 50%",
      }}
    >
      {children}
    </div>
  );
};

// ── Single EDL slot ───────────────────────────────────────────────────────────

const EDLSlot: React.FC<{ entry: EDLEntry }> = ({ entry }) => {
  const frame = useCurrentFrame();

  const videoSrc = entry.clip_path.startsWith("/")
    ? entry.clip_path
    : staticFile(entry.clip_path);

  const videoEl = (
    <Video
      src={videoSrc}
      startFrom={entry.start_frame}
      endAt={entry.start_frame + entry.duration_frames}
      style={{ width: "100%", height: "100%", objectFit: "cover" }}
    />
  );

  return (
    <AbsoluteFill>
      {/* Video layer */}
      {entry.transition === "zoom_in_1.2x" ? (
        <ZoomEffect durationFrames={entry.duration_frames}>
          {videoEl}
        </ZoomEffect>
      ) : (
        videoEl
      )}

      {/* Transition overlay (rendered on top) */}
      {entry.transition === "glitch" && <GlitchOverlay durationFrames={4} />}

      {/* Optional: impact heat indicator (debug, hidden in prod) */}
      {process.env.NODE_ENV === "development" && (
        <div
          style={{
            position: "absolute",
            bottom: 8,
            right: 8,
            background: `rgba(255,${Math.round((1 - entry.impact_score) * 255)},0,0.7)`,
            color: "#fff",
            fontSize: 11,
            padding: "2px 5px",
            borderRadius: 3,
            fontFamily: "monospace",
          }}
        >
          {entry.impact_score.toFixed(3)} {entry.section}
        </div>
      )}
    </AbsoluteFill>
  );
};

// ── Main composition ──────────────────────────────────────────────────────────

export const BrutalBeatMontage: React.FC<MontageProps> = ({
  edl,
  audioPath,
}) => {
  const { fps } = useVideoConfig();

  // Accumulate frame offsets for each EDL slot in the output timeline
  let timelineCursor = 0;
  const timelineSlots: Array<{ from: number; entry: EDLEntry }> = [];

  for (const entry of edl.edl) {
    timelineSlots.push({ from: timelineCursor, entry });
    timelineCursor += entry.duration_frames;
  }

  return (
    <AbsoluteFill style={{ background: "#000" }}>
      {/* Audio track */}
      {audioPath && (
        <Audio
          src={audioPath.startsWith("/") ? audioPath : staticFile(audioPath)}
          startFrom={0}
        />
      )}

      {/* EDL slots */}
      {timelineSlots.map(({ from, entry }) => (
        <Sequence
          key={entry.slot}
          from={from}
          durationInFrames={entry.duration_frames}
          layout="none"
        >
          <EDLSlot entry={entry} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

// ── Sample EDL for Remotion Studio preview ────────────────────────────────────

export const SAMPLE_EDL: EditDecisionList = {
  tempo: 92.3,
  beat_grid: [],
  template_id: "preview",
  target_duration: 15,
  total_duration: 12.8,
  fps: 30,
  edl: [
    {
      slot: 0,
      clip_path: "library/clips/bp4ever.ae_7542208986699844869_0cb59b00_scene000.mp4",
      start_frame: 0,
      duration_frames: 91,
      cut_time: 0.07,
      section: "verse",
      transition: "hard_cut",
      impact_score: 0.317,
      face_angle: null,
    },
    {
      slot: 1,
      clip_path: "library/clips/bp4ever.ae_7542208986699844869_0cb59b00_scene003.mp4",
      start_frame: 45,
      duration_frames: 97,
      cut_time: 3.11,
      section: "drop",
      transition: "glitch",
      impact_score: 0.626,
      face_angle: null,
    },
    {
      slot: 2,
      clip_path: "library/clips/bp4ever.ae_7542208986699844869_0cb59b00_scene001.mp4",
      start_frame: 20,
      duration_frames: 98,
      cut_time: 6.34,
      section: "buildup",
      transition: "zoom_in_1.2x",
      impact_score: 0.243,
      face_angle: null,
    },
    {
      slot: 3,
      clip_path: "library/clips/bp4ever.ae_7542208986699844869_0cb59b00_scene004.mp4",
      start_frame: 0,
      duration_frames: 98,
      cut_time: 9.59,
      section: "verse",
      transition: "hard_cut",
      impact_score: 0.351,
      face_angle: null,
    },
  ],
  stats: {
    slot_count: 4,
    beats_per_cut: 5,
    avg_cut_sec: 3.2,
    transitions: { hard_cut: 3, glitch: 1 },
  },
};

export default BrutalBeatMontage;
