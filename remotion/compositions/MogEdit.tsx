/**
 * MogEdit.tsx — Selfie Mog + Cinematic Mog archetype
 * Validated: @bp_masterx 388.5K (natural_indoor), @roge.editz 285.9K (dark_cinema)
 *
 * Formula: face clip(s) + grade + optional lyric sync + no CTA + watermark
 */
import {
  AbsoluteFill, Sequence, Video, Audio,
  useCurrentFrame, useVideoConfig, interpolate, staticFile,
} from "remotion";
import React from "react";

const GRADES: Record<string, string> = {
  dark_cinema:    "brightness(0.75) contrast(1.35) saturate(0.85) sepia(0.15)",
  natural:        "saturate(1.05) contrast(1.05) brightness(1.0)",
  natural_indoor: "brightness(0.95) contrast(1.08) saturate(1.0)",
  warm_ambient:   "saturate(1.1) sepia(0.2) brightness(0.85) contrast(1.2) hue-rotate(-5deg)",
  night_car:      "brightness(0.82) contrast(1.2) saturate(0.95)",
};

export interface MogEditProps {
  clips: { src: string; trimStart?: number; duration?: number }[];
  colorGrade: keyof typeof GRADES;
  musicPath?: string;
  caption?: string;
  watermark?: string;
  lyricText?: string;
  showLyric?: boolean;
}

// Subtle push-in zoom on each clip
const ZoomedClip: React.FC<{ src: string; trimStart?: number }> = ({ src, trimStart = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const zoom = interpolate(frame, [0, fps * 3], [1.0, 1.06], { extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ transform: `scale(${zoom})`, overflow: "hidden" }}>
      <Video
        src={staticFile(src)}
        startFrom={trimStart}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    </AbsoluteFill>
  );
};

// *pretty boy makes — lyric sync style
const LyricText: React.FC<{ text: string }> = ({ text }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 8, 50, 60], [0, 1, 1, 0], { extrapolateRight: "clamp" });
  return (
    <div style={{
      position: "absolute",
      top: "50%", left: 40, right: 40,
      transform: "translateY(-50%)",
      textAlign: "center",
      fontFamily: "Helvetica Neue, Arial, sans-serif",
      fontWeight: 300,
      fontSize: "2.2rem",
      color: "rgba(255,255,255,0.85)",
      letterSpacing: "0.02em",
      opacity,
      zIndex: 10,
    }}>
      {text}
    </div>
  );
};

// Watermark — account handle, bottom center
const Watermark: React.FC<{ text: string }> = ({ text }) => (
  <div style={{
    position: "absolute",
    bottom: 220,
    left: 0, right: 0,
    textAlign: "center",
    fontFamily: "Helvetica Neue, Arial, sans-serif",
    fontWeight: 500,
    fontSize: "1.1rem",
    letterSpacing: "0.15em",
    color: "rgba(255,255,255,0.45)",
    zIndex: 20,
  }}>
    {text.toUpperCase()}
  </div>
);

export const MogEdit: React.FC<MogEditProps> = ({
  clips = [],
  colorGrade = "dark_cinema",
  musicPath,
  watermark = "",
  lyricText = "",
  showLyric = false,
}) => {
  const { fps, durationInFrames } = useVideoConfig();
  const cutFrames = Math.round(2.0 * fps); // 2s cuts

  const filter = GRADES[colorGrade] || GRADES.dark_cinema;

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* Graded video layer */}
      <AbsoluteFill style={{ filter }}>
        {clips.map((clip, i) => {
          const from = i * cutFrames;
          if (from >= durationInFrames) return null;
          const dur = Math.min(cutFrames, durationInFrames - from);
          return (
            <Sequence key={i} from={from} durationInFrames={dur}>
              <ZoomedClip src={clip.src} trimStart={clip.trimStart || 0} />
            </Sequence>
          );
        })}
      </AbsoluteFill>

      {/* Lyric sync text (optional) */}
      {showLyric && lyricText && (
        <Sequence from={Math.round(fps * 1.5)} durationInFrames={Math.round(fps * 3)}>
          <LyricText text={lyricText} />
        </Sequence>
      )}

      {/* Watermark */}
      {watermark && <Watermark text={watermark} />}

      {/* Music */}
      {musicPath && <Audio src={staticFile(musicPath)} volume={0.8} />}
    </AbsoluteFill>
  );
};
