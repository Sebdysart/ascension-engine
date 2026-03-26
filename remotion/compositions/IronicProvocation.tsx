/**
 * IronicProvocation.tsx — Ironic Provocation archetype
 * Validated: @bp.editz04 259.7K — 5.9% save + 5.6% share (debate content)
 *
 * Formula: face close-up + centered subtitle "unpopular opinion / [claim]"
 * The self-referential irony IS the hook.
 */
import {
  AbsoluteFill, Video, Audio,
  useCurrentFrame, useVideoConfig, interpolate, staticFile,
} from "remotion";
import React from "react";

const GRADES: Record<string, string> = {
  dark_indoor:    "brightness(0.8) contrast(1.25) saturate(0.9)",
  dark_cinema:    "brightness(0.75) contrast(1.35) saturate(0.85) sepia(0.15)",
  natural:        "saturate(1.05) contrast(1.05) brightness(1.0)",
  warm_ambient:   "saturate(1.1) sepia(0.2) brightness(0.85) contrast(1.2) hue-rotate(-5deg)",
};

export interface IronicProvocationProps {
  clipPath: string;
  line1: string;   // "unpopular opinion"
  line2: string;   // "I've never seen a pretty blond"
  colorGrade: keyof typeof GRADES;
  musicPath?: string;
  textAppearFrame?: number;
}

// Subtitle-style centered white text — NOT Impact, rounded/clean
const SubtitleText: React.FC<{ line1: string; line2: string }> = ({ line1, line2 }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 10, 80, 90], [0, 1, 1, 0.8], { extrapolateRight: "clamp" });

  return (
    <div style={{
      position: "absolute",
      bottom: "28%",
      left: 30, right: 30,
      textAlign: "center",
      zIndex: 10,
      opacity,
    }}>
      <div style={{
        display: "inline-block",
        backgroundColor: "rgba(0,0,0,0.55)",
        borderRadius: 6,
        padding: "12px 24px",
      }}>
        <div style={{
          fontFamily: "Helvetica Neue, Arial, sans-serif",
          fontWeight: 400,
          fontSize: "1.55rem",
          color: "#FFFFFF",
          lineHeight: 1.5,
          letterSpacing: "0.01em",
        }}>
          {line1}
        </div>
        <div style={{
          fontFamily: "Helvetica Neue, Arial, sans-serif",
          fontWeight: 500,
          fontSize: "1.55rem",
          color: "#FFFFFF",
          lineHeight: 1.5,
          letterSpacing: "0.01em",
        }}>
          {line2}
        </div>
      </div>
    </div>
  );
};

export const IronicProvocation: React.FC<IronicProvocationProps> = ({
  clipPath,
  line1 = "unpopular opinion",
  line2 = "I've never seen a pretty blond",
  colorGrade = "dark_indoor",
  musicPath,
  textAppearFrame = 15,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const zoom = interpolate(frame, [0, fps * 10], [1.0, 1.04], { extrapolateRight: "clamp" });
  const filter = GRADES[colorGrade] || GRADES.dark_indoor;

  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a0a" }}>
      {/* Graded clip */}
      <AbsoluteFill style={{ filter, transform: `scale(${zoom})`, overflow: "hidden" }}>
        {clipPath && (
          <Video
            src={staticFile(clipPath)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        )}
      </AbsoluteFill>

      {/* Subtitle text */}
      {frame >= textAppearFrame && (
        <SubtitleText line1={line1} line2={line2} />
      )}

      {/* Music */}
      {musicPath && <Audio src={staticFile(musicPath)} volume={0.75} />}
    </AbsoluteFill>
  );
};
