/**
 * BpRating.tsx — BP Rating Reveal archetype
 * Validated: @bp4ever.ae 643.3K — 7.56% share rate (highest in library)
 *
 * Formula: face close-up + bold white Impact word reveals letter by letter
 * Text IS the hook — each frame reveals one more letter of the rating word
 */
import {
  AbsoluteFill, Video, Audio,
  useCurrentFrame, useVideoConfig, interpolate, staticFile,
} from "remotion";
import React from "react";

const GRADES: Record<string, string> = {
  natural:        "saturate(1.05) contrast(1.05) brightness(1.0)",
  dark_cinema:    "brightness(0.75) contrast(1.35) saturate(0.85) sepia(0.15)",
  dark_indoor:    "brightness(0.8) contrast(1.25) saturate(0.9)",
  natural_indoor: "brightness(0.95) contrast(1.08) saturate(1.0)",
};

export interface BpRatingProps {
  clipPath: string;
  revealWord: string;        // e.g. "MOGGED" — revealed letter by letter
  colorGrade: keyof typeof GRADES;
  musicPath?: string;
  watermark?: string;
}

// Reveals the word one letter at a time across the video duration
const WordReveal: React.FC<{ word: string }> = ({ word }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  // Each letter appears at equal intervals across the first 70% of video
  const revealDuration = durationInFrames * 0.7;
  const framesPerLetter = revealDuration / word.length;
  const lettersVisible = Math.min(
    word.length,
    Math.floor(frame / framesPerLetter) + 1
  );
  const visibleText = word.slice(0, lettersVisible);

  const opacity = interpolate(frame, [0, 6], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div style={{
      position: "absolute",
      top: "35%",
      left: 40,
      fontFamily: "Impact, Arial Black, sans-serif",
      fontWeight: 900,
      fontSize: "5.5rem",
      color: "#FFFFFF",
      letterSpacing: "0.05em",
      lineHeight: 1.0,
      opacity,
      zIndex: 10,
    }}>
      {visibleText}
    </div>
  );
};

const Watermark: React.FC<{ text: string }> = ({ text }) => (
  <div style={{
    position: "absolute",
    top: 60, left: 0, right: 0,
    textAlign: "center",
    fontFamily: "Helvetica Neue, Arial, sans-serif",
    fontWeight: 600,
    fontSize: "1.0rem",
    letterSpacing: "0.2em",
    color: "rgba(255,255,255,0.5)",
    zIndex: 20,
  }}>
    {text.toUpperCase()}
  </div>
);

export const BpRating: React.FC<BpRatingProps> = ({
  clipPath,
  revealWord = "MOGGED",
  colorGrade = "natural",
  musicPath,
  watermark = "",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const zoom = interpolate(frame, [0, fps * 10], [1.0, 1.05], { extrapolateRight: "clamp" });
  const filter = GRADES[colorGrade] || GRADES.natural;

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* Graded + zoomed clip */}
      <AbsoluteFill style={{ filter, transform: `scale(${zoom})`, overflow: "hidden" }}>
        {clipPath && (
          <Video
            src={staticFile(clipPath)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        )}
      </AbsoluteFill>

      {/* Word reveal */}
      <WordReveal word={revealWord.toUpperCase()} />

      {/* Watermark */}
      {watermark && <Watermark text={watermark} />}

      {/* Music */}
      {musicPath && <Audio src={staticFile(musicPath)} volume={0.75} />}
    </AbsoluteFill>
  );
};
