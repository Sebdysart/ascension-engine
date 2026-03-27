/**
 * BrutalBeatMontage.tsx — High-energy three-act phonk montage
 *
 * Visual effects not present in MogEdit.tsx:
 *   - Shake: spring-animated position offset (intensity 0–100)
 *   - Zoom pulse: scale 1.0→1.15→1.0 on bass drop (0.3s)
 *   - Color grade shift: CSS filter warm(5600K)→cold(4200K) across acts
 *   - Pre-drop hold: single-frame freeze 0.04–0.08s before drop
 *   - Act-aware grade: warm Act 1, neutral ramp Act 2, cold Act 3
 *
 * Props:
 *   slots      — NarrativeSlot array from build_narrative_sequence()
 *   hookSpec   — HookSpec JSON from hook_generator.py
 *   bpm        — Song BPM
 *   musicPath  — Audio file path
 *   watermark  — Optional account handle
 */
import React from "react";
import {
  AbsoluteFill,
  Sequence,
  Video,
  Audio,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  staticFile,
  spring,
} from "remotion";

// ── Grade filters per act ──────────────────────────────────────────────────────
const ACT_GRADES: Record<string, string> = {
  victim:    "brightness(1.4) saturate(1.1) sepia(0.15)",
  awakening: "brightness(1.0) saturate(0.85) contrast(1.1)",
  ascension: "brightness(0.7) saturate(0.7) contrast(1.35) hue-rotate(10deg)",
};

// ── Types ──────────────────────────────────────────────────────────────────────
export interface SlotSpec {
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  act: "victim" | "awakening" | "ascension";
  is_victim_slot: boolean;
  zoom_pulse: boolean;
  shake_intensity: number;    // 0–100
  slow_mo: boolean;
  pre_drop_silence_sec: number;
  clip_src?: string;          // filled in by generate_batch
  trim_start_sec?: number;
}

export interface HookSpecJson {
  flaw_clip_id: string;
  flaw_clip_path: string;
  zoom_start_pct: number;
  zoom_end_pct: number;
  zoom_duration_sec: number;
  bass_hit_time_sec: number;
  shake_peak: number;
  shake_duration_sec: number;
  zoom_pulse_peak_pct: number;
  zoom_pulse_duration_sec: number;
}

export interface BrutalBeatMontageProps {
  slots: SlotSpec[];
  hookSpec?: HookSpecJson;
  bpm: number;
  musicPath?: string;
  watermark?: string;
}

// ── Shake effect ──────────────────────────────────────────────────────────────
const ShakeWrapper: React.FC<{
  intensity: number;
  isActive: boolean;
  children: React.ReactNode;
}> = ({ intensity, isActive, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const shakeX = isActive
    ? spring({
        frame,
        fps,
        config: { damping: 8, stiffness: 300, mass: 0.3 },
      }) * (intensity * 0.08)
    : 0;

  const shakeY = isActive
    ? spring({
        frame: frame + 2,
        fps,
        config: { damping: 8, stiffness: 300, mass: 0.3 },
      }) * (intensity * 0.05)
    : 0;

  return (
    <AbsoluteFill
      style={{
        transform: `translate(${shakeX}px, ${shakeY}px)`,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

// ── Zoom pulse on drop ────────────────────────────────────────────────────────
const ZoomPulseWrapper: React.FC<{
  active: boolean;
  children: React.ReactNode;
}> = ({ active, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pulseDurationFrames = Math.round(0.3 * fps);
  const scale = active
    ? interpolate(
        frame,
        [0, Math.round(pulseDurationFrames * 0.4), pulseDurationFrames],
        [1.0, 1.15, 1.0],
        { extrapolateRight: "clamp" },
      )
    : 1.0;

  return (
    <AbsoluteFill style={{ transform: `scale(${scale})`, overflow: "hidden" }}>
      {children}
    </AbsoluteFill>
  );
};

// ── Hook zoom-out (frames 1-15) ───────────────────────────────────────────────
const HookZoomOut: React.FC<{
  src: string;
  trimStart: number;
  spec: HookSpecJson;
}> = ({ src, trimStart, spec }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const zoomDurFrames = Math.round(spec.zoom_duration_sec * fps);
  const scale = interpolate(
    frame,
    [0, zoomDurFrames],
    [spec.zoom_start_pct / 100, spec.zoom_end_pct / 100],
    { extrapolateRight: "clamp" },
  );

  return (
    <AbsoluteFill style={{ transform: `scale(${scale})`, overflow: "hidden" }}>
      <Video
        src={staticFile(src)}
        startFrom={Math.round(trimStart * fps)}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    </AbsoluteFill>
  );
};

// ── Single clip with grade ────────────────────────────────────────────────────
const GradedClip: React.FC<{
  src: string;
  trimStart: number;
  act: string;
  shakeIntensity: number;
  zoomPulse: boolean;
}> = ({ src, trimStart, act, shakeIntensity, zoomPulse }) => {
  const { fps } = useVideoConfig();
  const filter = ACT_GRADES[act] || ACT_GRADES.ascension;
  const isHighEnergy = shakeIntensity > 20;

  return (
    <AbsoluteFill style={{ filter }}>
      <ShakeWrapper intensity={shakeIntensity} isActive={isHighEnergy}>
        <ZoomPulseWrapper active={zoomPulse}>
          <AbsoluteFill>
            <Video
              src={staticFile(src)}
              startFrom={Math.round(trimStart * fps)}
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
          </AbsoluteFill>
        </ZoomPulseWrapper>
      </ShakeWrapper>
    </AbsoluteFill>
  );
};

// ── Watermark ────────────────────────────────────────────────────────────────
const Watermark: React.FC<{ text: string }> = ({ text }) => (
  <div
    style={{
      position: "absolute",
      bottom: 220,
      left: 0,
      right: 0,
      textAlign: "center",
      fontFamily: "Helvetica Neue, Arial, sans-serif",
      fontWeight: 500,
      fontSize: "1.1rem",
      letterSpacing: "0.15em",
      color: "rgba(255,255,255,0.45)",
      zIndex: 20,
    }}
  >
    {text.toUpperCase()}
  </div>
);

// ── Main composition ──────────────────────────────────────────────────────────
export const BrutalBeatMontage: React.FC<BrutalBeatMontageProps> = ({
  slots = [],
  hookSpec,
  bpm,
  musicPath,
  watermark = "",
}) => {
  const { fps, durationInFrames } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* Hook sequence — always first */}
      {hookSpec && slots.length > 0 && slots[0].clip_src && (
        <Sequence from={0} durationInFrames={Math.round(hookSpec.zoom_duration_sec * fps)}>
          <HookZoomOut
            src={slots[0].clip_src}
            trimStart={slots[0].trim_start_sec ?? 0}
            spec={hookSpec}
          />
        </Sequence>
      )}

      {/* Act slots */}
      {slots.map((slot, i) => {
        if (!slot.clip_src) return null;

        const fromFrame = Math.round(slot.start_sec * fps);
        const durFrames = Math.max(1, Math.round(slot.duration_sec * fps));
        if (fromFrame >= durationInFrames) return null;
        const actualDur = Math.min(durFrames, durationInFrames - fromFrame);

        return (
          <Sequence key={i} from={fromFrame} durationInFrames={actualDur}>
            <GradedClip
              src={slot.clip_src}
              trimStart={slot.trim_start_sec ?? 0}
              act={slot.act}
              shakeIntensity={slot.shake_intensity}
              zoomPulse={slot.zoom_pulse}
            />
          </Sequence>
        );
      })}

      {watermark && <Watermark text={watermark} />}

      {musicPath && <Audio src={staticFile(musicPath)} volume={0.85} />}
    </AbsoluteFill>
  );
};
