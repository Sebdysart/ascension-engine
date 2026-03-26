/**
 * Ascension Engine v4.0 — BP/Mog Edit Type System
 */

export interface BPCutPoint {
  time_sec: number;
  duration_sec: number;
  on_beat: boolean;
  beat_offset_ms: number;
  tags: string[];
}

export interface BrutalBPBlueprint {
  video_id: string;
  source_creator: string;
  total_duration_sec: number;
  bpm: number;
  beat_times_sec: number[];
  cut_rhythm: {
    avg_cut_sec: number;
    total_cuts: number;
    cuts_on_beat_pct: number;
    beat_offset_avg_ms: number;
    cut_points: BPCutPoint[];
  };
  visual_grade: {
    color_grade: string;
    description: string;
  };
  audio: {
    peak_moments_sec: number[];
    onset_times_sec: number[];
    energy_profile: string;
  };
  motion_fx: {
    zoom_punch: boolean;
    zoom_easing: string;
    impact_hold_ms: number;
  };
  ingest_timestamp: string;
}

export interface BPClipTag {
  tag: string;
  confidence: number;
}

export interface BPClip {
  clip_id: string;
  source_video_id: string;
  scene_index: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  file: string;
  thumbnail: string;
  tags: string[];
  rank: number;
  beat_aligned: boolean;
  nearest_beat_offset_ms: number;
  ingest_timestamp: string;
}

export interface BPTimeline {
  clips: BPTimelineEntry[];
  total_frames: number;
  total_duration_sec: number;
  cuts_on_beat_pct: number;
  avg_cut_sec: number;
  color_grade: string;
}

export interface BPTimelineEntry {
  clip_id: string;
  file: string;
  startFrame: number;
  durationFrames: number;
  trimStartFrames: number;
  on_beat: boolean;
  is_impact: boolean;
}
