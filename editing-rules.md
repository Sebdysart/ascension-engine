# Ascension Engine — Editing Rules v2.0

## 1. Hook Rules (First 1.5 Seconds — Non-Negotiable)

- **Movement must begin in frame 1.** Static opening = immediate scroll. Use zoom, pan, or cut within first 12 frames.
- **Text overlay must appear within 1.5s.** Hook text should be provocative, self-improvement framed: "I CHANGED MY FACE IN 90 DAYS", "FROM AVERAGE TO AESTHETIC", "THE GLOW-UP PROTOCOL"
- **Sound must hit immediately.** Phonk drop or beat hit on frame 1. No fade-in intros.
- **First frame must contain a face or transformation visual.** No B-roll openings.
- Hook types ranked by retention (exploit order): before_after_zoom > text_punch > mogging_comparison > transformation_reveal > stats_overlay

## 2. Cut Timing Rules

- **Default cut rate**: 1.8–2.2s between cuts during body.
- **Hook section (0–4s)**: 1.0–1.5s cuts. Aggressive pacing.
- **Climax/reveal**: Single slow cut or hold (2.5–4s) for contrast effect.
- **Beat alignment**: Cuts MUST land on beat downbeats when BPM ≥ 140. Use librosa beat data from ingest.
- **No jump cuts within same scene** unless stylized glitch effect.
- **Zoom pattern**: Every 3rd cut should include a subtle push-in (1.0x → 1.08x over 20 frames).

## 3. Music Rules

- **Genre priority**: Phonk > Hyperpop > Drill > Trap > Ambient electronic
- **BPM range**: 140–165 BPM for main content. 120–140 for skin/hair transformation pieces.
- **Beat cut alignment**: Required when BPM is detected. Tolerance: ±2 frames.
- **Volume balance**: Music at 75%, voice-over at 100%. Duck music to 40% under VO sentences.
- **Track sourcing**: Prefer TikTok trending audio. Use royalty-free phonk packs as fallback.
- **No copyrighted music on YouTube Shorts** without Content ID license confirmation.

## 4. Caption Rules

- **Font**: Impact, bold, ALL CAPS only.
- **Glow effect**: White glow blur 6px, black stroke 3px. Required — no plain text.
- **Max 18 characters per line**. Break at natural speech pause, not mid-word.
- **Position**: Bottom third of frame. Never overlap face in hero shots.
- **Animation**: Punch-in (scale 1.2 → 1.0 over 6 frames). No slow fades.
- **Duration**: Caption visible for full spoken word duration + 4 frames buffer.
- **Forbidden caption content**: See Forbidden Content List below.

## 5. Archetype-Specific Rules

| Archetype | Pacing | Color Grade | Hook Type | Music BPM |
|-----------|--------|-------------|-----------|-----------|
| glow_up | Fast (1.8s) | Teal/Orange | before_after_zoom | 148–160 |
| frame_maxxing | Medium (2.2s) | Cold Blue | stats_overlay | 140–155 |
| skin_maxxing | Slow (2.8s) | Warm Gold | transformation_reveal | 120–140 |
| style_maxxing | Medium (2.0s) | Desaturated | text_punch | 140–160 |

## 6. Forbidden Content List

The following are non-negotiable prohibitions. Claude Brain must refuse any prompt containing these:

**Language forbidden**:
- Any slur (racial, gender, sexual orientation, disability)
- Incel terminology: redpill, blackpill, femoid, hypergamy (as negative), Chad/Stacy (used to demean)
- Statements implying women/others are inferior or objects
- Self-harm language or implications
- Targeting real named individuals negatively

**Visual forbidden**:
- Before/after comparisons that mock the "before" state
- Unrealistic body modification implications (surgical without disclosure)
- Content designed to make viewer feel inadequate rather than motivated

**Framing forbidden**:
- "You'll never get girls unless..." framing
- Blackpill nihilism ("it's over", "cope")
- Any implication that looks determine human worth

**Allowed edge**:
- Mogging comparisons (you vs. best self, not vs. others)
- Discipline-heavy language ("no excuses", "built different")
- Aesthetic transformation content (skincare, jawline, posture, style)

## 7. Weekly Mutation Rules

Every Sunday the engine must run a mutation batch:

- **Archive losers**: Any hook_type or color_grade with avg_watch_pct < 0.35 over 14+ days → demote to explore-only (weight 0.05)
- **Promote winners**: Top 2 patterns by avg_watch_pct → increase exploit_weight to 0.80 for that pattern
- **Force 30% new experiments**: Minimum 3 of 10 weekly videos must use an untested combination
- **Rotate music**: If same track used > 5 times in 14 days → retire it, add new track
- **Caption variant test**: Every mutation week, test one new caption animation style
- **Reversion rule**: If a mutation batch produces avg_watch_pct drop > 15% week-over-week → revert style-profile.json to previous version and run analysis before next mutation

## 8. Quality Gate (Pre-Post Checklist)

Before any video is approved for posting, verify:

- [ ] Hook has movement + text within 1.5s
- [ ] Music hits on frame 1
- [ ] No forbidden content in captions or visuals
- [ ] CTA present in final 3s
- [ ] Color grade applied
- [ ] Beat cuts aligned
- [ ] File named: `{archetype}_{hook_type}_{YYYYMMDD}_{variant}.mp4`
- [ ] Predicted hook score logged to analytics DB
- [ ] experiment_flag set correctly (exploit/explore)
