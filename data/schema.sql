-- Ascension Engine v2.0 — SQLite Analytics Schema
-- Run: sqlite3 data/analytics.db < data/schema.sql

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ─────────────────────────────────────────────
-- Core video performance table
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS videos (
  video_id              TEXT PRIMARY KEY,
  title                 TEXT,
  hook_type             TEXT NOT NULL,
  archetype             TEXT NOT NULL,
  cut_rate              REAL,
  font_style            TEXT,
  color_grade           TEXT,
  music_bpm             INTEGER,
  music_track           TEXT,
  platform              TEXT NOT NULL DEFAULT 'tiktok',
  posted_at             TEXT,                          -- ISO8601
  views_24h             INTEGER DEFAULT 0,
  views_7d              INTEGER DEFAULT 0,
  avg_watch_pct         REAL,                          -- 0.0–1.0
  retention_dropoff     TEXT,                          -- JSON array of {sec, pct}
  ctr_thumbnail         REAL,                          -- 0.0–1.0
  sentiment_score       REAL,                          -- -1.0 to 1.0
  comment_count         INTEGER DEFAULT 0,
  experiment_flag       TEXT DEFAULT 'exploit'         CHECK(experiment_flag IN ('exploit','explore')),
  style_profile_version TEXT DEFAULT '2.0',
  runway_used           INTEGER DEFAULT 0              CHECK(runway_used IN (0,1)),
  render_time_sec       REAL,
  file_path             TEXT,
  notes                 TEXT,
  created_at            TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────
-- Weekly analysis results
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS weekly_analysis (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  week_start            TEXT NOT NULL,                 -- ISO8601 date
  week_end              TEXT NOT NULL,
  total_videos          INTEGER,
  avg_watch_pct         REAL,
  top_hook_type         TEXT,
  top_color_grade       TEXT,
  top_archetype         TEXT,
  top_bpm_range         TEXT,
  winning_patterns      TEXT,                          -- JSON
  recommendations       TEXT,                          -- JSON
  style_profile_delta   TEXT,                          -- JSON diff applied
  created_at            TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────
-- Mutation log
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mutation_log (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  mutation_date         TEXT NOT NULL,
  type                  TEXT NOT NULL                  CHECK(type IN ('promote','demote','retire','new_experiment','reversion')),
  target                TEXT NOT NULL,                 -- what was changed (hook_type, color_grade, etc.)
  old_value             TEXT,
  new_value             TEXT,
  reason                TEXT,
  avg_watch_pct_before  REAL,
  avg_watch_pct_after   REAL,
  created_at            TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────
-- Content queue (planned but not yet rendered)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS content_queue (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  hook_type             TEXT NOT NULL,
  archetype             TEXT NOT NULL,
  color_grade           TEXT,
  music_bpm             INTEGER,
  experiment_flag       TEXT DEFAULT 'exploit',
  script_notes          TEXT,
  status                TEXT DEFAULT 'queued'          CHECK(status IN ('queued','rendering','rendered','posted','rejected')),
  priority              INTEGER DEFAULT 5,
  scheduled_for         TEXT,                          -- ISO8601
  created_at            TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────
-- Trend research log
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trend_research (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  research_date         TEXT NOT NULL,
  platform              TEXT DEFAULT 'tiktok',
  top_hooks             TEXT,                          -- JSON array
  top_music             TEXT,                          -- JSON array
  top_creators          TEXT,                          -- JSON array
  avg_views_top20       INTEGER,
  notes                 TEXT,
  created_at            TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_videos_platform        ON videos(platform);
CREATE INDEX IF NOT EXISTS idx_videos_hook_type       ON videos(hook_type);
CREATE INDEX IF NOT EXISTS idx_videos_archetype       ON videos(archetype);
CREATE INDEX IF NOT EXISTS idx_videos_posted_at       ON videos(posted_at);
CREATE INDEX IF NOT EXISTS idx_videos_experiment      ON videos(experiment_flag);
CREATE INDEX IF NOT EXISTS idx_videos_watch_pct       ON videos(avg_watch_pct);

-- ─────────────────────────────────────────────
-- Analytical views
-- ─────────────────────────────────────────────

-- Top performers (last 14 days)
CREATE VIEW IF NOT EXISTS top_performers_14d AS
SELECT
  video_id, hook_type, archetype, color_grade, music_bpm,
  avg_watch_pct, views_7d, ctr_thumbnail, experiment_flag
FROM videos
WHERE posted_at >= date('now', '-14 days')
  AND avg_watch_pct IS NOT NULL
ORDER BY avg_watch_pct DESC;

-- Pattern performance aggregates
CREATE VIEW IF NOT EXISTS pattern_performance AS
SELECT
  hook_type,
  color_grade,
  archetype,
  COUNT(*) as video_count,
  ROUND(AVG(avg_watch_pct), 4) as avg_watch_pct,
  ROUND(AVG(ctr_thumbnail), 4) as avg_ctr,
  ROUND(AVG(views_7d), 0) as avg_views_7d
FROM videos
WHERE avg_watch_pct IS NOT NULL
  AND posted_at >= date('now', '-30 days')
GROUP BY hook_type, color_grade, archetype
ORDER BY avg_watch_pct DESC;

-- Weekly trend view
CREATE VIEW IF NOT EXISTS weekly_trends AS
SELECT
  strftime('%Y-W%W', posted_at) as week,
  COUNT(*) as videos_posted,
  ROUND(AVG(avg_watch_pct), 4) as avg_watch_pct,
  ROUND(AVG(views_7d), 0) as avg_views,
  SUM(CASE WHEN experiment_flag='explore' THEN 1 ELSE 0 END) as explore_count,
  SUM(CASE WHEN experiment_flag='exploit' THEN 1 ELSE 0 END) as exploit_count
FROM videos
WHERE avg_watch_pct IS NOT NULL
GROUP BY week
ORDER BY week DESC;

-- Bandit arm stats
CREATE VIEW IF NOT EXISTS bandit_arms AS
SELECT
  hook_type || '|' || color_grade as arm,
  hook_type, color_grade,
  COUNT(*) as pulls,
  ROUND(AVG(avg_watch_pct), 4) as mean_reward,
  ROUND(MAX(avg_watch_pct), 4) as max_reward
FROM videos
WHERE avg_watch_pct IS NOT NULL
GROUP BY hook_type, color_grade
HAVING pulls >= 3
ORDER BY mean_reward DESC;

-- ─────────────────────────────────────────────
-- Seed row (sanity check)
-- ─────────────────────────────────────────────
INSERT OR IGNORE INTO videos (
  video_id, title, hook_type, archetype, cut_rate, color_grade,
  music_bpm, platform, experiment_flag, style_profile_version, notes
) VALUES (
  'seed_001', 'Schema test row', 'text_punch', 'glow_up', 2.0, 'teal_orange',
  148, 'tiktok', 'explore', '2.0', 'DELETE before going live'
);
