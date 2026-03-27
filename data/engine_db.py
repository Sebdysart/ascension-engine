#!/usr/bin/env python3
"""
Ascension Engine v4.2 — Central Engine Database
SQLite single source of truth for all library metadata.
Makes ingest fully idempotent and crash-recoverable.

Usage:
    from data.engine_db import EngineDB
    db = EngineDB()
    db.init()
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger("engine_db")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "library" / "engine.db"


class EngineDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init(self):
        """Create all tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS ingests (
                video_id        TEXT PRIMARY KEY,
                source_file     TEXT NOT NULL,
                source_creator  TEXT,
                file_hash       TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending'
                                CHECK(status IN ('pending','scene_detect','audio','color','clips','complete','failed')),
                bpm             INTEGER DEFAULT 0,
                color_grade     TEXT,
                total_clips     INTEGER DEFAULT 0,
                cuts_on_beat_pct REAL DEFAULT 0,
                error_msg       TEXT,
                started_at      TEXT,
                completed_at    TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS clips (
                clip_id         TEXT PRIMARY KEY,
                video_id        TEXT NOT NULL REFERENCES ingests(video_id),
                scene_index     INTEGER,
                start_sec       REAL,
                end_sec         REAL,
                duration_sec    REAL,
                file_path       TEXT,
                thumbnail       TEXT,
                rank            REAL DEFAULT 0.5,
                phash           TEXT,
                track           TEXT DEFAULT 'unclassified'
                                CHECK(track IN ('unclassified','good_parts','victim_contrast','archived')),
                tags            TEXT DEFAULT '[]',
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS generations (
                gen_id          TEXT PRIMARY KEY,
                mode            TEXT NOT NULL CHECK(mode IN ('blueprint','remix')),
                style           TEXT,
                template_id     TEXT,
                blueprint_id    TEXT,
                total_cuts      INTEGER,
                on_beat_pct     REAL,
                fidelity_score  REAL,
                verdict         TEXT CHECK(verdict IN ('pass','fail','pending',NULL)),
                output_file     TEXT,
                approved        INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS health_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT DEFAULT (datetime('now')),
                total_clips     INTEGER,
                good_parts      INTEGER,
                victim_contrast INTEGER,
                total_ingests   INTEGER,
                avg_fidelity_7d REAL,
                pending_approvals INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id);
            CREATE INDEX IF NOT EXISTS idx_clips_track ON clips(track);
            CREATE INDEX IF NOT EXISTS idx_clips_rank ON clips(rank);
            CREATE INDEX IF NOT EXISTS idx_clips_phash ON clips(phash);
            CREATE INDEX IF NOT EXISTS idx_gen_verdict ON generations(verdict);
        """)
        self.conn.commit()
        log.info("Engine DB initialized at %s", self.db_path)

    def close(self):
        self.conn.close()

    # ── Ingest tracking ───────────────────────────────────────────────────────

    def is_ingested(self, file_hash: str) -> bool:
        """Check if a file has already been fully ingested (idempotency)."""
        row = self.conn.execute(
            "SELECT status FROM ingests WHERE file_hash = ? AND status = 'complete'",
            (file_hash,)
        ).fetchone()
        return row is not None

    def start_ingest(self, video_id: str, source_file: str, file_hash: str, creator: str = ""):
        self.conn.execute(
            "INSERT OR REPLACE INTO ingests (video_id, source_file, file_hash, source_creator, status, started_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (video_id, source_file, file_hash, creator, datetime.now(timezone.utc).isoformat())
        )
        self.conn.commit()

    def update_ingest_status(self, video_id: str, status: str, **kwargs):
        sets = ["status = ?"]
        vals = [status]
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(v)
        if status == "complete":
            sets.append("completed_at = ?")
            vals.append(datetime.now(timezone.utc).isoformat())
        vals.append(video_id)
        self.conn.execute(f"UPDATE ingests SET {', '.join(sets)} WHERE video_id = ?", vals)
        self.conn.commit()

    def mark_ingest_failed(self, video_id: str, error: str):
        self.update_ingest_status(video_id, "failed", error_msg=error)

    # ── Clip tracking ─────────────────────────────────────────────────────────

    def add_clip(self, clip_id: str, video_id: str, scene_index: int,
                 start_sec: float, end_sec: float, duration_sec: float,
                 file_path: str, thumbnail: str, phash: str = ""):
        self.conn.execute(
            "INSERT OR REPLACE INTO clips "
            "(clip_id, video_id, scene_index, start_sec, end_sec, duration_sec, file_path, thumbnail, phash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (clip_id, video_id, scene_index, start_sec, end_sec, duration_sec, file_path, thumbnail, phash)
        )
        self.conn.commit()

    def find_duplicate_phash(self, phash: str, threshold: int = 4) -> list[str]:
        """Find clips with similar perceptual hashes (hamming distance ≤ threshold)."""
        if not phash:
            return []
        rows = self.conn.execute(
            "SELECT clip_id, phash FROM clips WHERE phash != '' AND phash IS NOT NULL"
        ).fetchall()
        dupes = []
        for clip_id, existing_hash in rows:
            if existing_hash and _hamming_distance(phash, existing_hash) <= threshold:
                dupes.append(clip_id)
        return dupes

    def set_clip_track(self, clip_id: str, track: str):
        self.conn.execute("UPDATE clips SET track = ? WHERE clip_id = ?", (track, clip_id))
        self.conn.commit()

    def update_clip_rank(self, clip_id: str, rank: float):
        self.conn.execute("UPDATE clips SET rank = ? WHERE clip_id = ?", (min(1, max(0, rank)), clip_id))
        self.conn.commit()

    # ── Generation tracking ───────────────────────────────────────────────────

    def log_generation(self, gen_id: str, mode: str, style: str, total_cuts: int,
                       on_beat_pct: float, template_id: str = "", blueprint_id: str = "",
                       output_file: str = ""):
        self.conn.execute(
            "INSERT OR REPLACE INTO generations "
            "(gen_id, mode, style, template_id, blueprint_id, total_cuts, on_beat_pct, output_file) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (gen_id, mode, style, template_id, blueprint_id, total_cuts, on_beat_pct, output_file)
        )
        self.conn.commit()

    def set_generation_verdict(self, gen_id: str, fidelity_score: float, verdict: str):
        self.conn.execute(
            "UPDATE generations SET fidelity_score = ?, verdict = ? WHERE gen_id = ?",
            (fidelity_score, verdict, gen_id)
        )
        self.conn.commit()

    def approve_generation(self, gen_id: str):
        self.conn.execute("UPDATE generations SET approved = 1 WHERE gen_id = ?", (gen_id,))
        self.conn.commit()

    # ── Health queries ────────────────────────────────────────────────────────

    def get_health(self) -> dict:
        total_clips = self.conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        good_parts = self.conn.execute("SELECT COUNT(*) FROM clips WHERE track='good_parts'").fetchone()[0]
        victim = self.conn.execute("SELECT COUNT(*) FROM clips WHERE track='victim_contrast'").fetchone()[0]
        total_ingests = self.conn.execute("SELECT COUNT(*) FROM ingests WHERE status='complete'").fetchone()[0]
        failed = self.conn.execute("SELECT COUNT(*) FROM ingests WHERE status='failed'").fetchone()[0]
        pending = self.conn.execute("SELECT COUNT(*) FROM generations WHERE verdict='pass' AND approved=0").fetchone()[0]

        avg_fidelity = self.conn.execute(
            "SELECT AVG(fidelity_score) FROM generations WHERE fidelity_score IS NOT NULL "
            "AND created_at >= datetime('now', '-7 days')"
        ).fetchone()[0] or 0.0

        return {
            "total_clips": total_clips,
            "good_parts": good_parts,
            "victim_contrast": victim,
            "unclassified": total_clips - good_parts - victim,
            "total_ingests": total_ingests,
            "failed_ingests": failed,
            "avg_fidelity_7d": round(avg_fidelity, 4),
            "pending_approvals": pending,
        }

    def log_health_snapshot(self):
        h = self.get_health()
        self.conn.execute(
            "INSERT INTO health_log (total_clips, good_parts, victim_contrast, total_ingests, avg_fidelity_7d, pending_approvals) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (h["total_clips"], h["good_parts"], h["victim_contrast"], h["total_ingests"], h["avg_fidelity_7d"], h["pending_approvals"])
        )
        self.conn.commit()


def _hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    if len(h1) != len(h2):
        return 64
    return bin(int(h1, 16) ^ int(h2, 16)).count('1')
