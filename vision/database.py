"""
vision/database.py — SQLite Database Handler
=============================================
Manages all persistent storage for events and visitor records.

Tables:
    events   — every visitor interaction logged here
    visitors — known face registry (face_index → name)
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.db_path = cfg.paths.db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()
        logger.info(f"Database ready at: {self.db_path}")

    def _connect(self) -> None:
        """Open connection with WAL mode for better concurrent access."""
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL mode — safer for RPi, allows reads during writes
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        """Create tables if they don't exist. Safe to call on every startup."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT,
                timestamp   TEXT    NOT NULL,
                face_index  INTEGER,
                image_path  TEXT,
                speech_text TEXT,
                intent      TEXT,
                spoken_name TEXT,
                conflict    INTEGER DEFAULT 0,
                language    TEXT    DEFAULT 'ur',
                duration_ms INTEGER,
                extra       TEXT
            );

            CREATE TABLE IF NOT EXISTS visitors (
                face_index  INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL,
                first_seen  TEXT,
                last_seen   TEXT,
                visit_count INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp);

            CREATE INDEX IF NOT EXISTS idx_events_name
                ON events(name);
        """)
        self._conn.commit()
        logger.debug("Database tables verified.")

    def insert(
        self,
        name: str,
        face_index: int,
        image_path: str = "",
        speech_text: str = "",
        intent: str = "unknown",
        spoken_name: str = "",
        conflict: int = 0,
        language: str = "ur",
        duration_ms: int = 0,
        extra: str = "",
    ) -> int:
        """
        Insert a new visitor event. Returns the new row ID.

        Args:
            name:        FAISS-matched name or 'Unknown'
            face_index:  FAISS index of the matched face (-1 if unknown)
            image_path:  Path to saved face crop
            speech_text: Raw Whisper transcript
            intent:      Detected intent label
            spoken_name: Name extracted from speech
            conflict:    1 if FAISS name != spoken name
            language:    'ur' or 'ps'
            duration_ms: Total dialogue duration in milliseconds
            extra:       JSON string for plugin-specific data
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self._conn.execute(
            """
            INSERT INTO events
                (name, timestamp, face_index, image_path, speech_text,
                 intent, spoken_name, conflict, language, duration_ms, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, timestamp, face_index, image_path, speech_text,
             intent, spoken_name, conflict, language, duration_ms, extra),
        )
        self._conn.commit()
        logger.debug(f"Event inserted: name={name} intent={intent} id={cursor.lastrowid}")
        return cursor.lastrowid

    def upsert_visitor(self, face_index: int, name: str) -> None:
        """
        Insert or update a visitor record.
        Updates last_seen and increments visit_count on repeat visits.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing = self._conn.execute(
            "SELECT face_index FROM visitors WHERE face_index = ?",
            (face_index,)
        ).fetchone()

        if existing:
            self._conn.execute(
                """
                UPDATE visitors
                SET last_seen = ?, visit_count = visit_count + 1
                WHERE face_index = ?
                """,
                (now, face_index),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO visitors (face_index, name, first_seen, last_seen, visit_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (face_index, name, now, now),
            )
        self._conn.commit()
        logger.debug(f"Visitor upserted: face_index={face_index} name={name}")

    def get_visitor_name(self, face_index: int) -> Optional[str]:
        """Return the stored name for a face_index, or None if not found."""
        row = self._conn.execute(
            "SELECT name FROM visitors WHERE face_index = ?",
            (face_index,)
        ).fetchone()
        return row["name"] if row else None

    def get_all_visitors(self) -> list:
        """Return all enrolled visitors as a list of dicts."""
        rows = self._conn.execute(
            "SELECT * FROM visitors ORDER BY face_index"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_events(self, limit: int = 20) -> list:
        """Return the most recent events as a list of dicts."""
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_visitor(self, face_index: int) -> bool:
        """Remove a visitor record. Returns True if a row was deleted."""
        cursor = self._conn.execute(
            "DELETE FROM visitors WHERE face_index = ?",
            (face_index,)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Visitor deleted: face_index={face_index}")
        return deleted

    def close(self) -> None:
        """Close the database connection cleanly."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the whole application
# ---------------------------------------------------------------------------
db = Database()
