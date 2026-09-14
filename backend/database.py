"""
Drake Survey — PostgreSQL database layer
========================================
All SQL lives here. main.py imports these functions and never touches SQL directly.

Connection string format:
    postgresql://user:password@host:port/dbname

Set the environment variable DATABASE_URL before starting the server, e.g.:
    export DATABASE_URL="postgresql://drake_user:password@localhost/drake_survey"
    uvicorn main:app --reload --port 8000

Tables created automatically on first startup via create_tables().
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg

# ── module-level pool (initialised in lifespan) ───────────────────────────
_pool: Optional[asyncpg.Pool] = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised. Call init_pool() first.")
    return _pool


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register a codec so jsonb columns round-trip as Python dicts,
    instead of raw JSON strings, on every connection in the pool."""
    await conn.set_type_codec(
        'jsonb',
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog',
    )


async def init_pool() -> None:
    """Create the connection pool. Called once at application startup."""
    global _pool
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://drake_user:drake_pass@localhost/drake_survey",
    )
    _pool = await asyncpg.create_pool(
        url,
        min_size=2,
        max_size=10,
        command_timeout=30,
        init=_init_connection,   # ← new
    )


async def close_pool() -> None:
    """Gracefully close the pool at shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── schema ────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT        PRIMARY KEY,          -- UUID as text
    lang         TEXT        NOT NULL DEFAULT 'pt',
    tcle_version TEXT        NOT NULL DEFAULT '1.0',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed    BOOLEAN     NOT NULL DEFAULT false,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS answers (
    id           BIGSERIAL   PRIMARY KEY,
    session_id   TEXT        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question_id  TEXT        NOT NULL,
    axis_mode    TEXT        NOT NULL DEFAULT 'log',
    t_lo         FLOAT       NOT NULL,
    t_mode       FLOAT       NOT NULL,
    t_hi         FLOAT       NOT NULL,
    skipped      BOOLEAN     NOT NULL DEFAULT false,
    answered_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- one row per session+question (upsert key)
    UNIQUE (session_id, question_id)
);

CREATE TABLE IF NOT EXISTS context_answers (
    session_id      TEXT        PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    area_grande     TEXT,
    area_especifica TEXT,
    beliefs         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    answered_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- index for the aggregate query (fetch all answers for completed sessions)
CREATE INDEX IF NOT EXISTS idx_answers_session ON answers (session_id);
"""


async def create_tables() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    async with get_pool().acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)


# ── sessions ──────────────────────────────────────────────────────────────

async def create_session(session_id: str, lang: str, tcle_version: str) -> None:
    """Insert a new session row."""
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (id, lang, tcle_version)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO NOTHING
            """,
            session_id, lang, tcle_version,
        )


async def mark_session_complete(session_id: str) -> bool:
    """Mark a session as completed. Returns True if the session existed."""
    async with get_pool().acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET completed = true, completed_at = now()
            WHERE id = $1
            """,
            session_id,
        )
    return result == "UPDATE 1"


async def session_exists(session_id: str) -> bool:
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM sessions WHERE id = $1", session_id
        )
    return row is not None


# ── answers ───────────────────────────────────────────────────────────────

async def upsert_answer(
    session_id: str,
    question_id: str,
    axis_mode: str,
    t_lo: float,
    t_mode: float,
    t_hi: float,
    skipped: bool,
) -> None:
    """
    Insert or replace an answer for a given session+question.
    Uses PostgreSQL's atomic ON CONFLICT DO UPDATE — no race conditions.
    """
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO answers
                (session_id, question_id, axis_mode, t_lo, t_mode, t_hi, skipped, answered_at)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, now())
            ON CONFLICT (session_id, question_id) DO UPDATE SET
                axis_mode   = EXCLUDED.axis_mode,
                t_lo        = EXCLUDED.t_lo,
                t_mode      = EXCLUDED.t_mode,
                t_hi        = EXCLUDED.t_hi,
                skipped     = EXCLUDED.skipped,
                answered_at = now()
            """,
            session_id, question_id, axis_mode,
            t_lo, t_mode, t_hi, skipped,
        )


async def get_answers_for_session(session_id: str) -> list[dict]:
    """Return all answers for one session as a list of dicts."""
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT question_id, axis_mode, t_lo, t_mode, t_hi, skipped, answered_at
            FROM answers
            WHERE session_id = $1
            ORDER BY answered_at
            """,
            session_id,
        )
    return [dict(r) for r in rows]


async def get_all_completed_answers() -> tuple[int, list[dict]]:
    """
    Return (n_completed_sessions, answers) for all completed sessions.
    Used by the aggregate/community computation.
    """
    async with get_pool().acquire() as conn:
        n_sessions = await conn.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE completed = true"
        )
        rows = await conn.fetch(
            """
            SELECT a.session_id, a.question_id, a.axis_mode,
                   a.t_lo, a.t_mode, a.t_hi, a.skipped
            FROM answers a
            JOIN sessions s ON s.id = a.session_id
            WHERE s.completed = true
            ORDER BY a.session_id, a.answered_at
            """
        )
    return int(n_sessions), [dict(r) for r in rows]


# ── context (background + prior beliefs) ───────────────────────────────────

async def upsert_context(
    session_id: str,
    area_grande: Optional[str],
    area_especifica: Optional[str],
    beliefs: dict,
) -> None:
    """Insert or replace the background/beliefs page for one session."""
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO context_answers
                (session_id, area_grande, area_especifica, beliefs, answered_at)
            VALUES
                ($1, $2, $3, $4, now())
            ON CONFLICT (session_id) DO UPDATE SET
                area_grande     = EXCLUDED.area_grande,
                area_especifica = EXCLUDED.area_especifica,
                beliefs         = EXCLUDED.beliefs,
                answered_at     = now()
            """,
            session_id, area_grande, area_especifica, beliefs,
        )


async def get_all_completed_contexts() -> list[dict]:
    """Background/belief answers for all completed sessions — used only to
    characterize the sample. area_especifica is included here (raw), but
    should never be surfaced in an aggregate endpoint — see main.py."""
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.session_id, c.area_grande, c.area_especifica, c.beliefs
            FROM context_answers c
            JOIN sessions s ON s.id = c.session_id
            WHERE s.completed = true
            """
        )
    return [dict(r) for r in rows]


# ── health ────────────────────────────────────────────────────────────────

async def health_stats() -> dict:
    async with get_pool().acquire() as conn:
        n_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        n_completed = await conn.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE completed = true"
        )
        n_answers = await conn.fetchval("SELECT COUNT(*) FROM answers")
    return {
        "sessions":  int(n_sessions),
        "completed": int(n_completed),
        "answers":   int(n_answers),
    }


# ── migration: import existing data.json ─────────────────────────────────

async def import_from_json(data: dict) -> tuple[int, int]:
    """
    One-time import of data from the old JSON file into PostgreSQL.
    Returns (sessions_imported, answers_imported).
    Call via:
        python -c "import asyncio, json, database as db
        asyncio.run(db.init_pool())
        data = json.load(open('data.json'))
        asyncio.run(db.import_from_json(data))"
    """
    n_sessions = n_answers = 0
    async with get_pool().acquire() as conn:
        async with conn.transaction():
            for sid, s in data.get("sessions", {}).items():
                await conn.execute(
                    """
                    INSERT INTO sessions (id, lang, tcle_version, created_at, completed, completed_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    sid,
                    s.get("lang", "pt"),
                    s.get("tcle_version", "1.0"),
                    datetime.fromisoformat(s["created_at"]) if "created_at" in s
                    else datetime.now(timezone.utc),
                    s.get("completed", False),
                    datetime.fromisoformat(s["completed_at"])
                    if s.get("completed_at") else None,
                )
                n_sessions += 1

            for a in data.get("answers", []):
                await conn.execute(
                    """
                    INSERT INTO answers
                        (session_id, question_id, axis_mode, t_lo, t_mode, t_hi, skipped, answered_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (session_id, question_id) DO UPDATE SET
                        t_lo = EXCLUDED.t_lo, t_mode = EXCLUDED.t_mode,
                        t_hi = EXCLUDED.t_hi, skipped = EXCLUDED.skipped
                    """,
                    a["session_id"],
                    a["question_id"],
                    a.get("axis_mode", "log"),
                    a["t_lo"], a["t_mode"], a["t_hi"],
                    a.get("skipped", False),
                    datetime.fromisoformat(a["answered_at"])
                    if "answered_at" in a else datetime.now(timezone.utc),
                )
                n_answers += 1

    return n_sessions, n_answers
