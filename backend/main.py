"""
Drake Survey — FastAPI backend
==============================
Step 2: PostgreSQL via asyncpg (replaces data.json file storage).

Run:
    export DATABASE_URL="postgresql://drake_user:password@localhost/drake_survey"
    uvicorn main:app --reload --port 8000

The DATABASE_URL defaults to localhost with the dev credentials if not set.
Tables are created automatically on startup.

To import existing data from the old data.json:
    python migrate.py
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database as db
from compute import compute_all_from_answers, compute_individual_N

# ── paths ──────────────────────────────────────────────────────────────────
HERE          = Path(__file__).parent
STATIC_DIR    = HERE.parent
QUESTIONS_FILE = STATIC_DIR / "questions.json"


# ── lifespan: open/close the connection pool ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool()
    await db.create_tables()
    yield
    await db.close_pool()

app = FastAPI(title="Drake Survey API", version="0.2.0", lifespan=lifespan)


# ── Pydantic models ────────────────────────────────────────────────────────

class ConsentPayload(BaseModel):
    lang: str = "pt"
    tcle_version: str = "1.0"

class AnswerPayload(BaseModel):
    session_id: str
    question_id: str
    axis_mode: str = "log"
    t_lo: float
    t_mode: float
    t_hi: float
    skipped: bool = False

class SessionResponse(BaseModel):
    session_id: str


# ── Survey API ─────────────────────────────────────────────────────────────

@app.post("/api/consent", response_model=SessionResponse, tags=["survey"])
async def create_session(payload: ConsentPayload):
    """Create a new survey session and return its UUID."""
    session_id = str(uuid.uuid4())
    await db.create_session(session_id, payload.lang, payload.tcle_version)
    return SessionResponse(session_id=session_id)


@app.post("/api/answer", tags=["survey"])
async def save_answer(payload: AnswerPayload):
    """Save or update one answer. Atomic upsert — safe under concurrent load."""
    if not await db.session_exists(payload.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    await db.upsert_answer(
        session_id=payload.session_id,
        question_id=payload.question_id,
        axis_mode=payload.axis_mode,
        t_lo=payload.t_lo,
        t_mode=payload.t_mode,
        t_hi=payload.t_hi,
        skipped=payload.skipped,
    )
    return {"ok": True}


@app.post("/api/complete", tags=["survey"])
async def mark_complete(request: Request):
    """Mark a session as fully completed."""
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    ok = await db.mark_session_complete(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.get("/api/results/{session_id}", tags=["results"])
async def get_results(session_id: str):
    """Return raw answers for one session (used by results page as fallback)."""
    if not await db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    answers = await db.get_answers_for_session(session_id)
    # serialise datetime objects
    for a in answers:
        if hasattr(a.get("answered_at"), "isoformat"):
            a["answered_at"] = a["answered_at"].isoformat()
    return {"session_id": session_id, "answers": answers}


@app.get("/api/results/{session_id}/computed", tags=["results"])
async def get_individual_computed(session_id: str):
    """Compute the N distribution for one session server-side."""
    if not await db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    answers = await db.get_answers_for_session(session_id)
    questions = json.loads(QUESTIONS_FILE.read_text())
    # serialise datetimes before returning
    clean_answers = []
    for a in answers:
        ca = dict(a)
        if hasattr(ca.get("answered_at"), "isoformat"):
            ca["answered_at"] = ca["answered_at"].isoformat()
        clean_answers.append(ca)
    try:
        result = compute_individual_N(clean_answers, questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"session_id": session_id, "N_distribution": result,
            "answers": clean_answers}


@app.get("/api/aggregate", tags=["results"])
async def get_aggregate():
    """Return raw answers from all completed sessions."""
    n_sessions, answers = await db.get_all_completed_answers()
    for a in answers:
        if hasattr(a.get("answered_at"), "isoformat"):
            a["answered_at"] = a["answered_at"].isoformat()
    return {"n_sessions": n_sessions, "answers": answers}


@app.get("/api/aggregate/computed", tags=["results"])
async def get_aggregate_computed():
    """Compute community PDFs and N distribution from all completed sessions."""
    n_sessions, answers = await db.get_all_completed_answers()
    if n_sessions == 0:
        return {"n_sessions": 0, "n_answers": 0,
                "parameters": {}, "community_N": {"error": "No data yet."}}
    questions = json.loads(QUESTIONS_FILE.read_text())
    try:
        result = compute_all_from_answers(answers, questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result["n_sessions"] = n_sessions
    return result


@app.get("/api/health", tags=["meta"])
async def health():
    """Liveness check — also shows pool and data stats."""
    stats = await db.health_stats()
    pool = db.get_pool()
    return {
        "status": "ok",
        **stats,
        "pool_size": pool.get_size(),
        "pool_free": pool.get_idle_size(),
    }


# ── Data export endpoints ─────────────────────────────────────────────────

@app.get("/api/export/sessions.csv", tags=["export"])
async def export_sessions_csv():
    """Download all sessions as a CSV file."""
    import csv, io
    async with db.get_pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, lang, tcle_version, created_at, completed, completed_at "
            "FROM sessions ORDER BY created_at"
        )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id","lang","tcle_version","created_at","completed","completed_at"])
    for r in rows:
        w.writerow([r["id"], r["lang"], r["tcle_version"],
                    r["created_at"].isoformat() if r["created_at"] else "",
                    r["completed"],
                    r["completed_at"].isoformat() if r["completed_at"] else ""])
    return Response(content=buf.getvalue(),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sessions.csv"})


@app.get("/api/export/answers.csv", tags=["export"])
async def export_answers_csv():
    """Download all answers as a CSV file."""
    import csv, io
    async with db.get_pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT session_id, question_id, axis_mode, t_lo, t_mode, t_hi, "
            "skipped, answered_at FROM answers ORDER BY session_id, answered_at"
        )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["session_id","question_id","axis_mode",
                "t_lo","t_mode","t_hi","skipped","answered_at"])
    for r in rows:
        w.writerow([r["session_id"], r["question_id"], r["axis_mode"],
                    r["t_lo"], r["t_mode"], r["t_hi"], r["skipped"],
                    r["answered_at"].isoformat() if r["answered_at"] else ""])
    return Response(content=buf.getvalue(),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=answers.csv"})


# ── Static files (must be last) ────────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
