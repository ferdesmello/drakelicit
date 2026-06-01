"""
Drake Survey — FastAPI backend (Step 1)
=======================================
Serves all static files and exposes a minimal REST API.
Answers are stored in a local JSON file for now (no database yet).
Replace the storage functions in step 2 when adding PostgreSQL.

Run:
    pip install fastapi uvicorn
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in your browser.

Directory layout expected (relative to this file):
    ../               ← all HTML pages
    ../i18n/          ← pt.json, es.json, en.json
    ../questions.json ← question definitions
    ./data.json       ← created automatically, stores all answers
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from compute import compute_all, compute_individual_N

# ── paths ──────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent          # .../backend/
STATIC_DIR = HERE.parent                    # .../drake-survey/
DATA_FILE       = HERE / "data.json"        # simple JSON store
QUESTIONS_FILE  = STATIC_DIR / "questions.json"

app = FastAPI(title="Drake Survey API", version="0.1.0")


# ── JSON file storage (replaced by PostgreSQL in step 2) ──────────────────

def _load() -> dict:
    """Load the entire data store from disk."""
    if not DATA_FILE.exists():
        return {"sessions": {}, "answers": []}
    with DATA_FILE.open() as f:
        return json.load(f)

def _save(data: dict) -> None:
    """Persist the data store to disk atomically."""
    tmp = DATA_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(DATA_FILE)


# ── Pydantic models ────────────────────────────────────────────────────────

class ConsentPayload(BaseModel):
    lang: str = "pt"
    tcle_version: str = "1.0"

class AnswerPayload(BaseModel):
    session_id: str
    question_id: str                    # e.g. "fp", "R*"
    axis_mode: str = "log"              # "log" | "lin"
    t_lo: float
    t_mode: float
    t_hi: float
    skipped: bool = False

class SessionResponse(BaseModel):
    session_id: str


# ── API routes ─────────────────────────────────────────────────────────────

@app.post("/api/consent", response_model=SessionResponse, tags=["survey"])
async def create_session(payload: ConsentPayload):
    """
    Called when the user ticks all consent boxes and clicks Start.
    Creates a new session and returns its UUID — the frontend stores
    this in sessionStorage and sends it with every subsequent request.
    """
    session_id = str(uuid.uuid4())
    data = _load()
    data["sessions"][session_id] = {
        "lang": payload.lang,
        "tcle_version": payload.tcle_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": False,
    }
    _save(data)
    return SessionResponse(session_id=session_id)


@app.post("/api/answer", tags=["survey"])
async def save_answer(payload: AnswerPayload):
    """
    Called after each question (on Next or Skip).
    Upserts the answer — if the same session+question already exists,
    the previous answer is replaced (allows returning users to revise).
    """
    data = _load()

    if payload.session_id not in data["sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove any existing answer for this session+question (upsert)
    data["answers"] = [
        a for a in data["answers"]
        if not (a["session_id"] == payload.session_id
                and a["question_id"] == payload.question_id)
    ]

    data["answers"].append({
        "session_id":  payload.session_id,
        "question_id": payload.question_id,
        "axis_mode":   payload.axis_mode,
        "t_lo":        payload.t_lo,
        "t_mode":      payload.t_mode,
        "t_hi":        payload.t_hi,
        "skipped":     payload.skipped,
        "answered_at": datetime.now(timezone.utc).isoformat(),
    })

    _save(data)
    return {"ok": True}


@app.post("/api/complete", tags=["survey"])
async def mark_complete(request: Request):
    """Mark a session as fully completed."""
    body = await request.json()
    session_id = body.get("session_id")
    data = _load()
    if session_id not in data["sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    data["sessions"][session_id]["completed"] = True
    data["sessions"][session_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    _save(data)
    return {"ok": True}


@app.get("/api/results/{session_id}", tags=["results"])
async def get_results(session_id: str):
    """
    Returns the answers for one session so the results page
    can compute the N distribution client-side.
    (In step 4 this will do the Monte Carlo server-side instead.)
    """
    data = _load()
    if session_id not in data["sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    answers = [a for a in data["answers"] if a["session_id"] == session_id]
    return {"session_id": session_id, "answers": answers}


@app.get("/api/aggregate", tags=["results"])
async def get_aggregate():
    """
    Returns all completed answers across all sessions.
    Used (eventually) to show the community distribution on the results page.
    Only includes sessions marked as completed.
    """
    data = _load()
    completed_ids = {
        sid for sid, s in data["sessions"].items() if s.get("completed")
    }
    answers = [a for a in data["answers"] if a["session_id"] in completed_ids]
    return {
        "n_sessions": len(completed_ids),
        "answers": answers,
    }


@app.get("/api/health", tags=["meta"])
async def health():
    """Quick liveness check."""
    data = _load()
    return {
        "status": "ok",
        "sessions": len(data.get("sessions", {})),
        "answers":  len(data.get("answers",  [])),
    }


# ── Computation endpoints ─────────────────────────────────────────────────

@app.get("/api/aggregate/computed", tags=["results"])
async def get_aggregate_computed():
    """
    Compute and return community PDFs for every parameter plus
    the community N distribution.  Only completed sessions are included.
    Results are computed fresh on each request (add caching later if needed).
    """
    if not DATA_FILE.exists():
        return {"n_sessions": 0, "n_answers": 0,
                "parameters": {}, "community_N": {"error": "No data yet."}}
    try:
        result = compute_all(DATA_FILE, QUESTIONS_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/api/results/{session_id}/computed", tags=["results"])
async def get_individual_computed(session_id: str):
    """
    Compute and return the N distribution for a single session.
    Uses the answers stored server-side, not the client localStorage.
    """
    data = _load()
    if session_id not in data["sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")

    answers   = [a for a in data["answers"] if a["session_id"] == session_id]
    questions = json.load(QUESTIONS_FILE.open())

    try:
        result = compute_individual_N(answers, questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    answers = [a for a in data["answers"] if a["session_id"] == session_id]
    return {"session_id": session_id, "N_distribution": result, "answers": answers}


# ── Static file serving ────────────────────────────────────────────────────
# All HTML pages, i18n JSON files, and questions.json are served from
# the parent directory (drake-survey/).
# This mount must come LAST so API routes take priority.

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
