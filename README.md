# Drake Survey — Backend (Step 1)

## Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

## Directory layout

```
drake-survey/
├── index.html
├── intro.html
├── consent.html
├── survey_question_page.html
├── results.html
├── questions.json
├── i18n/
│   ├── pt.json
│   ├── es.json
│   └── en.json
└── backend/
    ├── main.py          ← FastAPI app (this file)
    ├── requirements.txt
    ├── README.md
    └── data.json        ← created automatically on first answer
```

**Important:** the i18n JSON files must be inside an `i18n/` subfolder,
and `questions.json` must be at the root of `drake-survey/`.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/consent` | Create session, returns `session_id` |
| POST | `/api/answer` | Save one answer (upsert) |
| POST | `/api/complete` | Mark session as finished |
| GET  | `/api/results/{session_id}` | Fetch answers for one session |
| GET  | `/api/aggregate` | All completed answers (community pool) |
| GET  | `/api/health` | Liveness check |

## Frontend integration (what to add next)

### consent.html — after all boxes checked, call /api/consent:
```javascript
const res = await fetch('/api/consent', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ lang, tcle_version: '1.0' })
});
const { session_id } = await res.json();
sessionStorage.setItem('drake_session', session_id);
```

### survey_question_page.html — replace saveDraft() body:
```javascript
async function saveDraft() {
  const session_id = sessionStorage.getItem('drake_session');
  if (!session_id) return;                // no session yet
  await fetch('/api/answer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id,
      question_id: Q.id,
      axis_mode:   Q.axisMode,
      t_lo:  tLo,
      t_mode: tMode,
      t_hi:  tHi,
      skipped,
    })
  });
}
```

### navigate() — mark complete on last question:
```javascript
if (nextQ >= window._questions.length) {
  const session_id = sessionStorage.getItem('drake_session');
  await fetch('/api/complete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ session_id })
  });
  location.href = 'results.html?lang=' + lang + '&session=' + session_id;
}
```

### results.html — load answers from server instead of localStorage:
```javascript
const session_id = new URLSearchParams(location.search).get('session');
const { answers } = await fetch(`/api/results/${session_id}`).then(r => r.json());
// use answers[] instead of localStorage reads
```

## Step 2 — Adding PostgreSQL

When ready, install `asyncpg` and `sqlalchemy[asyncio]`, create the three
tables (sessions, answers, consents) from the schema in the project notes,
and replace the `_load()` / `_save()` calls in `main.py` with async
database queries. The API surface stays identical so no frontend changes
are needed.
