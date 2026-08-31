# Drakelicit

Drakelicit is a small web application for expert elicitation around the [Drake equation](https://en.wikipedia.org/wiki/Drake_equation). Instead of asking for one-point estimates, it asks each respondent to express uncertainty as a distribution for every parameter and then combines those answers into a distribution for the possible number of detectable civilizations in the Milky Way.

The interface is intentionally simple: a static HTML front end guides the user through the survey, and a FastAPI backend stores sessions, answers, and aggregate results.

## What this project does

The survey walks a respondent through the Drake equation parameters one by one, using slider-based distributions in T-space. Each answer records a lower bound, a mode, and an upper bound, along with whether the respondent skipped the question and which axis mode was used.

After the survey is complete, the backend can:

1. Save the respondent’s session and answers.
2. Return a per-user result for the computed N distribution.
3. Aggregate completed sessions into a community result.
4. Export sessions and answers as CSV files.

The numerical work is handled in the backend, where the answer distributions are sampled and combined into a Monte Carlo estimate of N.

## How the survey works

The user flow is:

1. index.html lets the user choose a language.
2. intro.html explains the Drake equation and the survey format.
3. consent.html creates a survey session on the server.
4. survey.html presents one parameter at a time and saves drafts.
5. results.html shows the individual and aggregated results.

The survey currently supports Portuguese, Spanish, and English through the files in i18n/.

## Repository structure

```text
n-elicit/
├── index.html
├── intro.html
├── consent.html
├── survey.html
├── results.html
├── questions.json
├── data/
├── i18n/
│   ├── en.json
│   ├── es.json
│   └── pt.json
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── compute.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

## Running the project locally

The backend serves the static front end as well, so you only need to start the FastAPI app and then open the browser on the local server.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000.

### Database configuration

The backend uses PostgreSQL via asyncpg. If DATABASE_URL is not set, it falls back to the local development connection in backend/database.py.

Set your own connection string if you want to use a different database:

```bash
$env:DATABASE_URL = "postgresql://user:password@localhost/drake_survey"
uvicorn main:app --reload --port 8000
```

The tables are created automatically on startup.

## Backend API

The main endpoints live in backend/main.py:

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/consent | Create a new session |
| POST | /api/answer | Save or update one answer |
| POST | /api/complete | Mark a session as finished |
| GET | /api/results/{session_id} | Fetch raw answers for one session |
| GET | /api/results/{session_id}/computed | Compute the individual N distribution |
| GET | /api/aggregate | Fetch all answers from completed sessions |
| GET | /api/aggregate/computed | Compute the community aggregate |
| GET | /api/health | Return service and database stats |
| GET | /api/export/sessions.csv | Export sessions as CSV |
| GET | /api/export/answers.csv | Export answers as CSV |

## Important files

- [backend/main.py](backend/main.py) defines the FastAPI app and API endpoints.
- [backend/database.py](backend/database.py) contains the PostgreSQL schema and database helpers.
- [backend/compute.py](backend/compute.py) implements the Monte Carlo and probability calculations.
- [questions.json](questions.json) contains the survey questions and parameter bounds.
- [i18n/en.json](i18n/en.json), [i18n/es.json](i18n/es.json), and [i18n/pt.json](i18n/pt.json) hold the translation strings used by the front end.

## Drake equation context

The survey uses this form of the Drake equation:

$$
N = N_{*} \cdot f_{p} \cdot n_{e} \cdot f_{l} \cdot f_{i} \cdot f_{c} \cdot f_{L}
$$

Where:

- $N$ is the number of civilizations in the Milky Way with which communication might be possible.
- $N_{*}$ is the number of stars in the Milky Way.
- $f_{p}$ is the fraction of those stars that have planets.
- $n_{e}$ is the average number of potentially habitable planets per star with planets.
- $f_{l}$ is the fraction of habitable planets that actually develop life.
- $f_{i}$ is the fraction of life-bearing planets that develop intelligent life.
- $f_{c}$ is the fraction of civilizations that develop detectable technology.
- $f_{L}$ is the fraction of a planetary lifetime spent in a detectable technological phase.

The important idea behind this project is that these parameters are uncertain by orders of magnitude, so the app models them as distributions instead of point estimates.

## Related work

This repository follows on from a precursor project that used Metaculus data:

https://github.com/ferdesmello/drake-metaculus

## Notes

- The data/slider_demo.gif image is used in the intro page as a quick visual explanation of the slider controls.
- The front end is static HTML and JavaScript, but it depends on the backend API for persistence and results.
- The backend README contains more detailed deployment notes for the database layer.
