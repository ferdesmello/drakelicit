# Drakelicit

Drakelicit is a small web application for expert elicitation around the [Drake equation](https://en.wikipedia.org/wiki/Drake_equation). Instead of asking for one-point estimates, it asks each respondent to express uncertainty as a distribution for every parameter and then combines those answers into a distribution for the possible number of detectable civilizations in the Milky Way galaxy.

The interface is intentionally simple: a static HTML front end guides the user through the survey, and a FastAPI backend stores completed sessions, answers, and aggregate results.

## What this project does

The survey walks a respondent through the Drake equation parameters one by one, using slider-based distributions in T-space. Each answer records a lower bound, a mode, and an upper bound, along with whether the respondent skipped the question and which axis mode was used. Draft answers remain in the respondent's browser until the survey is submitted.

After the survey is complete, the backend can:

1. Save the respondent’s session, answers, and optional context in one transaction.
2. Return a per-user result for the computed $N$ distribution.
3. Aggregate completed sessions into a community result.
4. Provide aggregate context and belief counts without exposing specific-area responses.
5. Export sessions, context, and answers as CSV files through admin-only endpoints.
6. Generate a shareable image of a respondent's results, including the personal and available community distributions.

The numerical work is handled in the backend, where the answer distributions are sampled and combined into a Monte Carlo estimate of $N$. The results page reports the median, IQR, 90% interval, and probabilities relative to ($N=1$) and ($N=5 \times 10^{-13}$). If a respondent skips a parameter, their personal N distribution is not calculated, but the community distribution can still be shown when enough completed sessions exist.

## How the survey works

The user flow is:

1. index.html lets the user choose a language.
2. intro.html explains the Drake equation and the survey format.
3. consent.html records consent and the selected language in the browser.
4. context.html presents optional background and beliefs questions.
5. survey.html presents one parameter at a time, saves drafts locally, and displays localized plain-language guidance at the ends of each parameter scale.
6. The final survey submission sends the complete response to the backend once.
7. results.html shows the individual and aggregated results, including the community curve once at least three completed sessions are available.
8. The results page can share the results image through the device's native share sheet or download it as a PNG when native file sharing is unavailable.

The survey currently supports Portuguese, Spanish, and English through the files in `i18n/`.

The results page can be bookmarked after submission. Opening it later with the
session link allows the community distribution to be refreshed as more
responses are collected.

## Repository structure

```text
drakelicit/
├── index.html
├── intro.html
├── consent.html
├── context.html
├── survey.html
├── results.html
├── questions.json
├── data/
│   └── slider_demo.gif
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

The backend uses PostgreSQL via asyncpg. If `DATABASE_URL` is not set, it falls back to the local development connection configured in `backend/database.py`. Tables are created automatically on startup.

Set your own connection string if you want to use a different database:

```powershell
$env:DATABASE_URL = "postgresql://user:password@localhost/drake_survey"
uvicorn main:app --reload --port 8000
```

For local development or deployments, set `ADMIN_TOKEN`. Raw-data, statistics, and CSV export endpoints are disabled when it is not set:

```powershell
$env:ADMIN_TOKEN = "use-a-long-random-value"
```

You can also place `DATABASE_URL` and `ADMIN_TOKEN` in a `.env` file in the
`backend/` directory:

```dotenv
DATABASE_URL=postgresql://drake_user:your_password@localhost/drake_survey
ADMIN_TOKEN=use-a-long-random-value
```

Start the app with the environment file:

```powershell
uvicorn main:app --reload --port 8000 --env-file .env
```

Do not commit `.env`, tokens, or exported CSV files. They are excluded by
`.gitignore`. Generate a strong admin token with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

For a production deployment, set `STATIC_DIR` to a directory containing only
the public website assets (`html`, `json`, `i18n/`, and `data/`). The backend
blocks requests for source files and other sensitive file types from the
static mount.

## Backend API

The main endpoints live in [backend/main.py](backend/main.py):

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/submit | Validate and atomically save a completed survey |
| GET | /api/results/{session_id} | Fetch raw answers for one session |
| GET | /api/results/{session_id}/computed | Compute the individual N distribution |
| GET | /api/aggregate/computed | Compute the community aggregate |
| GET | /api/aggregate/context | Aggregate background and belief counts |
| GET | /api/health | Return the service liveness status |
| GET | /api/stats | Return database statistics (admin token required) |
| GET | /api/aggregate | Fetch raw completed answers (admin token required) |
| GET | /api/export/sessions.csv | Export sessions as CSV |
| GET | /api/export/context.csv | Export context as CSV |
| GET | /api/export/answers.csv | Export answers as CSV |

Send the admin token in the `X-Admin-Token` header when accessing protected endpoints.

## Important files

- [backend/main.py](backend/main.py) defines the FastAPI app and API endpoints.
- [backend/database.py](backend/database.py) contains the PostgreSQL schema, connection-pool setup, and database helpers.
- [backend/compute.py](backend/compute.py) implements the Monte Carlo and probability calculations.
- [questions.json](questions.json) contains the survey questions and parameter bounds.
- [context.html](context.html) contains the optional background and beliefs page.
- [results.html](results.html) renders the personal and community distributions and creates the share/download image.
- [i18n/en.json](i18n/en.json), [i18n/es.json](i18n/es.json), and [i18n/pt.json](i18n/pt.json) hold the translation strings used by the front end.
- [backend/requirements.txt](backend/requirements.txt) lists the backend dependencies, including `python-dotenv` for `.env` configuration.

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

The important idea behind this project is that these parameters are uncertain by orders of magnitude, so the app models them as distributions instead of point estimates. Each question requires a probability distribution:

![Answering a demo question](./data/slider_demo.gif "Slider demo")

## Related work

This repository follows on from a precursor project that used Metaculus data:

https://github.com/ferdesmello/drake-metaculus

## Notes

- The data/slider_demo.gif image is used in the intro page as a quick visual explanation of the slider controls.
- The front end is static HTML and JavaScript. Drafts are stored locally in the browser, while the backend is used for the final submission, persistence, and results.
- Results sharing is client-side: the page renders a PNG summary in the browser and either invokes native file sharing or downloads the image.
- The [backend README](./backend/README.md) contains additional PostgreSQL setup and deployment notes.
