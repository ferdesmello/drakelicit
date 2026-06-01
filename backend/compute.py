"""
Drake Survey — N distribution computation
==========================================
All numerical work lives here. main.py imports and calls these functions.

Dependencies:  numpy  scipy  (pandas optional, used for data loading)
    pip install numpy scipy pandas
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

# ── Constants (match the JS frontend exactly) ─────────────────────────────
Z90     = 1.2816   # z-score for 10th/90th percentiles
SIG_MIN = 0.02     # minimum sigma in T-space (prevents degenerate distributions)
N_MC    = 20_000   # Monte Carlo samples per computation
N_GRID  = 500      # points in the T-grid for community PDFs


# ── Split-normal helpers ───────────────────────────────────────────────────

def _sigmas(t_mode: float, t_lo: float, t_hi: float) -> tuple[float, float]:
    """Return (σ_L, σ_R) in T-space, clamped to SIG_MIN."""
    sig_L = max(SIG_MIN, (t_mode - t_lo) / Z90)
    sig_R = max(SIG_MIN, (t_hi  - t_mode) / Z90)
    return sig_L, sig_R


def split_normal_pdf(t: np.ndarray,
                     t_mode: float, t_lo: float, t_hi: float) -> np.ndarray:
    """
    Evaluate the split-normal PDF at each point in t.
    Matches the JavaScript splitPDF() function exactly.
    """
    sig_L, sig_R = _sigmas(t_mode, t_lo, t_hi)
    A   = np.sqrt(2 / np.pi) / (sig_L + sig_R)
    sig = np.where(t <= t_mode, sig_L, sig_R)
    z   = (t - t_mode) / sig
    y   = A * np.exp(-0.5 * z ** 2)
    return np.where(np.isfinite(y), y, 0.0)


def sample_split_normal(t_mode: float, t_lo: float, t_hi: float,
                        n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n samples from the split-normal in T-space using the
    exact analytic formula (no numerical CDF inversion needed).
    """
    sig_L, sig_R = _sigmas(t_mode, t_lo, t_hi)
    p_left = sig_L / (sig_L + sig_R)
    u      = rng.uniform(size=n)
    left   = u < p_left
    # half-normal draws on each side
    half_L = np.abs(rng.normal(scale=sig_L, size=n))
    half_R = np.abs(rng.normal(scale=sig_R, size=n))
    return np.where(left, t_mode - half_L, t_mode + half_R)


def sample_community_pdf(t_grid: np.ndarray, pdf: np.ndarray,
                         n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n samples from an arbitrary discrete PDF via inverse-CDF.
    pdf must already be normalized (integral ≈ 1).
    """
    dt  = t_grid[1] - t_grid[0]
    cdf = np.cumsum(pdf) * dt
    cdf = cdf / cdf[-1]               # ensure it reaches exactly 1
    u   = rng.uniform(size=n)
    return np.interp(u, cdf, t_grid)


def t_to_value(t: np.ndarray, axis_mode: str) -> np.ndarray:
    """Convert T-space samples to parameter value space."""
    if axis_mode == "log":
        return np.power(10.0, t)
    return t                           # linear: identity


# ── Individual user N distribution ────────────────────────────────────────

def compute_individual_N(answers: list[dict],
                          questions: list[dict],
                          n_mc: int = N_MC,
                          seed: Optional[int] = None) -> dict:
    """
    Monte Carlo N distribution for a single user's set of answers.

    answers   : list of answer dicts (from /api/results/{session_id})
    questions : list of question dicts (from questions.json)
    returns   : dict with smoothed pdf grid and summary statistics,
                or {"error": reason} if any parameter is skipped.
    """
    rng = np.random.default_rng(seed)

    # Index answers by question_id
    ans_by_id = {a["question_id"]: a for a in answers}

    log_N_samples = np.zeros(n_mc)

    for Q in questions:
        qid = Q["id"]
        a   = ans_by_id.get(qid)

        if a is None or a.get("skipped"):
            return {"error": f"Question '{qid}' was skipped — N cannot be computed."}

        axis_mode = a.get("axis_mode", Q.get("axisMode", "log"))
        t_mode, t_lo, t_hi = a["t_mode"], a["t_lo"], a["t_hi"]
        t_min, t_max = Q["tMin"], Q["tMax"]

        t_samples = sample_split_normal(t_mode, t_lo, t_hi, n_mc, rng)
        t_samples = np.clip(t_samples, t_min, t_max)
        v_samples = t_to_value(t_samples, axis_mode)

        # accumulate in log space to avoid overflow
        if axis_mode == "log":
            log_N_samples += t_samples          # log10(v) = t
        else:
            # for linear parameters, add log10 of the value
            v_samples = np.maximum(v_samples, 1e-300)
            log_N_samples += np.log10(v_samples)

    return _summarise_log_N(log_N_samples)


# ── Community PDFs ────────────────────────────────────────────────────────

def compute_community_pdfs(all_answers: list[dict],
                            questions: list[dict]) -> dict[str, dict]:
    """
    For each question, compute the linear opinion pool of all users' PDFs.
    Returns a dict keyed by question_id with keys:
        t_grid      : list of N_GRID t-values
        community_pdf : list of normalised PDF values
        n_answers   : number of non-skipped answers included
    """
    result = {}

    for Q in questions:
        qid   = Q["id"]
        t_min = Q["tMin"]
        t_max = Q["tMax"]
        t_grid = np.linspace(t_min, t_max, N_GRID)

        pooled   = np.zeros(N_GRID)
        n_valid  = 0

        for a in all_answers:
            if a["question_id"] != qid or a.get("skipped"):
                continue
            axis_mode = a.get("axis_mode", Q.get("axisMode", "log"))
            pdf = split_normal_pdf(t_grid, a["t_mode"], a["t_lo"], a["t_hi"])
            pooled  += pdf
            n_valid += 1

        if n_valid == 0:
            result[qid] = {"t_grid": t_grid.tolist(),
                           "community_pdf": np.zeros(N_GRID).tolist(),
                           "n_answers": 0}
            continue

        # Average and normalise
        pooled /= n_valid
        area    = np.trapz(pooled, t_grid)
        if area > 0:
            pooled /= area

        result[qid] = {
            "t_grid":        t_grid.tolist(),
            "community_pdf": pooled.tolist(),
            "n_answers":     n_valid,
        }

    return result


# ── Community N distribution ──────────────────────────────────────────────

def compute_community_N(community_pdfs: dict[str, dict],
                         questions: list[dict],
                         n_mc: int = N_MC,
                         seed: Optional[int] = None) -> dict:
    """
    Monte Carlo N distribution using the community (pooled) PDFs.
    """
    rng = np.random.default_rng(seed)
    log_N_samples = np.zeros(n_mc)

    for Q in questions:
        qid  = Q["id"]
        cpdf = community_pdfs.get(qid)

        if cpdf is None or cpdf["n_answers"] == 0:
            return {"error": f"No community answers for '{qid}'."}

        axis_mode = Q.get("axisMode", "log")
        t_grid    = np.array(cpdf["t_grid"])
        pdf       = np.array(cpdf["community_pdf"])
        t_min, t_max = Q["tMin"], Q["tMax"]

        t_samples = sample_community_pdf(t_grid, pdf, n_mc, rng)
        t_samples = np.clip(t_samples, t_min, t_max)

        if axis_mode == "log":
            log_N_samples += t_samples
        else:
            v = np.maximum(t_samples, 1e-300)
            log_N_samples += np.log10(v)

    return _summarise_log_N(log_N_samples)


# ── Shared summary helper ─────────────────────────────────────────────────

def _summarise_log_N(log_N: np.ndarray) -> dict:
    """
    Given an array of log10(N) samples, return a dict with:
      - a KDE-smoothed PDF evaluated on a regular grid
      - summary statistics (median, IQR, 5th/95th percentiles)
    """
    # Remove any -inf / nan that could appear from log of 0
    log_N = log_N[np.isfinite(log_N)]
    if len(log_N) == 0:
        return {"error": "All N samples were non-finite."}

    # KDE in log10(N) space — Scott's rule bandwidth
    kde  = gaussian_kde(log_N, bw_method="scott")
    lo   = np.percentile(log_N, 0.5)
    hi   = np.percentile(log_N, 99.5)
    # extend grid slightly beyond the data
    grid = np.linspace(lo - 0.5, hi + 0.5, 600)
    pdf  = kde(grid)
    # normalize so trapz = 1
    pdf /= np.trapz(pdf, grid)

    pct = lambda p: float(np.percentile(log_N, p))

    return {
        "log_N_grid": grid.tolist(),
        "pdf":        pdf.tolist(),
        "log10_median": pct(50),
        "log10_q05":    pct(5),
        "log10_q25":    pct(25),
        "log10_q75":    pct(75),
        "log10_q95":    pct(95),
        # convenience: the actual values at those percentiles
        "N_median": float(10 ** pct(50)),
        "N_q05":    float(10 ** pct(5)),
        "N_q25":    float(10 ** pct(25)),
        "N_q75":    float(10 ** pct(75)),
        "N_q95":    float(10 ** pct(95)),
    }


# ── Top-level: compute everything for the API endpoint ───────────────────

def compute_all(data_file: Path, questions_file: Path) -> dict:
    """
    Load data.json + questions.json and return the full computation result:
      - community PDFs for each parameter
      - community N distribution
    This is what GET /api/aggregate/computed calls.
    """
    with data_file.open() as f:
        data = json.load(f)
    with questions_file.open() as f:
        questions = json.load(f)

    completed_ids = {
        sid for sid, s in data["sessions"].items() if s.get("completed")
    }
    all_answers = [
        a for a in data["answers"] if a["session_id"] in completed_ids
    ]

    community_pdfs = compute_community_pdfs(all_answers, questions)
    community_N    = compute_community_N(community_pdfs, questions)

    return {
        "n_sessions":    len(completed_ids),
        "n_answers":     len(all_answers),
        "parameters":    community_pdfs,
        "community_N":   community_N,
    }
