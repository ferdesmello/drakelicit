"""
Drake Survey — N distribution computation
==========================================
All numerical work lives here. main.py imports and calls these functions.

Dependencies:  numpy  scipy
    pip install numpy scipy

Truncated split-normal distributions
─────────────────────────────────────
The survey uses a split-normal distribution in T-space (log₁₀ or linear),
defined by three parameters: t_mode (peak), t_lo (10th percentile), t_hi
(90th percentile). The axis has hard boundaries [t_min, t_max] — no
probability mass is assigned outside those bounds.

Previously the sampler used np.clip(), which piles excess mass at the
boundary points (a spike, not a truncation). This has been replaced with
true truncated sampling via the inverse-CDF method:

    For each half (left: [t_min, t_mode], right: [t_mode, t_max]):
        a = (boundary - t_mode) / sigma       # standardised boundary
        u ~ Uniform(Φ(a_left), Φ(0))          # left half
        u ~ Uniform(Φ(0),      Φ(a_right))    # right half
        t = t_mode + sigma * Φ⁻¹(u)

    where Φ is the standard normal CDF (scipy.special.ndtr) and
    Φ⁻¹ is its inverse (scipy.special.ndtri).

    Every draw is guaranteed to lie within [t_min, t_max]. No rejection
    needed. The mixing weight between left and right is proportional to
    the visible mass under each half-Gaussian.

The split_normal_pdf() function similarly uses a truncation-aware
normalization constant A that integrates only over [t_min, t_max],
keeping the PDF and the sampler strictly consistent with each other.

This also matches the JavaScript frontend, which was updated to use the
same truncated normalization via a polynomial approximation of Φ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import ndtr, ndtri   # Φ and Φ⁻¹ — fast, vectorised
from scipy.stats import gaussian_kde

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


def _trunc_norm_const(t_mode: float, sig_L: float, sig_R: float,
                      t_min: float, t_max: float) -> float:
    """
    Truncation-aware normalization constant A for the split-normal PDF.

    For an untruncated split-normal:
        A = sqrt(2/π) / (σ_L + σ_R)

    With hard boundaries [t_min, t_max], the visible mass is:
        mass_L = Φ(0) - Φ((t_min - t_mode) / σ_L)   [left half-Gaussian]
        mass_R = Φ((t_max - t_mode) / σ_R) - Φ(0)   [right half-Gaussian]

    Each half-Gaussian has un-normalized integral = σ · √(π/2), so the
    total visible integral (before normalization) is:
        I = σ_L · √(π/2) · 2·mass_L  +  σ_R · √(π/2) · 2·mass_R

    And A = 1 / I.

    When t_mode is far from both boundaries, mass_L → 0.5 and mass_R → 0.5,
    which recovers the standard formula A = sqrt(2/π) / (σ_L + σ_R).
    """
    mass_L = float(ndtr(0) - ndtr((t_min - t_mode) / sig_L))
    mass_R = float(ndtr((t_max - t_mode) / sig_R) - ndtr(0))
    integral = sig_L * np.sqrt(2 * np.pi) * mass_L + sig_R * np.sqrt(2 * np.pi) * mass_R
    if integral <= 0:
        return 1.0   # degenerate guard
    return 1.0 / integral


def split_normal_pdf(t: np.ndarray,
                     t_mode: float, t_lo: float, t_hi: float,
                     t_min: float = -np.inf, t_max: float = np.inf) -> np.ndarray:
    """
    Evaluate the truncated split-normal PDF at each point in t.

    The PDF is zero outside [t_min, t_max] and renormalized so that
    its integral over [t_min, t_max] equals 1.

    When t_min=-inf and t_max=inf (defaults), this reduces to the standard
    split-normal — backward compatible with community PDF pooling where the
    grid already spans [t_min, t_max] exactly and boundary effects are tiny.
    """
    sig_L, sig_R = _sigmas(t_mode, t_lo, t_hi)
    A = _trunc_norm_const(t_mode, sig_L, sig_R, t_min, t_max)
    sig = np.where(t <= t_mode, sig_L, sig_R)
    z   = (t - t_mode) / sig
    y   = A * np.exp(-0.5 * z ** 2)
    # zero out mass beyond boundaries
    y   = np.where((t >= t_min) & (t <= t_max), y, 0.0)
    return np.where(np.isfinite(y), y, 0.0)


def sample_split_normal_truncated(t_mode: float, t_lo: float, t_hi: float,
                                   t_min: float, t_max: float,
                                   n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n samples from the truncated split-normal on [t_min, t_max].

    Uses the inverse-CDF (quantile) method — no rejection sampling needed,
    so every draw is guaranteed to land within [t_min, t_max].

    Method (per half):
        Left half  (t in [t_min, t_mode], σ = σ_L):
            Standardised boundary: a_L = (t_min - t_mode) / σ_L  (negative)
            Visible CDF range: [Φ(a_L), Φ(0)] = [Φ(a_L), 0.5]
            Draw u ~ Uniform(Φ(a_L), 0.5), then t = t_mode + σ_L · Φ⁻¹(u)

        Right half (t in [t_mode, t_max], σ = σ_R):
            Standardised boundary: a_R = (t_max - t_mode) / σ_R  (positive)
            Visible CDF range: [Φ(0), Φ(a_R)] = [0.5, Φ(a_R)]
            Draw u ~ Uniform(0.5, Φ(a_R)), then t = t_mode + σ_R · Φ⁻¹(u)

    Mixing probability:
        The weight assigned to the left half is proportional to the visible
        mass under the left half-Gaussian, and vice versa for the right.
        This replaces the old σ_L/(σ_L+σ_R) weight, which assumed untruncated
        tails and would be wrong when t_mode is near a boundary.
    """
    sig_L, sig_R = _sigmas(t_mode, t_lo, t_hi)

    # Standardised boundaries for each half
    a_L = (t_min - t_mode) / sig_L   # ≤ 0
    a_R = (t_max - t_mode) / sig_R   # ≥ 0

    # Visible CDF mass on each side
    phi_aL  = float(ndtr(a_L))       # Φ(a_L), close to 0 when t_min is far left
    phi_aR  = float(ndtr(a_R))       # Φ(a_R), close to 1 when t_max is far right
    mass_L  = 0.5 - phi_aL           # Φ(0) - Φ(a_L)
    mass_R  = phi_aR - 0.5           # Φ(a_R) - Φ(0)
    total   = mass_L + mass_R
    if total <= 0:
        return np.full(n, t_mode)    # fully degenerate: return the mode
    p_left  = mass_L / total

    # Assign each sample to left or right half
    left = rng.uniform(size=n) < p_left

    # Left half: u ~ Uniform(Φ(a_L), 0.5)
    u_L = rng.uniform(phi_aL, 0.5, size=n)
    t_L = t_mode + sig_L * ndtri(u_L)

    # Right half: u ~ Uniform(0.5, Φ(a_R))
    u_R = rng.uniform(0.5, phi_aR, size=n)
    t_R = t_mode + sig_R * ndtri(u_R)

    samples = np.where(left, t_L, t_R)

    # Numerical safety: clip any floating-point overshoot (should be < 1 ULP)
    return np.clip(samples, t_min, t_max)


def sample_community_pdf(t_grid: np.ndarray, pdf: np.ndarray,
                         n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n samples from an arbitrary discrete PDF via inverse-CDF.

    The community PDF is already defined on [t_min, t_max] (it is zero
    outside that grid by construction), so no additional truncation is
    needed here. The np.interp call is monotone and will never produce
    values outside the grid range.
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

    Samples are drawn from the truncated split-normal on [t_min, t_max]
    using sample_split_normal_truncated(). No clipping after the fact.
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

        # Truncated sampler: all draws are within [t_min, t_max], no clipping needed
        t_samples = sample_split_normal_truncated(
            t_mode, t_lo, t_hi, t_min, t_max, n_mc, rng
        )

        # Accumulate in log space to avoid overflow
        if axis_mode == "log":
            log_N_samples += t_samples        # log10(v) = t for log-axis parameters
        else:
            v = np.maximum(t_samples, 1e-300)
            log_N_samples += np.log10(v)

    return _summarise_log_N(log_N_samples)


# ── Community PDFs ────────────────────────────────────────────────────────

def compute_community_pdfs(all_answers: list[dict],
                            questions: list[dict]) -> dict[str, dict]:
    """
    For each question, compute the linear opinion pool of all users' PDFs.

    Each individual PDF is evaluated on the question's [t_min, t_max] grid
    using the truncated split-normal normalization, so that the pooled PDF
    also integrates to 1 over [t_min, t_max] (no mass outside the range).

    Returns a dict keyed by question_id with keys:
        t_grid        : list of N_GRID t-values spanning [t_min, t_max]
        community_pdf : list of normalised PDF values
        n_answers     : number of non-skipped answers included
    """
    result = {}

    for Q in questions:
        qid   = Q["id"]
        t_min = Q["tMin"]
        t_max = Q["tMax"]
        t_grid = np.linspace(t_min, t_max, N_GRID)

        pooled  = np.zeros(N_GRID)
        n_valid = 0

        for a in all_answers:
            if a["question_id"] != qid or a.get("skipped"):
                continue
            # Pass t_min/t_max so each individual PDF is truncation-normalized
            pdf = split_normal_pdf(
                t_grid, a["t_mode"], a["t_lo"], a["t_hi"],
                t_min=t_min, t_max=t_max
            )
            pooled  += pdf
            n_valid += 1

        if n_valid == 0:
            result[qid] = {"t_grid": t_grid.tolist(),
                           "community_pdf": np.zeros(N_GRID).tolist(),
                           "n_answers": 0}
            continue

        # Average and renormalize (linear opinion pool)
        pooled /= n_valid
        area    = np.trapezoid(pooled, t_grid)
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

    The community PDF is already defined and normalised on [t_min, t_max],
    so sample_community_pdf() produces samples strictly within that range.
    No clipping needed.
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

        # Inverse-CDF sampling from the discrete community PDF
        # (already zero outside [t_min, t_max] by construction)
        t_samples = sample_community_pdf(t_grid, pdf, n_mc, rng)

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
    log_N = log_N[np.isfinite(log_N)]
    if len(log_N) == 0:
        return {"error": "All N samples were non-finite."}

    kde  = gaussian_kde(log_N, bw_method="scott")
    lo   = np.percentile(log_N, 0.5)
    hi   = np.percentile(log_N, 99.5)
    grid = np.linspace(lo - 0.5, hi + 0.5, 600)
    pdf  = kde(grid)
    pdf /= np.trapezoid(pdf, grid)

    pct = lambda p: float(np.percentile(log_N, p))

    return {
        "log_N_grid":   grid.tolist(),
        "pdf":          pdf.tolist(),
        "log10_median": pct(50),
        "log10_q05":    pct(5),
        "log10_q25":    pct(25),
        "log10_q75":    pct(75),
        "log10_q95":    pct(95),
        "N_median": float(10 ** pct(50)),
        "N_q05":    float(10 ** pct(5)),
        "N_q25":    float(10 ** pct(25)),
        "N_q75":    float(10 ** pct(75)),
        "N_q95":    float(10 ** pct(95)),
    }


# ── Top-level: compute everything for the API endpoint ───────────────────

def compute_all_from_answers(answers: list[dict], questions: list[dict]) -> dict:
    """
    Compute community PDFs and N distribution from a pre-loaded list of answers.
    Called by main.py after fetching from the database.
    """
    community_pdfs = compute_community_pdfs(answers, questions)
    community_N    = compute_community_N(community_pdfs, questions)
    return {
        "n_answers":   len(answers),
        "parameters":  community_pdfs,
        "community_N": community_N,
    }


def compute_all(data_file: Path, questions_file: Path) -> dict:
    """
    Load data.json + questions.json and return the full computation result.
    Kept for CLI / offline use.
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
        "n_sessions":  len(completed_ids),
        "n_answers":   len(all_answers),
        "parameters":  community_pdfs,
        "community_N": community_N,
    }
