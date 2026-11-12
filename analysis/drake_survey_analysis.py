"""
Drake survey: analysis and figures from the 3 CSVs exported by the backend
==========================================================================

Inputs (downloaded with the admin endpoints):
    sessions.csv   <- /api/export/sessions.csv
    answers.csv    <- /api/export/answers.csv
    context.csv    <- /api/export/context.csv
    questions.json <- the same file the website uses

Method ("A": pooled marginals, then multiply)
---------------------------------------------
1. Each person's answer (t_lo, t_mode, t_hi) defines a truncated split-normal
   PDF in T-space (T = log10 of the parameter). Same maths as compute.py.
2. For each parameter, all individual PDFs are put on a common grid and
   averaged with equal weight (linear opinion pool), then renormalised.
   People who skipped a parameter are left out of THAT parameter only.
3. N is obtained by Monte Carlo: draw each parameter independently from its
   pooled PDF and add the log10 values (log10 N = sum of log10 parameters).

Usage:
    python drake_survey_analysis.py
    python drake_survey_analysis.py --data-dir data --questions questions.json \
                                    --out-dir figures --n-mc 200000

Dependencies: numpy scipy pandas matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, NullLocator
from scipy.special import ndtr
from scipy.stats import gaussian_kde

# numpy >= 2.0 renamed trapz -> trapezoid
trapz = getattr(np, "trapezoid", None) or np.trapz

# ── constants (same as compute.py / the website) ──────────────────────────
Z90 = 1.2816
SIG_MIN = 0.02
N_GRID = 500

REF_MW = 0.0                      # log10(1)
REF_UNIV = float(np.log10(5e-13))  # about -12.301

DARK = (0.2, 0.2, 0.2)

# labels for the figures (English; change freely)
PARAM_LABELS = {
    "Ns": (r"$N_{\star}$", "Number of stars in the Milky Way"),
    "fp": (r"$f_{p}$", "Fraction of stars with planets"),
    "ne": (r"$n_{e}$", "Habitable planets per star"),
    "fl": (r"$f_{l}$", "Fraction of habitable planets\nwith life"),
    "fi": (r"$f_{i}$", "Fraction of planets with life\nthat develop intelligence"),
    "fc": (r"$f_{c}$", "Fraction of intelligent life capable\nof interstellar communication"),
    "fL": (r"$f_{L}$", "Fraction of stellar lifetime with a\ndetectable civilization"),
}

BELIEF_TITLES = {
    "life_solar": "Life elsewhere in the\nSolar System (now or past)",
    "life_galaxy": "Life now elsewhere in\nthe Milky Way",
    "intel_galaxy": "Detectable intelligent life now\nelsewhere in the Milky Way",
    "life_universe": "Life now elsewhere in the\nobservable universe",
    "intel_universe": "Detectable intelligent life now\nelsewhere in the observable universe",
}
SCALE_ORDER = ["very_unlikely", "unlikely", "unsure", "likely", "very_likely"]
SCALE_LABELS = ["Very\nunlikely", "Unlikely", "Unsure", "Likely", "Very\nlikely"]


# ── split-normal helpers (copied from compute.py) ─────────────────────────

def split_normal_pdf(t, t_mode, t_lo, t_hi, t_min, t_max):
    """Truncated split-normal PDF, integrates to 1 over [t_min, t_max]."""
    sig_L = max(SIG_MIN, (t_mode - t_lo) / Z90)
    sig_R = max(SIG_MIN, (t_hi - t_mode) / Z90)
    mass_L = float(ndtr(0) - ndtr((t_min - t_mode) / sig_L))
    mass_R = float(ndtr((t_max - t_mode) / sig_R) - ndtr(0))
    integral = np.sqrt(2 * np.pi) * (sig_L * mass_L + sig_R * mass_R)
    A = 1.0 / integral if integral > 0 else 1.0
    sig = np.where(t <= t_mode, sig_L, sig_R)
    y = A * np.exp(-0.5 * ((t - t_mode) / sig) ** 2)
    y = np.where((t >= t_min) & (t <= t_max), y, 0.0)
    return np.where(np.isfinite(y), y, 0.0)


def sample_from_grid_pdf(t_grid, pdf, n, rng):
    """Inverse-CDF sampling from a PDF tabulated on a grid."""
    dt = np.diff(t_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dt)])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, t_grid)


# ── loading ───────────────────────────────────────────────────────────────

def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "t", "1"])


def load_data(data_dir: Path):
    sessions = pd.read_csv(data_dir / "sessions.csv")
    answers = pd.read_csv(data_dir / "answers.csv")
    context = pd.read_csv(data_dir / "context.csv")

    sessions["completed"] = as_bool(sessions["completed"])
    done = set(sessions.loc[sessions["completed"], "id"])

    answers = answers[answers["session_id"].isin(done)].copy()
    answers["skipped"] = as_bool(answers["skipped"])
    context = context[context["session_id"].isin(done)].copy()
    return sessions[sessions["completed"]], answers, context


# ── community PDFs (linear opinion pool) ──────────────────────────────────

def pooled_pdf(rows: pd.DataFrame, q: dict):
    """Return (t_grid, individual_pdfs [n x N_GRID], pooled_pdf)."""
    t_min, t_max = q["tMin"], q["tMax"]
    t_grid = np.linspace(t_min, t_max, N_GRID)
    pdfs = np.array([
        split_normal_pdf(t_grid, r.t_mode, r.t_lo, r.t_hi, t_min, t_max)
        for r in rows.itertuples()
    ])
    pool = pdfs.mean(axis=0)
    pool /= trapz(pool, t_grid)
    return t_grid, pdfs, pool


def to_x(t, q):
    return 10.0 ** t if q.get("axisMode", "log") == "log" else t


# ── N by Monte Carlo (method A) ───────────────────────────────────────────

def community_logN(pools: dict, questions: list, n_mc: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    total = np.zeros(n_mc)
    for q in questions:
        t_grid, pool = pools[q["id"]]
        t = sample_from_grid_pdf(t_grid, pool, n_mc, rng)
        if q.get("axisMode", "log") == "log":
            total += t
        else:
            total += np.log10(np.maximum(t, 1e-300))
    return total


# ── plotting helpers ──────────────────────────────────────────────────────

def log_axis_ticks(ax):
    """
    Ticks of a log10 x-axis, same rules as the survey slider (survey.html):
      - every whole power of ten gets a tick; labels are thinned out by the
        number of orders of magnitude (OOM) on the axis:
            <= 10 OOM: every power | <= 20: every 2nd | <= 50: every 5th | else every 10th
        (and further thinned if the labels would still collide on a narrow panel)
        (labelled powers are multiples of the step, so 10^0 is always a candidate);
      - unlabelled powers of ten get a slightly shorter tick without text;
      - minor ticks, depending on the number of OOM:
            <= 3 OOM: all subdivisions 2..9 | 4-6 OOM: only the 5 | 7+ OOM: none.
    Call it AFTER the x-limits are final.
    """
    xl, yl = ax.get_xlim(), ax.get_ylim()
    t0, t1 = np.log10(xl[0]), np.log10(xl[1])
    span = t1 - t0
    step = 1 if span <= 10 else 2 if span <= 20 else 5 if span <= 50 else 10
    # like the website, also thin the labels out when they would collide
    # (about 0.5 inch per label like "10^-10"); the step is never below the rule above
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    needed = span * 0.5 / max(width_in, 0.5)
    step = max(step, next((k for k in (1, 2, 5, 10, 20) if k >= needed), 20))
    decades = list(range(int(np.ceil(t0 - 1e-9)), int(np.floor(t1 + 1e-9)) + 1))
    major = [e for e in decades if e % step == 0]
    unlabelled = [e for e in decades if e % step != 0]

    nice = {-2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100"}
    ax.set_xticks([10.0 ** e for e in major])
    ax.set_xticklabels([nice.get(e, f"$10^{{{e}}}$") for e in major])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())

    mult = [2, 3, 4, 5, 6, 7, 8, 9] if span <= 3 + 1e-9 else [5] if span <= 6 + 1e-9 else []
    minor = {k: [] for k in mult}
    for e in range(int(np.floor(t0)), int(np.ceil(t1))):
        for k in mult:
            t = e + np.log10(k)
            if t0 + 1e-9 < t < t1 - 1e-9:
                minor[k].append(10.0 ** t)

    def put(xs, size):
        if xs:
            ax.plot(xs, [0] * len(xs), transform=ax.get_xaxis_transform(), linestyle="none",
                    marker=mmarkers.TICKDOWN, markersize=size, markeredgewidth=0.8,
                    color=DARK, clip_on=False, zorder=3)

    put([10.0 ** e for e in unlabelled], 3.5)
    put(minor.get(5, []), 3)
    put([x for k, xs in minor.items() if k != 5 for x in xs], 2)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def style_axis(ax, xlabel, ylabel, title, logx=True):
    """Set the x-limits BEFORE calling this (the log ticks depend on them)."""
    ax.set_xlabel(xlabel, fontsize=12, color=DARK)
    ax.set_ylabel(ylabel, fontsize=10, color=DARK)
    ax.set_title(title, fontsize=10, pad=10, color=DARK)
    if logx:
        xl = ax.get_xlim()
        ax.set_xscale("log")
        ax.set_xlim(xl)
    ax.tick_params(which="major", direction="out", length=4, color=DARK)
    ax.tick_params(which="minor", direction="in", length=0)
    ax.tick_params(axis="both", colors=DARK)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    if logx:
        log_axis_ticks(ax)


def figure_context(context: pd.DataFrame, out: Path):
    # panel order = order of BELIEF_TITLES (the order shown to participants);
    # the CSV columns come alphabetically sorted from the export, so they must not set the order.
    # Any belief column not listed in BELIEF_TITLES goes at the end.
    others = [c for c in context.columns
              if c not in ("session_id", "area_grande", "area_especifica")]
    belief_cols = [c for c in BELIEF_TITLES if c in others] + [c for c in others if c not in BELIEF_TITLES]
    n_panels = len(belief_cols) + 1
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Participants' background and beliefs (n = {len(context)} completed sessions)",
                 fontsize=14, color=DARK)

    # grande area only (area_especifica is never shown: re-identification risk)
    ax = axes[0]
    areas = context["area_grande"].dropna().value_counts().sort_values()
    ax.barh(areas.index, areas.values, color=(0, 0, 1, 0.25), edgecolor="blue")
    for y, v in enumerate(areas.values):
        ax.text(v + 0.05, y, str(v), va="center", fontsize=9, color=DARK)
    style_axis(ax, "Participants", "", "Broad field (CNPq)", logx=False)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, max(areas.max() * 1.15, 1) if len(areas) else 1)

    for ax, col in zip(axes[1:], belief_cols):
        counts = context[col].value_counts().reindex(SCALE_ORDER).fillna(0).astype(int)
        ax.bar(range(5), counts.values, color=(1, 0, 0, 0.2), edgecolor="red")
        for x, v in enumerate(counts.values):
            ax.text(x, v + 0.05, str(v), ha="center", va="bottom", fontsize=9, color=DARK)
        ax.set_xticks(range(5))
        ax.set_xticklabels(SCALE_LABELS, fontsize=8)
        style_axis(ax, "", "Participants",
                   f"{BELIEF_TITLES.get(col, col)}\n(answered: {counts.sum()})", logx=False)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_ylim(0, max(counts.max() * 1.18, 1))

    for ax in axes[n_panels:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def draw_param(ax, q, t_grid, pdfs, pool, n):
    x = to_x(t_grid, q)
    #for p in pdfs:                       # thin individual curves
    #    ax.plot(x, p, color="blue", alpha=0.12, linewidth=0.8)
    ax.plot(x, pool, color="blue", linewidth=2)
    ax.fill_between(x, pool, color="blue", alpha=0.1)
    ax.set_xlim(x[0], x[-1])             # axis = the question's own range, as in the survey
    sym, title = PARAM_LABELS.get(q["id"], (q["id"], q["id"]))
    style_axis(ax, sym, "Prob. density per decade", f"{title}\n(n = {n})",
               logx=q.get("axisMode", "log") == "log")
    ax.set_ylim(bottom=0)


def n_axis_limits(log_N):
    """x-limits (in log10 N) shared by the PDF and CDF panels."""
    lo = min(np.percentile(log_N, 0.1) - 1, REF_UNIV - 1)
    hi = max(np.percentile(log_N, 99.9) + 1, REF_MW + 1)
    return float(lo), float(hi)


def fmt_pct(p):
    if p < 0.5:
        return "<1%"
    if p > 99.5:
        return ">99%"
    return f"{p:.0f}%"


def annotate_N_regions(ax, grid, pdf, probs):
    """
    Labels inside the big N figure, in the style of drake_metaculus.py: the same
    texts, a big percentage at the bottom of each region and dashed reference
    lines that go up to the curve only. The x-positions are computed from the
    regions themselves (not hand-tuned), so they adapt to any data.
    """
    ymax = ax.get_ylim()[1]
    lo, hi = grid[0], grid[-1]
    dens = lambda t: float(np.interp(t, grid, pdf))
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.55)

    ax.vlines(x=1, ymin=0, ymax=dens(REF_MW), color="green", linestyle="dashed",
              linewidth=1.5, zorder=3)
    ax.vlines(x=5e-13, ymin=0, ymax=dens(REF_UNIV), color="red", linestyle="dashed",
              linewidth=1.5, zorder=4)

    def put(t_center, y_text, text, pct, ha="center"):
        x = 10.0 ** t_center
        ax.text(x, y_text, text, fontsize=8, color=DARK, ha=ha, va="center",
                zorder=10, bbox=box)
        ax.text(x, ymax * 0.05, pct, fontsize=12, color=DARK, ha=ha, va="bottom",
                zorder=10)

    # right region: N > 1 (if it is too narrow for the text, right-align it to the axis end)
    narrow = (hi - REF_MW) / (hi - lo) < 0.16
    put(hi - 0.15 if narrow else (REF_MW + hi) / 2,
        max(ymax * 0.3, dens(REF_MW) * 0.5),
        "Probability of\nNOT being\nalone in the\nMilky Way\ngalaxy\n($N > 1$): "
        + fmt_pct(probs["not_alone_mw"]),
        fmt_pct(probs["not_alone_mw"]), ha="right" if narrow else "center")

    # middle region: 5e-13 < N < 1 (alone in the Milky Way, but not in the universe)
    put((REF_UNIV + REF_MW) / 2,
        max(ymax * 0.2, min(dens(REF_UNIV) * 1.1, ymax * 0.8)),
        "Probability of being alone\nin the Milky Way galaxy\n($N < 1$): "
        + fmt_pct(probs["alone_mw"]),
        fmt_pct(probs["alone_mw"]))

    # left region: N < 5e-13 (alone in the observable universe)
    put((lo + REF_UNIV) / 2,
        max(ymax * 0.2, min(dens(REF_UNIV) * 0.5, ymax * 0.8)),
        "Probability of being alone\nin the observable Universe\n($N < 5 \\times 10^{-13}$): "
        + fmt_pct(probs["alone_univ"]),
        fmt_pct(probs["alone_univ"]))


def draw_N_pdf(ax, log_N, probs, big=False):
    sub = log_N if len(log_N) <= 30_000 else np.random.default_rng(0).choice(log_N, 30_000, replace=False)
    kde = gaussian_kde(sub, bw_method="scott")
    lo, hi = n_axis_limits(log_N)
    grid = np.linspace(lo, hi, 800)
    pdf = kde(grid)
    pdf /= trapz(pdf, grid)
    x = 10.0 ** grid

    ax.plot(x, pdf, color="red", linewidth=2, zorder=5)
    ax.fill_between(x, pdf, color="red", alpha=0.1)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)

    if big:
        ax.fill_between(x, pdf, where=grid >= REF_MW, hatch="||", facecolor="none",
                        edgecolor=(0, 0, 0, 0.3), linewidth=0, zorder=1)
        ax.fill_between(x, pdf, where=grid <= REF_MW, hatch="\\\\", facecolor=(0, 1, 0, 0.1),
                        edgecolor=(0, 0, 0, 0.3), linewidth=0, zorder=1)
        ax.fill_between(x, pdf, where=grid <= REF_UNIV, hatch="/", facecolor=(0, 1, 0, 0.1),
                        edgecolor=(0, 0, 0, 0.3), linewidth=0, zorder=2)
        annotate_N_regions(ax, grid, pdf, probs)


def draw_N_cdf(ax, log_N):
    q = np.linspace(0, 1, 2000)
    ax.plot(10.0 ** np.quantile(log_N, q), q, color="red", linewidth=2, zorder=7)
    ax.fill_between(10.0 ** np.quantile(log_N, q), q, color="red", alpha=0.1)
    lo, hi = n_axis_limits(log_N)
    ax.set_xlim(10.0 ** lo, 10.0 ** hi)
    ax.axvline(1, color="green", linestyle="dashed", linewidth=1.5)
    ax.axvline(5e-13, color="red", linestyle="dashed", linewidth=1.5)

# ── cumulative products: N using only the first k parameters ──────────────

def community_logN_cumulative(pools: dict, questions: list, n_mc: int,
                              seed: int = 1, k_min: int = 3) -> dict:
    """
    log10 of the product of the first k parameters, for k = k_min ... len(questions).

    Each parameter is sampled ONCE (same order and same seed as community_logN),
    and the running sum of log10 values is stored after each parameter. So the
    curves are nested, and k = 7 is identical to community_logN().
    Returns {k: array of log10 samples}.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(n_mc)
    out = {}
    for i, q in enumerate(questions, start=1):
        t_grid, pool = pools[q["id"]]
        t = sample_from_grid_pdf(t_grid, pool, n_mc, rng)
        if q.get("axisMode", "log") == "log":
            total += t
        else:
            total += np.log10(np.maximum(t, 1e-300))
        if i >= k_min:
            out[i] = total.copy()
    return out


# what the product of the first k terms means (k = 7 is N itself)
CUM_MEANING = {
    3: "Habitable planets in the Milky Way",
    4: "Planets with life",
    5: "Planets with intelligent life",
    6: "Planets that ever develop civilizations\ncapable of interstellar communication",
    7: "Detectable civilizations now ($N$)",
}


def cum_formula(questions: list, k: int) -> str:
    """Mathtext of the product of the first k parameters, e.g. $N_{\\star}\\,f_{p}\\,n_{e}$."""
    syms = [PARAM_LABELS.get(q["id"], (q["id"], ""))[0].strip("$") for q in questions[:k]]
    return "$" + r"\,\cdot\," .join(syms) + "$"


def summarize_logX(log_X: np.ndarray) -> dict:
    pc = lambda p: float(np.percentile(log_X, p))
    return {
        "median": 10 ** pc(50),
        "q05": 10 ** pc(5), "q25": 10 ** pc(25),
        "q75": 10 ** pc(75), "q95": 10 ** pc(95),
        "p_gt_1": 100 * float(np.mean(log_X > 0)),
    }


def kde_curve(log_X: np.ndarray, lo: float, hi: float, n: int = 800):
    sub = log_X if len(log_X) <= 30_000 else np.random.default_rng(0).choice(log_X, 30_000, replace=False)
    kde = gaussian_kde(sub, bw_method="scott")
    grid = np.linspace(lo, hi, n)
    pdf = kde(grid)
    pdf /= trapz(pdf, grid)
    return grid, pdf


def limits_logX(log_X: np.ndarray):
    return float(np.percentile(log_X, 0.1) - 1), float(np.percentile(log_X, 99.9) + 1)


# noun phrase used in the labels of each cumulative figure
CUM_NOUN = {
    3: "habitable\nplanet",
    4: "planet\nwith life",
    5: "planet with\nintelligent life",
    6: "planet with\ncommunicating\ncivilizations",
}


def draw_partial_N(ax, log_X: np.ndarray, k: int, questions: list, n_sessions: int,
                   show_universe: bool = False):
    """
    One figure for the product of the first k parameters (k < 7), in the same
    style as the final N figure: hatched regions, dashed reference lines and
    the probability of each region written inside the figure.

    show_universe=True adds the observable-universe part (line at 5e-13, extra
    hatching and a third label). The axis is then widened to contain both
    reference lines, as in the final N figure.

    If 1 falls outside the axis (show_universe=False and all the mass on one
    side), a single label says so.
    """
    lo, hi = limits_logX(log_X)
    if show_universe:
        lo = min(lo, REF_UNIV - 1)
        hi = max(hi, REF_MW + 1)
    grid, pdf = kde_curve(log_X, lo, hi)
    x = 10.0 ** grid
    p_gt = 100 * float(np.mean(log_X > REF_MW))        # P(X > 1)
    p_lt = 100 * float(np.mean(log_X < REF_MW))        # P(X < 1)
    p_univ = 100 * float(np.mean(log_X < REF_UNIV))    # P(X < 5e-13)
    noun = CUM_NOUN.get(k, "such planets")

    ax.plot(x, pdf, color="red", linewidth=2, zorder=5)
    ax.fill_between(x, pdf, color="red", alpha=0.1)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, float(pdf.max()) * 1.18)
    ymax = ax.get_ylim()[1]

    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.55)
    dens = lambda t: float(np.interp(t, grid, pdf))

    def put(t_center, y_text, text, pct, ha="center"):
        xx = 10.0 ** t_center
        ax.text(xx, y_text, text, fontsize=8, color=DARK, ha=ha, va="center",
                zorder=10, bbox=box)
        ax.text(xx, ymax * 0.05, pct, fontsize=12, color=DARK, ha=ha, va="bottom",
                zorder=10)

    if lo < REF_MW < hi:
        # hatches, as in the final N figure
        ax.fill_between(x, pdf, where=grid >= REF_MW, hatch="||", facecolor="none",
                        edgecolor=(0, 0, 0, 0.3), linewidth=0, zorder=1)
        ax.fill_between(x, pdf, where=grid <= REF_MW, hatch="\\\\", facecolor=(0, 1, 0, 0.1),
                        edgecolor=(0, 0, 0, 0.3), linewidth=0, zorder=1)
        ax.vlines(x=1, ymin=0, ymax=dens(REF_MW), color="green", linestyle="dashed",
                  linewidth=1.5, zorder=3)

        # right region: X > 1 (right-aligned to the axis end if too narrow for the text)
        narrow_r = (hi - REF_MW) / (hi - lo) < 0.16
        put(hi - 0.15 if narrow_r else (REF_MW + hi) / 2,
            max(ymax * 0.3, dens(REF_MW) * 0.5),
            f"Probability of\nMORE than one\n{noun}\nin the Milky Way\n($> 1$): {fmt_pct(p_gt)}",
            fmt_pct(p_gt), ha="right" if narrow_r else "center")

        if show_universe and lo < REF_UNIV:
            # extra hatching and line for the observable universe
            ax.fill_between(x, pdf, where=grid <= REF_UNIV, hatch="/", facecolor=(0, 1, 0, 0.1),
                            edgecolor=(0, 0, 0, 0.3), linewidth=0, zorder=2)
            ax.vlines(x=5e-13, ymin=0, ymax=dens(REF_UNIV), color="red", linestyle="dashed",
                      linewidth=1.5, zorder=4)

            # middle region: 5e-13 < X < 1 (the percentage is P(X < 1), as in the final figure)
            put((REF_UNIV + REF_MW) / 2,
                max(ymax * 0.2, min(dens(REF_UNIV) * 1.1, ymax * 0.8)),
                f"Probability of\nFEWER than one\n{noun}\nin the Milky Way\n($< 1$): {fmt_pct(p_lt)}",
                fmt_pct(p_lt))

            # left region: X < 5e-13 (fewer than one in the whole observable universe)
            narrow_l = (REF_UNIV - lo) / (hi - lo) < 0.16
            put(lo + 0.15 if narrow_l else (lo + REF_UNIV) / 2,
                max(ymax * 0.2, min(dens(REF_UNIV) * 0.5, ymax * 0.8)),
                f"Probability of\nFEWER than one\n{noun}\nin the observable\nUniverse\n"
                f"($< 5 \\times 10^{{-13}}$): {fmt_pct(p_univ)}",
                fmt_pct(p_univ), ha="left" if narrow_l else "center")
        else:
            # left region: X < 1 (no universe part)
            narrow_l = (REF_MW - lo) / (hi - lo) < 0.16
            put(lo + 0.15 if narrow_l else (lo + REF_MW) / 2,
                max(ymax * 0.3, dens(REF_MW) * 0.5),
                f"Probability of\nFEWER than one\n{noun}\nin the Milky Way\n($< 1$): {fmt_pct(p_lt)}",
                fmt_pct(p_lt), ha="left" if narrow_l else "center")
    else:
        # 1 is outside the axis: all the mass is on one side
        above = lo > REF_MW
        p = p_gt if above else p_lt
        word, sign = ("MORE", ">") if above else ("FEWER", "<")
        put((lo + hi) / 2, ymax * 0.35,
            f"Probability of\n{word} than one\n{noun}\nin the Milky Way\n(${sign} 1$): {fmt_pct(p)}",
            fmt_pct(p))

    style_axis(ax, cum_formula(questions, k), "Prob. density per decade",
               f"{CUM_MEANING[k]}\n(first {k} parameters; n = {n_sessions} participants)")


def figure_cumulative_overlay(cum: dict, questions: list, out: Path, n_sessions: int):
    """All the cumulative curves in one panel, to see how the distribution moves and widens."""
    lo = min(limits_logX(v)[0] for v in cum.values())
    hi = max(limits_logX(v)[1] for v in cum.values())
    ks = sorted(cum)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for c, k in zip(colors, ks):
        grid, pdf = kde_curve(cum[k], lo, hi)
        ax.plot(10.0 ** grid, pdf, color=c, linewidth=2, label=f"{k}: {cum_formula(questions, k)}")
        ax.fill_between(10.0 ** grid, pdf, color=c, alpha=0.06)
    ax.set_xlim(10.0 ** lo, 10.0 ** hi)
    ax.set_ylim(bottom=0)
    ax.axvline(1, color="green", linestyle="dashed", linewidth=1.2)
    ax.axvline(5e-13, color="red", linestyle="dashed", linewidth=1.2)
    style_axis(ax, "Product of the first $k$ parameters", "Prob. density per decade",
               f"Adding one Drake parameter at a time (community, n = {n_sessions} participants)")
    ax.legend(fontsize=8, frameon=False, loc="upper right", title="k : terms included", title_fontsize=8)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)

# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder with sessions.csv, answers.csv, context.csv")
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--n-mc", type=int, default=200_000, help="Monte Carlo samples for N")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k-min", type=int, default=3,
                    help="smallest number of parameters in the cumulative figures")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    print("Loading the data...")
    sessions, answers, context = load_data(data_dir)
    print(f"  completed sessions: {len(sessions)}")
    if len(sessions) < 10:
        print("  WARNING: very few sessions; treat the curves as anecdotal.")

    # ── context bar charts ──
    print("Making the context figure...")
    figure_context(context, out_dir / "drake_survey_context.png")

    # ── parameters ──
    print("Pooling the parameters...")
    pools, summary = {}, []
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5))
    axes = axes.ravel()
    fig.suptitle("Drake equation factors: community opinion pool "
                 "(thin = individuals, thick = pooled)", fontsize=14, color=DARK)

    missing = []
    for ax, q in zip(axes, questions):
        rows = answers[(answers["question_id"] == q["id"]) & (~answers["skipped"])]
        n = len(rows)
        print(f"  {q['id']:>3}: {n} answers (skipped: "
              f"{int(((answers['question_id'] == q['id']) & answers['skipped']).sum())})")
        if n == 0:
            missing.append(q["id"])
            ax.set_visible(False)
            continue
        t_grid, pdfs, pool = pooled_pdf(rows, q)
        pools[q["id"]] = (t_grid, pool)
        draw_param(ax, q, t_grid, pdfs, pool, n)

        cdf = np.concatenate([[0], np.cumsum(0.5 * (pool[1:] + pool[:-1]) * np.diff(t_grid))])
        qt = lambda p: float(np.interp(p, cdf / cdf[-1], t_grid))
        summary.append({"parameter": q["id"], "n_answers": n,
                        "T_mode_of_pool": float(t_grid[np.argmax(pool)]),
                        "T_q10": qt(0.10), "T_q25": qt(0.25), "T_median": qt(0.5),
                        "T_q75": qt(0.75), "T_q90": qt(0.90)})

    if missing:
        print(f"  Cannot compute N: no answers for {missing}")
        plt.tight_layout()
        plt.savefig(out_dir / "drake_survey_PDFs.png", bbox_inches="tight", dpi=150)
        return

    # ── N (method A) ──
    print(f"Monte Carlo for N ({args.n_mc:,} samples)...")
    log_N = community_logN(pools, questions, args.n_mc, args.seed)
    probs = {
        "not_alone_mw": 100 * np.mean(log_N > REF_MW),
        "alone_mw": 100 * np.mean(log_N < REF_MW),
        "alone_univ": 100 * np.mean(log_N < REF_UNIV),
    }

    draw_N_pdf(axes[7], log_N, probs)
    style_axis(axes[7], "$N$", "Prob. density per decade", "Distribution of N (community)")
    axes[7].axvline(1, color="green", linestyle="dashed", linewidth=1)
    axes[7].axvline(5e-13, color="red", linestyle="dashed", linewidth=1)
    draw_N_cdf(axes[8], log_N)
    style_axis(axes[8], "$N$", "Cumulative probability", "CDF of N (community)")
    plt.tight_layout()
    plt.savefig(out_dir / "drake_survey_PDFs.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── big N figure ──
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    draw_N_pdf(ax, log_N, probs, big=True)
    style_axis(ax, "$N$", "Prob. density per decade",
               f"Community distribution of the number of detectable civilizations\n"
               f"in the Milky Way (n = {len(sessions)} participants)")
    plt.tight_layout()
    plt.savefig(out_dir / "drake_survey_N.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── cumulative figures: first k parameters (k = k_min ... 7) ──
    print("Cumulative figures (adding one parameter at a time)...")
    cum = community_logN_cumulative(pools, questions, args.n_mc, args.seed, k_min=args.k_min)
    cum_rows = []
    for k, log_X in cum.items():
        st = summarize_logX(log_X)
        cum_rows.append({"k": k, "terms": "*".join(q["id"] for q in questions[:k]),
                         "meaning": CUM_MEANING.get(k, "").replace("\n", " "),
                         "median": st["median"], "q05": st["q05"], "q25": st["q25"],
                         "q75": st["q75"], "q95": st["q95"], "P_gt_1_pct": st["p_gt_1"]})
        print(f"  k={k}: median = {st['median']:.3g}, "
              f"90% = [{st['q05']:.3g}, {st['q95']:.3g}], P(>1) = {st['p_gt_1']:.1f}%")
        if k < len(questions):                      # k = 7 already has drake_survey_N.png
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            draw_partial_N(ax, log_X, k, questions, len(sessions),
               show_universe=(k >= len(questions) - 2))
            plt.tight_layout()
            plt.savefig(out_dir / f"drake_survey_N_first{k}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)
    figure_cumulative_overlay(cum, questions, out_dir / "drake_survey_cumulative.png", len(sessions))
    pd.DataFrame(cum_rows).to_csv(data_dir / "drake_survey_cumulative_summary.csv", index=False)

    # ── numbers to the terminal and to a CSV ──
    pd.DataFrame(summary).to_csv("data/drake_survey_param_summary.csv", index=False)
    pc = lambda p: float(np.percentile(log_N, p))
    print("\nResults (community, method A)")
    print(f"  median N       = {10**pc(50):.3g}")
    print(f"  IQR            = [{10**pc(25):.3g}, {10**pc(75):.3g}]")
    print(f"  90% interval   = [{10**pc(5):.3g}, {10**pc(95):.3g}]")
    print(f"  P(N > 1)       = {probs['not_alone_mw']:.1f}%   (not alone in the Milky Way)")
    print(f"  P(N < 1)       = {probs['alone_mw']:.1f}%   (alone in the Milky Way)")
    print(f"  P(N < 5e-13)   = {probs['alone_univ']:.1f}%   (alone in the observable universe)")
    print(f"\nFigures saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
