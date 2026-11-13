"""
Drake survey: exploratory comparison by academic background (CNPq grande área)
==============================================================================

Put this file next to drake_survey_analysis.py (analysis/) and run it from the
repository root:

    python analysis/drake_survey_by_area.py
    python analysis/drake_survey_by_area.py --include-others --save-person-table

Inputs  (same as drake_survey_analysis.py): data/sessions.csv, data/answers.csv,
        data/context.csv, questions.json

THIS IS AN EXPLORATORY ANALYSIS. With ~10-15 people per group only very large
differences can be detected; report effect sizes with their intervals, and treat
p-values as a rough guide (a Holm-adjusted p is given, per family of tests).

What it does
------------
Groups (decided up front, not by looking at the data):
    Biológicas = "Ciências Biológicas"
    Exatas     = "Ciências Exatas e da Terra"
    Outras     = every other grande área (shown in the plots, never tested)
    Sessions without an area are left out (and counted).
Groups with fewer than --min-n people are not shown at all (re-identification).
area_especifica is never used.

1. Per-person quantities (NOT pooled curves). For every person and parameter:
   median, 10th-90th percentile of their own truncated split-normal, and the
   width (q90 - q10, in decades). Plus, from a per-person Monte Carlo:
       astro = log10(Ns * fp * ne)            (first three parameters)
       bio   = log10(fl * fi * fc * fL)       (last four parameters)
       N     = log10 of the full product, and P(N > 1) per person
   Each person's N keeps THEIR OWN correlations between parameters, which the
   pooled-marginals method (A) throws away.
2. Tests, Biológicas - Exatas: difference of group medians (in decades) with a
   bootstrap interval over people, Cliff's delta, Mann-Whitney p, Holm p.
   Two families: location (median) and width (q90 - q10).
3. Skip rates and "edge" checks per group and parameter: skipped, mode near the
   lower/upper axis limit (floor/ceiling effect), wing pinned at the limit, and
   answers identical to the slider's starting position (anchoring).
4. Pooled curves per group with bootstrap bands over people, and N by two
   methods: B = mean of individual N distributions, A = pooled marginals
   multiplied independently (as in drake_survey_analysis.py).
5. Beliefs by group, and Spearman correlation belief vs. N.

Outputs
-------
figures/drake_by_area_dots.png        one dot per person, per parameter
figures/drake_by_area_profile.png     Biológicas - Exatas differences (location, width)
figures/drake_by_area_pooled.png      pooled curves per group + bootstrap bands
figures/drake_by_area_skips_edges.png skip rates and edge/anchoring checks
figures/drake_by_area_beliefs.png     beliefs by group
data/drake_by_area_tests.csv
data/drake_by_area_N_summary.csv
data/drake_by_area_skips_edges.csv
data/drake_by_area_belief_corr.csv
data/drake_by_area_persons.csv        only with --save-person-table (do not commit)

Dependencies: numpy scipy pandas matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr, ndtri
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drake_survey_analysis as base  # noqa: E402  (reuses its helpers and constants)

trapz = base.trapz
Z90, SIG_MIN = base.Z90, base.SIG_MIN
REF_MW, REF_UNIV = base.REF_MW, base.REF_UNIV
DARK = base.DARK

# ── groups ────────────────────────────────────────────────────────────────
G_BIO, G_EXATAS, G_OTHER = "Biológicas", "Exatas", "Outras"
PRIMARY = (G_BIO, G_EXATAS)                  # differences are always BIO - EXATAS
AREA_TO_GROUP = {
    "Ciências Biológicas": G_BIO,
    "Ciências Exatas e da Terra": G_EXATAS,
}
COLORS = {G_BIO: "tab:green", G_EXATAS: "tab:blue", G_OTHER: "0.5"}

ASTRO_IDS = ["Ns", "fp", "ne"]
BIO_IDS = ["fl", "fi", "fc", "fL"]
DERIVED_LABELS = {
    "astro": "astro\n$N_\\star f_p n_e$",
    "bio": "bio/social\n$f_l f_i f_c f_L$",
    "N": "$N$",
}


def group_of(area) -> str | None:
    if area is None or (isinstance(area, float) and np.isnan(area)) or str(area).strip() == "":
        return None
    return AREA_TO_GROUP.get(str(area).strip(), G_OTHER)


# ── truncated split-normal: exact quantile function ───────────────────────
# Same distribution as the website / compute.py PDF. The weight of each half
# is sigma * visible Gaussian mass, so that the PDF is continuous at the mode.
# (compute.py's sampler weights the halves by the mass only; see the notes
#  delivered with this script.)

def _trunc_parts(t_mode, t_lo, t_hi, t_min, t_max):
    sL = max(SIG_MIN, (t_mode - t_lo) / Z90)
    sR = max(SIG_MIN, (t_hi - t_mode) / Z90)
    pL = float(ndtr((t_min - t_mode) / sL))      # Phi(a_L)
    pR = float(ndtr((t_max - t_mode) / sR))      # Phi(a_R)
    wL = sL * (0.5 - pL)                         # sqrt(2 pi) cancels everywhere
    wR = sR * (pR - 0.5)
    return sL, sR, pL, wL, wR


def trunc_ppf(p, t_mode, t_lo, t_hi, t_min, t_max):
    """Quantile function of the truncated split-normal on [t_min, t_max]."""
    sL, sR, pL, wL, wR = _trunc_parts(t_mode, t_lo, t_hi, t_min, t_max)
    p = np.atleast_1d(np.asarray(p, dtype=float))
    tot = wL + wR
    if tot <= 0:
        return np.full(p.shape, float(t_mode))
    pc = p * tot
    uL = np.clip(pL + pc / sL, 1e-300, 1 - 1e-16)
    uR = np.clip(0.5 + (pc - wL) / sR, 1e-300, 1 - 1e-16)
    t = np.where(pc <= wL, t_mode + sL * ndtri(uL), t_mode + sR * ndtri(uR))
    return np.clip(t, t_min, t_max)


def log10_of(t: np.ndarray, q: dict) -> np.ndarray:
    if q.get("axisMode", "log") == "log":
        return t
    return np.log10(np.maximum(t, 1e-300))


# ── small statistics helpers ──────────────────────────────────────────────

def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    d = a[:, None] - b[None, :]
    return float((np.sum(d > 0) - np.sum(d < 0)) / d.size)


def boot_median_diff(a, b, n_boot, ci, rng):
    ma = np.median(rng.choice(a, (n_boot, len(a))), axis=1)
    mb = np.median(rng.choice(b, (n_boot, len(b))), axis=1)
    d = ma - mb
    lo, hi = np.percentile(d, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def hist_density(samples, edges, sigma_bins):
    h, _ = np.histogram(samples, bins=edges, density=True)
    h = gaussian_filter1d(h, sigma_bins, mode="nearest")
    area = h.sum() * (edges[1] - edges[0])
    return h / area if area > 0 else h


def summarize_logN(s: np.ndarray) -> dict:
    p = np.percentile(s, [5, 25, 50, 75, 95])
    return {
        "median": 10 ** p[2], "q05": 10 ** p[0], "q25": 10 ** p[1],
        "q75": 10 ** p[3], "q95": 10 ** p[4],
        "P_gt_1_pct": 100 * float(np.mean(s > REF_MW)),
        "P_lt_1_pct": 100 * float(np.mean(s < REF_MW)),
        "P_lt_5e-13_pct": 100 * float(np.mean(s < REF_UNIV)),
    }


# ── per-person table ──────────────────────────────────────────────────────

def build_person_table(answers, questions, sess_group, shown, n_mc, rng):
    """One row per person. Also returns {pid: log10 N samples} for complete people."""
    qids = [q["id"] for q in questions]
    recs, n_samples = [], {}
    pid = 0
    for sid, rows in answers.groupby("session_id", sort=True):
        g = sess_group.get(sid)
        if g not in shown:
            continue
        by_q = {r.question_id: r for r in rows.itertuples()}
        rec = {"pid": pid, "session_id": sid, "group": g}
        logs = {}
        for q in questions:
            k = q["id"]
            r = by_q.get(k)
            if r is None or bool(r.skipped):
                rec[f"{k}_skipped"] = True
                for s in ("q10", "q50", "q90", "w", "mode"):
                    rec[f"{k}_{s}"] = np.nan
                continue
            args = (r.t_mode, r.t_lo, r.t_hi, q["tMin"], q["tMax"])
            q10, q50, q90 = trunc_ppf([0.1, 0.5, 0.9], *args)
            rec.update({f"{k}_skipped": False, f"{k}_q10": q10, f"{k}_q50": q50,
                        f"{k}_q90": q90, f"{k}_w": q90 - q10, f"{k}_mode": r.t_mode})
            logs[k] = log10_of(trunc_ppf(rng.uniform(size=n_mc), *args), q)

        for name, ids in (("astro", ASTRO_IDS), ("bio", BIO_IDS), ("N", qids)):
            ids = [i for i in ids if i in qids]
            if ids and all(i in logs for i in ids):
                s = sum(logs[i] for i in ids)
                p10, p50, p90 = np.percentile(s, [10, 50, 90])
                rec.update({f"{name}_q10": p10, f"{name}_q50": p50,
                            f"{name}_q90": p90, f"{name}_w": p90 - p10})
                if name == "N":
                    rec["N_pgt1"] = float(np.mean(s > REF_MW))
                    rec["N_plt_univ"] = float(np.mean(s < REF_UNIV))
                    n_samples[pid] = s
            else:
                for sfx in ("q10", "q50", "q90", "w"):
                    rec[f"{name}_{sfx}"] = np.nan
                if name == "N":
                    rec["N_pgt1"] = np.nan
                    rec["N_plt_univ"] = np.nan
        recs.append(rec)
        pid += 1
    return pd.DataFrame(recs), n_samples


# ── tests ─────────────────────────────────────────────────────────────────

def compare_groups(persons, keys, n_boot, ci, rng, min_group=3):
    ga, gb = PRIMARY
    A = persons[persons.group == ga]
    B = persons[persons.group == gb]
    rows = []
    specs = [("location", "q50", keys), ("width", "w", keys), ("prob", "pgt1", ["N"])]
    for family, sfx, klist in specs:
        for k in klist:
            col = "N_pgt1" if sfx == "pgt1" else f"{k}_{sfx}"
            a = A[col].dropna().to_numpy()
            b = B[col].dropna().to_numpy()
            row = {"quantity": k, "family": family, "group_a": ga, "group_b": gb,
                   "n_a": len(a), "n_b": len(b),
                   "median_a": np.median(a) if len(a) else np.nan,
                   "median_b": np.median(b) if len(b) else np.nan,
                   "diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                   "cliffs_delta": np.nan, "p_mwu": np.nan}
            if len(a) >= min_group and len(b) >= min_group:
                row["diff"] = float(np.median(a) - np.median(b))
                row["ci_lo"], row["ci_hi"] = boot_median_diff(a, b, n_boot, ci, rng)
                row["cliffs_delta"] = cliffs_delta(a, b)
                row["p_mwu"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for family in df.family.unique():
        m = df.family == family
        df.loc[m, "p_holm"] = holm(df.loc[m, "p_mwu"].to_numpy())
    return df


# ── skips / edges / anchoring ─────────────────────────────────────────────

def skips_and_edges(answers, questions, sess_group, groups, edge):
    ans = answers.copy()
    ans["group"] = ans.session_id.map(sess_group)
    rows = []
    for g in groups:
        for q in questions:
            sub = ans[(ans.group == g) & (ans.question_id == q["id"])]
            ok = sub[~sub.skipped]
            tmin, tmax = q["tMin"], q["tMax"]
            c, sp = (tmin + tmax) / 2, q.get("initSpread", 0)
            default_like = ((ok.t_mode - c).abs() < 0.05) & \
                           ((ok.t_lo - np.maximum(tmin, c - sp)).abs() < 0.05) & \
                           ((ok.t_hi - np.minimum(tmax, c + sp)).abs() < 0.05)
            rows.append({
                "group": g, "parameter": q["id"], "n_total": len(sub),
                "n_skipped": int(sub.skipped.sum()), "n_answered": len(ok),
                "n_mode_near_floor": int((ok.t_mode <= tmin + edge).sum()),
                "n_mode_near_ceiling": int((ok.t_mode >= tmax - edge).sum()),
                "n_lo_pinned": int((ok.t_lo <= tmin + 1e-6).sum()),
                "n_hi_pinned": int((ok.t_hi >= tmax - 1e-6).sum()),
                "n_default_like": int(default_like.sum()),
            })
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────────

def _pow10_fmt():
    return FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$")


def fig_dots(persons, questions, groups, tests, out, rng):
    panels = [(q["id"], base.PARAM_LABELS.get(q["id"], (q["id"], q["id"]))[1].replace("\n", " "), q)
              for q in questions]
    panels += [("astro", "Product of the first 3 parameters (astro)", None),
               ("bio", "Product of the last 4 parameters (bio/social)", None),
               ("N", "$N$ (per-person median)", None),
               ("N_pgt1", "$P(N>1)$ (per person)", None)]
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    ga, gb = PRIMARY
    fig.suptitle("One dot = one person (median of their distribution; thin bar = their 10th-90th percentile)\n"
                 "black line = group median · dotted = axis limits of the question", fontsize=11, color=DARK)

    for ax, (key, title, q) in zip(axes, panels):
        is_prob = key == "N_pgt1"
        counts = []
        for gi, g in enumerate(groups):
            sub = persons[persons.group == g]
            y = sub["N_pgt1" if is_prob else f"{key}_q50"]
            ok = y.notna().to_numpy()
            x = gi + rng.uniform(-0.22, 0.22, size=len(sub))
            if not is_prob:
                ax.vlines(x[ok], sub[f"{key}_q10"][ok], sub[f"{key}_q90"][ok],
                          color=COLORS[g], alpha=0.35, lw=1)
            ax.scatter(x[ok], y[ok], s=24, color=COLORS[g], edgecolor="white", lw=0.5, zorder=3)
            if ok.any():
                ax.hlines(float(np.median(y[ok])), gi - 0.32, gi + 0.32, color="black", lw=2, zorder=4)
            counts.append(int(ok.sum()))
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(groups, counts)], fontsize=8)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        if q is not None:
            ax.set_ylim(q["tMin"] - 0.4, q["tMax"] + 0.4)
            ax.axhline(q["tMin"], color="0.6", ls=":", lw=1)
            ax.axhline(q["tMax"], color="0.6", ls=":", lw=1)
        if is_prob:
            ax.set_ylim(-0.03, 1.03)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(_pow10_fmt())
        sub_t = tests[(tests.quantity == ("N" if is_prob else key)) &
                      (tests.family == ("prob" if is_prob else "location"))]
        delta = ""
        if len(sub_t) and not np.isnan(sub_t.iloc[0]["diff"]):
            r = sub_t.iloc[0]
            unit = "" if is_prob else " dec"
            delta = (f"\n$\\Delta$({ga}−{gb}) = {r['diff']:+.2f} "
                     f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]{unit}")
        ax.set_title(title + delta, fontsize=8.5, color=DARK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_profile(tests, questions, out, ci):
    keys = [q["id"] for q in questions] + ["astro", "bio", "N"]
    labels = [base.PARAM_LABELS.get(k, (k, ""))[0] if k in {q["id"] for q in questions}
              else DERIVED_LABELS[k] for k in keys]
    ga, gb = PRIMARY
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, family, title in zip(axes, ("location", "width"),
                                 (f"Location: median of each person's distribution ({ga} − {gb})",
                                  f"Width: 10th-90th percentile range of each person ({ga} − {gb})")):
        for i, k in enumerate(keys):
            r = tests[(tests.quantity == k) & (tests.family == family)]
            if r.empty or np.isnan(r.iloc[0]["diff"]):
                continue
            r = r.iloc[0]
            ax.errorbar(i, r["diff"], yerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]],
                        fmt="o", color=DARK, capsize=4, lw=1.6, ms=6)
        ax.axhline(0, color="0.5", lw=1)
        ax.axvline(len(questions) - 0.5, color="0.7", ls=":", lw=1)
        ax.set_ylabel("difference in decades", fontsize=10, color=DARK)
        ax.set_title(f"{title}\nbars = {ci:g}% bootstrap interval over people", fontsize=10, color=DARK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[0].text(0.01, 0.97, f"negative = {ga} lower (more pessimistic) than {gb}",
                 transform=axes[0].transAxes, va="top", fontsize=9, color="0.4")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _n_axis(samples_all):
    lo = min(np.percentile(samples_all, 0.1) - 1, REF_UNIV - 1)
    hi = max(np.percentile(samples_all, 99.9) + 1, REF_MW + 1)
    return float(lo), float(hi)


def fig_pooled(answers, questions, sess_group, plot_groups, n_samples, persons,
               n_boot, n_mc_A, out, rng):
    """Pooled curves per group (+ bootstrap bands over people) and N by two methods."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.ravel()
    sess_of = {g: {s for s, gg in sess_group.items() if gg == g} for g in plot_groups}
    pools = {g: {} for g in plot_groups}

    for ax, q in zip(axes, questions):
        counts = {}
        for g in plot_groups:
            rows = answers[(answers.question_id == q["id"]) & (~answers.skipped) &
                           (answers.session_id.isin(sess_of[g]))]
            counts[g] = len(rows)
            if len(rows) == 0:
                continue
            t_grid, pdfs, pool = base.pooled_pdf(rows, q)
            pools[g][q["id"]] = (t_grid, pool)
            x = base.to_x(t_grid, q)
            ax.plot(x, pool, color=COLORS[g], lw=2, label=f"{g}")
            ax.fill_between(x, pool, color=COLORS[g], alpha=0.07)
            if len(rows) >= 3:
                bands = np.empty((n_boot, len(t_grid)))
                for b in range(n_boot):
                    m = pdfs[rng.integers(0, len(pdfs), len(pdfs))].mean(axis=0)
                    bands[b] = m / trapz(m, t_grid)
                lo, hi = np.percentile(bands, [10, 90], axis=0)
                ax.fill_between(x, lo, hi, color=COLORS[g], alpha=0.18, lw=0)
        xs = base.to_x(np.array([q["tMin"], q["tMax"]]), q)
        ax.set_xlim(xs[0], xs[1])
        sym, title = base.PARAM_LABELS.get(q["id"], (q["id"], q["id"]))
        cnt = " · ".join(f"{g} n={counts[g]}" for g in plot_groups)
        base.style_axis(ax, sym, "Prob. density per decade", f"{title}\n({cnt})",
                        logx=q.get("axisMode", "log") == "log")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, title="line = pooled, band = 10-90% bootstrap",
                   title_fontsize=7)

    # ---- N: method B (mean of individual N distributions) and method A ----
    ids = list(n_samples)
    all_s = np.concatenate([n_samples[i][:2000] for i in ids]) if ids else np.array([0.0])
    lo, hi = _n_axis(all_s)
    step = 0.05
    edges = np.arange(lo, hi + step, step)
    centers = 0.5 * (edges[1:] + edges[:-1])
    sigma_bins = 0.25 / step
    summary_rows = []

    axB, axA = axes[7], axes[8]
    ctx_B = []
    for g in plot_groups:
        pids = persons.loc[(persons.group == g) & persons.pid.isin(ids), "pid"].tolist()
        if not pids:
            continue
        sub = [n_samples[p][:1000] for p in pids]
        full = np.concatenate([n_samples[p] for p in pids])
        axB.plot(10.0 ** centers, hist_density(np.concatenate(sub), edges, sigma_bins),
                 color=COLORS[g], lw=2)
        if len(pids) >= 3:
            bb = np.empty((n_boot, len(centers)))
            for b in range(n_boot):
                pick = rng.integers(0, len(sub), len(sub))
                bb[b] = hist_density(np.concatenate([sub[i] for i in pick]), edges, sigma_bins)
            l, h = np.percentile(bb, [10, 90], axis=0)
            axB.fill_between(10.0 ** centers, l, h, color=COLORS[g], alpha=0.18, lw=0)
        ctx_B.append(f"{g} n={len(pids)}")
        summary_rows.append({"group": g, "method": "B: mean of individual N", "n": len(pids),
                             **summarize_logN(full)})

    ctx_A = []
    for g in plot_groups:
        if not all(q["id"] in pools[g] for q in questions):
            continue
        total = np.zeros(n_mc_A)
        for q in questions:
            t_grid, pool = pools[g][q["id"]]
            total += log10_of(base.sample_from_grid_pdf(t_grid, pool, n_mc_A, rng), q)
        axA.plot(10.0 ** centers, hist_density(total, edges, sigma_bins), color=COLORS[g], lw=2)
        ctx_A.append(g)
        summary_rows.append({"group": g, "method": "A: pooled marginals, independent",
                             "n": int(min(len(answers[(answers.question_id == q["id"]) & (~answers.skipped) &
                                                     (answers.session_id.isin(sess_of[g]))])
                                          for q in questions)),
                             **summarize_logN(total)})

    for ax, title in ((axB, "$N$ — mean of individual $N$ distributions\n(complete answers only; "
                            + " · ".join(ctx_B) + ")"),
                      (axA, "$N$ — pooled marginals, then multiplied independently\n(method A; "
                            "n = min over parameters)")):
        ax.set_xlim(10.0 ** lo, 10.0 ** hi)
        ax.axvline(1, color="green", ls="dashed", lw=1)
        ax.axvline(5e-13, color="red", ls="dashed", lw=1)
        base.style_axis(ax, "$N$", "Prob. density per decade", title)
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def _heat(ax, M, N, rows, cols, title, cmap):
    frac = M / np.maximum(N, 1)
    ax.imshow(frac, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]}/{N[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.6 else DARK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title(title, fontsize=9.5, color=DARK)


def fig_skips_edges(table, questions, groups, edge, out):
    cols = [base.PARAM_LABELS.get(q["id"], (q["id"], ""))[0] for q in questions]
    qids = [q["id"] for q in questions]

    def mat(col):
        return np.array([[int(table[(table.group == g) & (table.parameter == k)][col].iloc[0])
                          for k in qids] for g in groups])

    tot = mat("n_total")
    ans = mat("n_answered")
    fig, axes = plt.subplots(2, 2, figsize=(12, 2.6 * 2 + 1.2))
    _heat(axes[0, 0], mat("n_skipped"), tot, groups, cols,
          "Skipped (\"not enough knowledge\")\nfraction of people", "Oranges")
    _heat(axes[0, 1], mat("n_default_like"), ans, groups, cols,
          "Answer identical to the slider's starting position\n(possible anchoring; fraction of answers)", "Purples")
    _heat(axes[1, 0], mat("n_mode_near_floor"), ans, groups, cols,
          f"Mode within {edge:g} decade of the LOWER axis limit\n(floor effect: pessimism is censored)", "Reds")
    _heat(axes[1, 1], mat("n_mode_near_ceiling"), ans, groups, cols,
          f"Mode within {edge:g} decade of the UPPER axis limit\n(ceiling effect)", "Blues")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_beliefs(context, sess_group, groups, out):
    others = [c for c in context.columns if c not in ("session_id", "area_grande", "area_especifica")]
    cols = [c for c in base.BELIEF_TITLES if c in others] + [c for c in others if c not in base.BELIEF_TITLES]
    ctx = context.copy()
    ctx["group"] = ctx.session_id.map(sess_group)
    ctx = ctx[ctx.group.isin(groups)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    axes = axes.ravel()

    #palette = plt.cm.RdYlBu(np.linspace(0.1, 0.9, 5))[::-1]
    #colors = ['#8b0000', '#f08080', '#b0b0b0', '#90ee90', '#006400']
    #palette = colors
    palette = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 5))

    long_rows = []
    for ax, col in zip(axes, cols):
        for gi, g in enumerate(groups):
            vals = ctx.loc[ctx.group == g, col].dropna()
            counts = np.array([(vals == s).sum() for s in base.SCALE_ORDER])
            n = counts.sum()
            left = 0.0
            for k, c in enumerate(counts):
                long_rows.append({"belief": col, "group": g, "level": base.SCALE_ORDER[k], "count": int(c)})
                if c:
                    w = c / n
                    ax.barh(gi, w, left=left, color=palette[k], edgecolor="white")
                    ax.text(left + w / 2, gi, str(int(c)), ha="center", va="center", fontsize=8, color=DARK)
                    left += w
            ax.text(1.01, gi, f"n={n}", va="center", fontsize=8, color=DARK)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=9)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50%", "100%"], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(base.BELIEF_TITLES.get(col, col), fontsize=9, color=DARK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    leg = axes[len(cols)] if len(cols) < len(axes) else None
    if leg is not None:
        leg.axis("off")
        for k, lab in enumerate(base.SCALE_LABELS):
            leg.barh(-k, 1, color=palette[k], edgecolor="white")
            leg.text(1.05, -k, lab.replace("\n", " "), va="center", fontsize=9, color=DARK)
        leg.set_xlim(0, 2)
    for ax in axes[len(cols) + 1:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return pd.DataFrame(long_rows)


def belief_vs_N(context, persons):
    scale = {s: i + 1 for i, s in enumerate(base.SCALE_ORDER)}
    m = persons[["session_id", "N_q50", "N_pgt1"]].merge(context, on="session_id", how="inner")
    rows = []
    for col in [c for c in context.columns if c not in ("session_id", "area_grande", "area_especifica")]:
        x = m[col].map(scale)
        for target in ("N_q50", "N_pgt1"):
            ok = x.notna() & m[target].notna()
            if ok.sum() >= 8:
                rho, p = spearmanr(x[ok], m.loc[ok, target])
                rows.append({"belief": col, "vs": target, "n": int(ok.sum()),
                             "spearman_rho": float(rho), "p": float(p)})
            else:
                rows.append({"belief": col, "vs": target, "n": int(ok.sum()),
                             "spearman_rho": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--n-mc", type=int, default=20_000, help="Monte Carlo samples per person")
    ap.add_argument("--n-mc-A", type=int, default=200_000, help="Monte Carlo samples for method A")
    ap.add_argument("--n-boot", type=int, default=5_000, help="bootstrap resamples for the tests")
    ap.add_argument("--n-boot-curves", type=int, default=300, help="bootstrap resamples for the curve bands")
    ap.add_argument("--ci", type=float, default=90.0, help="bootstrap interval for differences (percent)")
    ap.add_argument("--min-n", type=int, default=5, help="hide groups with fewer people than this")
    ap.add_argument("--edge", type=float, default=0.5, help="decades from an axis limit counted as 'edge'")
    ap.add_argument("--include-others", action="store_true", help="also draw the 'Outras' group in the pooled figure")
    ap.add_argument("--save-person-table", action="store_true",
                    help="also write a per-person CSV (no session ids; do not commit it)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    print("Loading the data...")
    sessions, answers, context = base.load_data(data_dir)
    sess_group = {}
    for r in context.itertuples():
        g = group_of(r.area_grande)
        if g is not None:
            sess_group[r.session_id] = g
    n_done = len(sessions)
    n_noarea = n_done - len(sess_group)
    sizes = pd.Series(sess_group).value_counts()
    print(f"  completed sessions: {n_done} (without area, left out: {n_noarea})")
    for g in (G_BIO, G_EXATAS, G_OTHER):
        print(f"  {g:<11}: {int(sizes.get(g, 0))}")
    shown = [g for g in (G_BIO, G_EXATAS, G_OTHER) if sizes.get(g, 0) >= args.min_n]
    if G_BIO not in shown or G_EXATAS not in shown:
        sys.exit(f"Need at least {args.min_n} people in both {G_BIO} and {G_EXATAS}. Stopping.")
    hidden = [g for g in (G_BIO, G_EXATAS, G_OTHER) if g not in shown and sizes.get(g, 0) > 0]
    if hidden:
        print(f"  (not shown, fewer than {args.min_n} people: {', '.join(hidden)})")
    plot_groups = list(PRIMARY) + ([G_OTHER] if (args.include_others and G_OTHER in shown) else [])

    print("Per-person distributions and Monte Carlo...")
    persons, n_samples = build_person_table(answers, questions, sess_group, set(shown), args.n_mc, rng)
    n_complete = {g: int(persons[(persons.group == g)]["N_q50"].notna().sum()) for g in shown}
    print(f"  people with a complete answer set (N computable): {n_complete}")

    keys = [q["id"] for q in questions] + ["astro", "bio", "N"]
    print("Tests (this takes a few seconds)...")
    tests = compare_groups(persons, keys, args.n_boot, args.ci, rng)
    tests.to_csv(data_dir / "drake_by_area_tests.csv", index=False)

    show = tests[tests.family == "location"][["quantity", "n_a", "n_b", "median_a", "median_b",
                                              "diff", "ci_lo", "ci_hi", "cliffs_delta", "p_mwu", "p_holm"]]
    print(f"\nLocation (median of T = log10 of each person's median), {G_BIO} − {G_EXATAS}, "
          f"{args.ci:g}% bootstrap interval. Exploratory: read the intervals, not the p-values:")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    pw = tests[tests.family == "width"][["quantity", "diff", "ci_lo", "ci_hi", "p_mwu", "p_holm"]]
    print("\nWidth (q90 − q10, in decades; positive = wider for Biológicas):")
    print(pw.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    edges_tab = skips_and_edges(answers, questions, sess_group, shown, args.edge)
    edges_tab.to_csv(data_dir / "drake_by_area_skips_edges.csv", index=False)
    print("\nSkips (skipped/total) per group:")
    piv = edges_tab.assign(s=lambda d: d.n_skipped.astype(str) + "/" + d.n_total.astype(str)) \
                   .pivot(index="group", columns="parameter", values="s")[[q["id"] for q in questions]]
    print(piv.to_string())

    print("\nFigures...")
    fig_dots(persons, questions, shown, tests, out_dir / "drake_by_area_dots.png", rng)
    fig_profile(tests, questions, out_dir / "drake_by_area_profile.png", args.ci)
    nsum = fig_pooled(answers, questions, sess_group, plot_groups, n_samples, persons,
                      args.n_boot_curves, args.n_mc_A, out_dir / "drake_by_area_pooled.png", rng)
    nsum.to_csv(data_dir / "drake_by_area_N_summary.csv", index=False)
    fig_skips_edges(edges_tab, questions, shown, args.edge, out_dir / "drake_by_area_skips_edges.png")
    bel = fig_beliefs(context, sess_group, shown, out_dir / "drake_by_area_beliefs.png")
    corr = belief_vs_N(context, persons)
    corr.to_csv(data_dir / "drake_by_area_belief_corr.csv", index=False)
    bel.to_csv(data_dir / "drake_by_area_belief_counts.csv", index=False)

    print("\nN by group (median, 90% interval, probabilities):")
    cols = ["group", "method", "n", "median", "q05", "q95", "P_gt_1_pct", "P_lt_1_pct", "P_lt_5e-13_pct"]
    print(nsum[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nSpearman, belief vs. N (all groups together, only if n >= 8):")
    print(corr.dropna().to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    if args.save_person_table:
        persons.drop(columns=["session_id", "pid"]).to_csv(data_dir / "drake_by_area_persons.csv", index=False)
        print("\nWrote the per-person table. Do not commit it.")
    print(f"\nFigures in {out_dir.resolve()}, tables in {data_dir.resolve()}")


if __name__ == "__main__":
    main()
