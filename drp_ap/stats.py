"""Non-parametric statistical testing (Demsar 2006): Friedman + Nemenyi, Wilcoxon + Holm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def average_ranks(pivot: pd.DataFrame) -> pd.Series:
    """pivot: datasets x methods of a 'lower is better' score -> mean rank per method."""
    return pivot.rank(axis=1, method="average").mean(0).sort_values()


def friedman(pivot: pd.DataFrame):
    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return np.nan, np.nan
    stat, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    return float(stat), float(p)


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference CD = q_alpha * sqrt(k(k+1)/(6N)) with q from the studentized range."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda t: t[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def wilcoxon_vs_ref(pivot: pd.DataFrame, ref: str, alternative: str = "two-sided") -> pd.DataFrame:
    """Paired Wilcoxon of every method vs `ref` across rows, Holm-corrected."""
    raw, eff = {}, {}
    for c in pivot.columns:
        if c == ref:
            continue
        diff = pivot[c].values - pivot[ref].values
        eff[c] = float(np.median(diff))
        if np.allclose(diff, 0):
            raw[c] = 1.0
            continue
        try:
            raw[c] = float(stats.wilcoxon(pivot[c].values, pivot[ref].values,
                                          alternative=alternative).pvalue)
        except ValueError:
            raw[c] = 1.0
    adj = holm(raw)
    return pd.DataFrame({"median_diff_vs_ref": eff, "p": raw, "p_holm": adj})


def cd_diagram(ranks: pd.Series, cd: float, path: str, title: str = ""):
    """Compact critical-difference diagram (methods within CD of each other are joined)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, vals = list(ranks.index), ranks.values
    k = len(vals)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.28 * k))
    lo, hi = 1, max(k, vals.max())
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(-k - 2.2, 1.2)
    ax.hlines(0, lo, hi, color="k", lw=1)
    for t in range(int(lo), int(hi) + 1):
        ax.vlines(t, 0, 0.12, color="k", lw=1)
        ax.text(t, 0.25, str(t), ha="center", fontsize=8)
    ax.hlines(0.9, lo, lo + cd, color="C3", lw=2)
    ax.text(lo + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", fontsize=8, color="C3")
    for i, (n, v) in enumerate(zip(names, vals)):
        y = -1 - i
        ax.hlines(y, lo - 0.3, v, color="0.6", lw=0.7)
        ax.vlines(v, y, 0, color="0.6", lw=0.7)
        ax.text(lo - 0.32, y, f"{n}  ({v:.2f})", ha="right", va="center", fontsize=8)
    y0 = -k - 0.8
    row = 0
    i = 0
    while i < k:  # maximal cliques of consecutive methods
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            ax.hlines(y0 - 0.25 * row, vals[i], vals[j], color="k", lw=3)
            row += 1
        i = j + 1 if j > i else i + 1
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
