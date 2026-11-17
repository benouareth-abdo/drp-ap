"""Held-out choice of the DRP-AP-SA hyper-parameters (T, q_max) by leave-one-dataset-out (LODO).

For every dataset d, the configuration with the lowest mean normalised SSE over the OTHER datasets
is selected and evaluated on d. The held-out score therefore never uses d for choosing (T, q_max),
which removes the objection that the default was tuned on the evaluation benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import baselines as B
from .core import DRPAPSA
from .experiments import evaluate

DEFAULT_TS = (0, 2, 5, 10)
DEFAULT_QS = (1, 2, 3, 5)


def cfg_name(T, q):
    return f"SA[T={T},q={q}]"


def grid_methods(Ts=DEFAULT_TS, qs=DEFAULT_QS):
    m = {cfg_name(T, q): ((lambda X, K, s, T=T, q=q:
                           DRPAPSA(local_iters=T, max_axes=q).fit_init(X, K)), True)
         for T in Ts for q in qs}
    m["PCA-Part"] = (lambda X, K, s: B.init_pca_part(X, K), True)
    return m


def run_grid(datasets, Ts=DEFAULT_TS, qs=DEFAULT_QS, verbose=False):
    """Deterministic configurations: one run each. Returns long df (dataset, method, sse)."""
    rows = []
    for name, (X, y, K) in datasets.items():
        rows += evaluate(name, X, y, K, grid_methods(Ts, qs), 1, verbose=verbose, with_sil=False)
    df = pd.DataFrame(rows)
    df["nsse"] = df["sse"] / df.groupby("dataset")["sse"].transform("min")
    return df


def lodo(df, default=cfg_name(5, 3), pool=None, ref="PCA-Part"):
    """df: columns dataset, method, nsse (e.g. from run_grid or from per-method means).
    Returns (per-dataset table, summary dict)."""
    piv = df.groupby(["dataset", "method"])["nsse"].mean().unstack()
    pool = pool or [c for c in piv.columns if c != ref]
    pool = [c for c in pool if c in piv.columns]
    rows = []
    for d in piv.index:
        others = piv.drop(index=d)
        # re-normalise on the other datasets only (their own best), then rank by mean
        pick = others[pool].mean().idxmin()
        rows.append(dict(dataset=d, selected=pick, lodo=piv.loc[d, pick],
                         fixed_default=piv.loc[d, default] if default in piv else np.nan,
                         reference=piv.loc[d, ref] if ref in piv else np.nan,
                         oracle=piv.loc[d, pool].min()))
    t = pd.DataFrame(rows).set_index("dataset")
    summ = {c: float(t[c].mean()) for c in ("lodo", "fixed_default", "reference", "oracle")}
    summ["global_best_in_sample"] = piv[pool].mean().idxmin()
    for c in ("fixed_default", "reference"):
        diff = t["lodo"] - t[c]
        if np.any(diff != 0) and len(diff) >= 6:
            summ[f"wilcoxon_p_lodo_vs_{c}"] = float(wilcoxon(diff).pvalue)
    return t, summ
