"""Measured directional rank on real datasets and diagnostics of where DRP-AP-SA helps or hurts.

For every dataset: spectrum of the root cluster, mean r_delta and lambda2/lambda1 along the splits
actually made, fraction of splits whose chosen axis is not the first principal axis, and the
initial and final SSE of PCA-Part and DRP-AP-SA after one Lloyd run.  `correlations` relates these
to the final-SSE gain of DRP-AP-SA over PCA-Part (exploratory with few datasets: report effect sizes,
not p-values)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import baselines as B
from . import metrics as M
from .core import DRPAPSA, directional_rank, spectrum
from .experiments import _final_sse


def diagnose(X, K, delta=0.05):
    Xc = X - X.mean(0)
    lam, _ = spectrum(Xc)
    m = DRPAPSA()
    C_sa = m.fit_init(X, K)
    axes = [t["axis"] for t in m.trace_]
    C_pca = B.init_pca_part(X, K)
    f_sa, f_pca = _final_sse(X, K, C_sa), _final_sse(X, K, C_pca)
    i_sa, i_pca = M.sse(X, C_sa), M.sse(X, C_pca)
    return dict(n=len(X), d=X.shape[1], K=K,
                root_r=directional_rank(lam, delta),
                root_ratio=float(lam[1] / lam[0]) if len(lam) > 1 else 0.0,
                mean_r=float(np.mean([t["r"] for t in m.trace_])) if m.trace_ else np.nan,
                mean_ratio=float(np.mean([t["ratio"] for t in m.trace_])) if m.trace_ else np.nan,
                frac_not_u1=float(np.mean([a != 0 for a in axes])) if axes else 0.0,
                init_ratio=i_sa / i_pca, final_ratio=f_sa / f_pca,
                gain_pct=100.0 * (1.0 - f_sa / f_pca))


def run_diagnostics(datasets, delta=0.05):
    rows = [dict(dataset=n, **diagnose(X, K, delta)) for n, (X, y, K) in datasets.items()]
    return pd.DataFrame(rows)


def correlations(df):
    """Spearman correlations (rho, p) of the gain with the measured quantities."""
    out = {}
    if len(df) < 4:
        return out
    for c in ("root_ratio", "root_r", "mean_r", "mean_ratio", "frac_not_u1", "K", "n", "d"):
        if df[c].nunique() > 1:
            rho, p = spearmanr(df[c], df["gain_pct"])
            out[c] = (float(rho), float(p))
    # does the initial-SSE improvement predict the final-SSE improvement?
    if df["init_ratio"].nunique() > 1:
        rho, p = spearmanr(1 - df["init_ratio"], df["gain_pct"])
        out["init_gain_vs_final_gain"] = (float(rho), float(p))
    return out
