"""Experiment runners implementing the protocol of the paper."""
from __future__ import annotations

import itertools
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import baselines as B
from . import metrics as M
from .core import DRPAP, DRPAPSA, DRPAPSAR
from .synthetic import FAMILIES

TOL, MAX_ITER = 1e-4, 300


# ----------------------------------------------------------------------------
# Method registry:  name -> (callable(X, K, seed) -> centroids, deterministic?)
# ----------------------------------------------------------------------------

def _rngf(f):
    return lambda X, K, seed: f(X, K, np.random.default_rng(seed))


def _drp(**kw):
    return lambda X, K, seed: DRPAP(random_state=seed, **kw).fit_init(X, K)


def _sa(**kw):
    """DRP-AP-SA with the recommended defaults (T=5, max_axes=3) unless overridden."""
    return lambda X, K, seed: DRPAPSA(random_state=seed, **kw).fit_init(X, K)


def _final_sse(X, K, C):
    return KMeans(K, init=C, n_init=1, max_iter=MAX_ITER, tol=TOL, algorithm="lloyd",
                  random_state=0).fit(X).inertia_


def best_of(f, R):
    """Equal-compute restart baseline: R independent (init + Lloyd) runs of the stochastic
    initialiser f(X, K, rng); returns the initial centroids with the lowest final SSE.
    The reported init time therefore includes the R Lloyd runs."""
    def run(X, K, seed):
        rng = np.random.default_rng(seed)
        best, best_in = None, np.inf
        for _ in range(R):
            C = np.asarray(f(X, K, rng), dtype=float)
            v = _final_sse(X, K, C)
            if v < best_in:
                best, best_in = C, v
        return best
    return run


def best_of_pool(anchor, f_rand, R):
    """Pool of R Lloyd runs: the deterministic `anchor(X, K)` first, then R-1 draws of the
    randomised initialiser f_rand(X, K, rng); keep the lowest final SSE (equal compute to best_of(R))."""
    def run(X, K, seed):
        rng = np.random.default_rng(seed)
        best = np.asarray(anchor(X, K), dtype=float)
        best_in = _final_sse(X, K, best)
        for _ in range(R - 1):
            C = np.asarray(f_rand(X, K, rng), dtype=float)
            v = _final_sse(X, K, C)
            if v < best_in:
                best, best_in = C, v
        return best
    return run


def _sar(X, K, rng):
    return DRPAPSAR(random_state=int(rng.integers(2 ** 31 - 1))).fit_init(X, K)


def init_portfolio(X, K, seed=0):
    """Run Lloyd from several deterministic initialisations; return the initial centroids whose
    final SSE is lowest (selection uses only the unsupervised objective)."""
    cands = [B.init_pca_part(X, K), B.init_var_part(X, K),
             DRPAPSA(random_state=seed).fit_init(X, K)]   # SA (T=5, q<=3)
    best, best_in = None, np.inf
    for C in cands:
        v = _final_sse(X, K, C)
        if v < best_in:
            best, best_in = C, v
    return best


def main_methods():
    return {
        "Random": (_rngf(B.init_random), False),
        "k-means++": (_rngf(B.init_kmeanspp), False),
        "Greedy k-means++": (_rngf(B.init_greedy_kmeanspp), False),
        "KKZ": (lambda X, K, s: B.init_kkz(X, K), True),
        "PCA-Part": (lambda X, K, s: B.init_pca_part(X, K), True),
        "Var-Part": (lambda X, K, s: B.init_var_part(X, K), True),
        "Bradley-Fayyad": (_rngf(B.init_bradley_fayyad), False),
        "k-means++ MD": (_rngf(B.init_kmeanspp_md), False),
        "k-means++ MD (best of 10)": (_rngf(B.init_kmeanspp_md_best), False),
        "k-means++ MD Hybrid": (_rngf(B.init_kmeanspp_md_hybrid), False),
        "GMM-init": (_rngf(B.init_gmm), False),
        "DRP-AP": (_drp(), False),
        "DRP-AP+": (_drp(adaptive_k=True), False),
        # recommended variant: deterministic SSE-arbitrated splits, per-split Lloyd T=5, top-3 axes
        "DRP-AP-SA": (_sa(), True),
        # faithful Su & Dy baselines: the original PCA-Part / Var-Part split at the MEAN of the
        # projection (hyperplane through the centroid); "PCA-Part"/"Var-Part" above split at the median
        "PCA-Part (Su-Dy, mean split)": (lambda X, K, s: B.init_pca_part(X, K, split_at="mean"), True),
        "Var-Part (Su-Dy, mean split)": (lambda X, K, s: B.init_var_part(X, K, split_at="mean"), True),
        "DRP-AP-SA (mean split)": (_sa(split_at="mean"), True),
        # cheaper variant without per-split refinement (== PCA-Part on real data in results4)
        "DRP-AP-SA (T=0)": (_sa(local_iters=0), True),
        # A3 as a first-class baseline: PCA-Part with the same local refinement
        "PCA-Part + local Lloyd (A3)": (lambda X, K, s: B.init_pca_part(X, K, local_iters=5), True),
        # equal-compute restart baselines (R full init+Lloyd runs, keep the best)
        "Greedy k-means++ best-of-3": (best_of(B.init_greedy_kmeanspp, 3), False),
        "Greedy k-means++ best-of-10": (best_of(B.init_greedy_kmeanspp, 10), False),
        # randomised DRP-AP-SA-R restarts and a pool anchored on the deterministic DRP-AP-SA
        "SA-R best-of-10": (best_of(_sar, 10), False),
        "SA pool-3": (best_of_pool(lambda X, K: DRPAPSA().fit_init(X, K), _sar, 3), False),
        "SA pool-10": (best_of_pool(lambda X, K: DRPAPSA().fit_init(X, K), _sar, 10), False),
        # deterministic portfolio: Lloyd from {PCA-Part, Var-Part, SA(T=5)}, keep the lowest final SSE
        # (costs 3 Lloyd runs; the init time reported includes them)
        "Portfolio {PCA,Var,SA}": (lambda X, K, s: init_portfolio(X, K, s), True),
    }


def ablation_methods():
    return {
        "DRP-AP (full)": (_drp(), False),
        "A1: delta=1 (=PCA-Part)": (_drp(delta=1.0), False),
        "A2: no shrinkage": (_drp(shrinkage="none"), False),
        "A2: fixed eps*I": (_drp(shrinkage="eps"), False),
        "A2: Ledoit-Wolf": (_drp(shrinkage="lw"), False),
        "A2: OAS": (_drp(shrinkage="oas"), False),
        "A3: PCA-Part + local Lloyd": (lambda X, K, s: B.init_pca_part(X, K, local_iters=5), True),
        "A4: random mode": (_drp(mode_policy="random"), False),
        "A5: Euclid in B/C": (_drp(metric="euclid"), False),
        "A6: full-space metric in C": (_drp(space="full"), False),
        "A7: rho-gate (rho=0.5)": (_drp(rho=0.5), False),
        "A8: SA (T=5, all axes)": (_drp(mode_policy="arbitrate"), True),
        "A8: SA (T=0, all axes)": (_drp(mode_policy="arbitrate", local_iters=0), True),
        "A8: SA (T=5, q<=3)": (_sa(), True),
        "A8: SA (T=5, q<=10)": (_sa(local_iters=5, max_axes=10), True),
        "A8: SA (T=0, q<=3)": (_sa(local_iters=0), True),
        "PCA-Part": (lambda X, K, s: B.init_pca_part(X, K), True),
    }


# ----------------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------------

def run_one(fn, X, y, K, seed, with_sil=True):
    t0 = time.perf_counter()
    C = np.asarray(fn(X, K, seed), dtype=float)
    t_init = time.perf_counter() - t0
    init_sse = M.sse(X, C)
    km = KMeans(K, init=C, n_init=1, max_iter=MAX_ITER, tol=TOL, algorithm="lloyd",
                random_state=0).fit(X)
    ari, nmi = M.external_scores(y, km.labels_)
    return dict(init_sse=init_sse, sse=float(km.inertia_), n_iter=int(km.n_iter_), ari=ari,
                nmi=nmi, sil=M.silhouette(X, km.labels_) if with_sil else float('nan'),
                t_init=t_init)


def evaluate(name, X, y, K, methods, n_seeds, verbose=True, with_sil=True):
    rows = []
    for mname, (fn, det) in methods.items():
        cache = None
        for s in range(n_seeds):
            try:
                if det and cache is not None:
                    r = dict(cache)
                else:
                    r = run_one(fn, X, y, K, s, with_sil)
                    cache = r
            except Exception as e:  # keep the benchmark running
                warnings.warn(f"{mname} failed on {name} (seed {s}): {e}")
                break
            rows.append(dict(dataset=name, method=mname, seed=s, K=K, **r))
        if verbose and rows and rows[-1]["method"] == mname:
            sub = [r["sse"] for r in rows if r["method"] == mname]
            print(f"  {name:9s} {mname:28s} SSE={np.mean(sub):.5g}")
    return rows


def run_real(datasets, methods, n_seeds=30, verbose=True):
    rows = []
    for name, (X, y, K) in datasets.items():
        if verbose:
            print(f"[real] {name}  (n={len(X)}, d={X.shape[1]}, K={K})")
        rows += evaluate(name, X, y, K, methods, n_seeds, verbose)
    return add_normalised(pd.DataFrame(rows))


def add_normalised(df, group="dataset"):
    """nsse = SSE / best SSE on the dataset (over all methods and seeds);  fail = nsse > 1.5."""
    best = df.groupby(group)["sse"].transform("min")
    df = df.copy()
    df["nsse"] = df["sse"] / best
    df["fail"] = (df["nsse"] > 1.5).astype(float)       # paper's threshold (almost never triggers)
    df["fail2"] = (df["nsse"] > 1.02).astype(float)     # stricter: >2% above the best SSE
    return df


# ----------------------------------------------------------------------------
# Seed stability (bootstrap resamples)
# ----------------------------------------------------------------------------

def run_stability(datasets, methods, n_boot=15, max_n=4000, verbose=True):
    rows = []
    for name, (X, y, K) in datasets.items():
        rng = np.random.default_rng(0)
        if len(X) > max_n:
            X = X[rng.choice(len(X), max_n, replace=False)]
        diam = M.diameter(X)
        boots = [X[rng.choice(len(X), len(X), replace=True)] for _ in range(n_boot)]
        for mname, (fn, det) in methods.items():
            try:
                cs = [np.asarray(fn(Xb, K, 0), float) for Xb in boots]
                rows.append(dict(dataset=name, method=mname, stability=M.seed_stability(cs, diam)))
            except Exception as e:
                warnings.warn(f"stability {mname}/{name}: {e}")
        if verbose:
            print(f"[stability] {name} done")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Mode usage (Prediction P6)
# ----------------------------------------------------------------------------

def run_modes(datasets, seeds=(0, 1, 2), **drp_kw):
    rows = []
    for name, (X, y, K) in datasets.items():
        for s in seeds:
            m = DRPAP(random_state=s, **drp_kw)
            m.fit_init(X, K)
            c = m.mode_counts_
            tot = max(sum(c.values()), 1)
            rows.append(dict(dataset=name, seed=s, A=c["A"] / tot, B=c["B"] / tot,
                             C=c["C"] / tot, frac_BC=(c["B"] + c["C"]) / tot,
                             mean_rank=np.mean([t["r"] for t in m.trace_]) if m.trace_ else np.nan))
    return pd.DataFrame(rows).groupby("dataset").mean(numeric_only=True).drop(columns="seed").reset_index()


# ----------------------------------------------------------------------------
# Synthetic experiments (Predictions P1-P3)
# ----------------------------------------------------------------------------

SYN_METHODS = ["k-means++", "PCA-Part", "Var-Part", "k-means++ MD", "DRP-AP", "DRP-AP-SA",
               "DRP-AP-SA (T=0)"]
# second synthetic suite: generators that are NOT the Theorem-2 construction
SYN2_METHODS = ["k-means++", "Greedy k-means++", "PCA-Part", "PCA-Part (Su-Dy, mean split)",
                "PCA-Part + local Lloyd (A3)",
                "DRP-AP-SA", "DRP-AP-SA (T=0)", "Greedy k-means++ best-of-3", "SA pool-3"]


def synthetic2_grid(quick=False):
    g = []
    for r in (2, 3) if quick else (2, 3, 5):
        for decay in (1.0, 0.7, 0.4):
            g.append(("subspace_gmm", dict(r=r, decay=decay, K=8)))
    for overlap in (1.0, 2.0) if quick else (0.7, 1.0, 2.0):
        g.append(("random_gmm", dict(overlap=overlap)))
    g.append(("student", {}))
    for K in (6, 12):
        g.append(("ring", dict(K=K)))
    return g


def run_synthetic2(n_draws=30, quick=False, verbose=True, methods=None):
    """Same protocol as run_synthetic on generators unrelated to the Theorem-2 construction."""
    reg = main_methods()
    methods = {k: reg[k] for k in (methods or SYN2_METHODS)}
    rows = []
    for fam, params in synthetic2_grid(quick):
        cfg = ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                       for k, v in params.items())
        for draw in range(n_draws):
            X, y, K = FAMILIES[fam](seed=draw, **params)
            rows += [dict(family=fam, config=cfg, draw=draw, **r)
                     for r in evaluate(f"{fam}[{cfg}]", X, y, K, methods, 1, verbose=False,
                                      with_sil=False)]
        if verbose:
            print(f"[synthetic2] {fam} {cfg} done")
    df = pd.DataFrame(rows)
    best = df.groupby(["family", "config", "draw"])["sse"].transform("min")
    df["nsse"] = df["sse"] / best
    return df


def synthetic_grid(quick=False):
    g = []
    g.append(("mono", dict(d=10, K=4)))
    for r in (2, 3) if quick else (2, 3, 5):
        for ratio in (0.2, 0.5, 0.8, 0.95):
            g.append(("multi_axial", dict(r=r, ratio=ratio)))
    for kappa in (1e2, 1e4) if quick else (1e2, 1e4, 1e6):
        g.append(("anisotropic", dict(kappa=kappa)))
    g.append(("shells", {}))
    for d in (20, 50) if quick else (20, 50, 100):
        for mult in (1, 2, 10):
            g.append(("highdim", dict(d=d, n=d * mult if mult > 1 else int(d * 1.2))))
    return g


def run_synthetic(n_draws=30, quick=False, verbose=True, methods=None):
    reg = main_methods()
    methods = {k: reg[k] for k in (methods or SYN_METHODS)}
    rows = []
    for fam, params in synthetic_grid(quick):
        cfg = ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                       for k, v in params.items())
        for draw in range(n_draws):
            X, y, K = FAMILIES[fam](seed=draw, **params)
            rows += [dict(family=fam, config=cfg, draw=draw, **r)
                     for r in evaluate(f"{fam}[{cfg}]", X, y, K, methods, 1, verbose=False,
                                      with_sil=False)]
        if verbose:
            print(f"[synthetic] {fam} {cfg} done")
    df = pd.DataFrame(rows)
    best = df.groupby(["family", "config", "draw"])["sse"].transform("min")
    df["nsse"] = df["sse"] / best
    return df


# ----------------------------------------------------------------------------
# Ablations and sensitivity
# ----------------------------------------------------------------------------

def run_ablation(datasets, n_seeds=10, verbose=True):
    rows = []
    for name, (X, y, K) in datasets.items():
        if verbose:
            print(f"[ablation] {name}")
        rows += evaluate(name, X, y, K, ablation_methods(), n_seeds, verbose=False,
                         with_sil=False)
    df = pd.DataFrame(rows)
    ref = df[df.method == "DRP-AP (full)"].groupby("dataset")["sse"].mean()
    df["rel_sse"] = df["sse"] / df["dataset"].map(ref)
    return df


def run_sensitivity(datasets, n_seeds=5, deltas=(0.01, 0.05, 0.1, 0.2),
                    kappas=(1e2, 1e3, 1e4, 1e6), Ts=(0, 5, 10), verbose=True):
    rows = []
    for name, (X, y, K) in datasets.items():
        base = None
        for delta, kappa0, T in itertools.product(deltas, kappas, Ts):
            fn = _drp(delta=delta, kappa0=kappa0, local_iters=T)
            for s in range(n_seeds):
                r = run_one(fn, X, y, K, s, with_sil=False)
                m = DRPAP(delta=delta, kappa0=kappa0, local_iters=T, random_state=s)
                m.fit_init(X, K)
                c = m.mode_counts_
                rows.append(dict(dataset=name, delta=delta, kappa0=kappa0, T=T, seed=s,
                                 sse=r["sse"], frac_B=c["B"] / max(sum(c.values()), 1),
                                 frac_C=c["C"] / max(sum(c.values()), 1)))
        if verbose:
            print(f"[sensitivity] {name} done")
    df = pd.DataFrame(rows)
    df["nsse"] = df["sse"] / df.groupby("dataset")["sse"].transform("min")
    return df
