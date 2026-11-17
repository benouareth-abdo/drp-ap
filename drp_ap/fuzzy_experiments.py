"""Fuzzy-initialization study: initializers for fuzzy c-means (FCM) and for k-means.

Metrics per run (FCM started from the initial centres): final FCM objective J_m, FCM iterations,
ARI/NMI of the arg-max partition, and the hard SSE of k-means started from the same centres.
Normalised J is J divided by the best J found on the same dataset (all methods and seeds)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import baselines as B
from .core import DRPAPSA
from .fuzzy import FuzzyDRPAPSA, fcm


def fuzzy_methods(m=2.0):
    rng_ = lambda f: (lambda X, K, s: f(X, K, np.random.default_rng(s)))
    return {
        "Random": (rng_(B.init_random), False),
        "k-means++": (rng_(B.init_kmeanspp), False),
        "Greedy k-means++": (rng_(B.init_greedy_kmeanspp), False),
        "PCA-Part": (lambda X, K, s: B.init_pca_part(X, K, split_at="mean"), True),
        "DRP-AP-SA": (lambda X, K, s: DRPAPSA().fit_init(X, K), True),
        "F-DRP-AP-SA (T=0)": (lambda X, K, s: FuzzyDRPAPSA(local_iters=0, m=m).fit_init(X, K), True),
        "F-DRP-AP-SA": (lambda X, K, s: FuzzyDRPAPSA(m=m).fit_init(X, K), True),
    }


def _fcm_best_of(f, R, m):
    """Equal-budget FCM restarts: R independent (init + FCM) runs, keep the centres of the lowest J.
    Returns (centres, J) so that the budget is counted in full FCM runs."""
    def run(X, K, seed):
        rng = np.random.default_rng(seed)
        best, bj = None, np.inf
        for _ in range(R):
            C = np.asarray(f(X, K, rng), dtype=float)
            Cf, _, J, _ = fcm(X, C, m=m, max_iter=MAX_FCM_ITER)
            if J < bj:
                best, bj = Cf, J
        return best
    return run


def restart_methods(m=2.0):
    return {
        "k-means++ best-of-3 (FCM)": (_fcm_best_of(B.init_kmeanspp, 3, m), False),
        "k-means++ best-of-10 (FCM)": (_fcm_best_of(B.init_kmeanspp, 10, m), False),
        "Greedy k-means++ best-of-3 (FCM)": (_fcm_best_of(B.init_greedy_kmeanspp, 3, m), False),
        "Greedy k-means++ best-of-10 (FCM)": (_fcm_best_of(B.init_greedy_kmeanspp, 10, m), False),
    }


MAX_FCM_ITER = 200


def run_fuzzy(datasets, m=2.0, n_seeds=20, verbose=True, with_restarts=True):
    """datasets: dict name -> (X, y, K)."""
    rows = []
    meths = fuzzy_methods(m)
    if with_restarts:
        meths.update(restart_methods(m))
    for name, (X, y, K) in datasets.items():
        for mname, (f, det) in meths.items():
            for seed in range(1 if det else n_seeds):
                t0 = time.time()
                C0 = np.asarray(f(X, K, seed), dtype=float)
                t_init = time.time() - t0
                C, U, J, it = fcm(X, C0, m=m, max_iter=MAX_FCM_ITER)
                lab = U.argmax(1)
                sse = KMeans(K, init=C0, n_init=1, algorithm="lloyd", tol=1e-4,
                             max_iter=300).fit(X).inertia_
                r = dict(dataset=name, method=mname, seed=seed, J=J, fcm_iter=it,
                         ari=adjusted_rand_score(y, lab), nmi=normalized_mutual_info_score(y, lab),
                         km_sse=sse, t_init=t_init)
                # deterministic methods are replicated across seeds for averaging
                rows += [dict(r, seed=s) for s in range(n_seeds)] if det else [r]
        if verbose:
            print(f"[fuzzy m={m}] {name} done", flush=True)
    df = pd.DataFrame(rows)
    df["nJ"] = df.J / df.groupby("dataset").J.transform("min")
    df["nSSE"] = df.km_sse / df.groupby("dataset").km_sse.transform("min")
    return df


def summarize(df):
    """Per fuzzifier: mean normalised J / hard SSE / FCM iterations / ARI, Friedman average rank
    on J, and win/tie/loss (1% margin on per-dataset mean) plus Wilcoxon-Holm of F-DRP-AP-SA
    against every other method."""
    from scipy.stats import rankdata, wilcoxon
    out = []
    for m, x in df.groupby("m"):
        pv = x.groupby(["dataset", "method"])[["nJ", "nSSE", "fcm_iter", "ari", "t_init"]].mean()
        agg = pv.groupby("method").mean()
        J = pv["nJ"].unstack()
        agg["rank_J"] = pd.DataFrame(rankdata(J.values, axis=1), index=J.index,
                                     columns=J.columns).mean()
        ref = "F-DRP-AP-SA"
        rows = []
        for b in J.columns:
            if b == ref:
                continue
            r = J[ref] / J[b]
            try:
                p = wilcoxon(J[ref], J[b]).pvalue
            except ValueError:
                p = 1.0
            rows.append(dict(method=b, win=int((r < 0.99).sum()), tie=int(((r >= .99) & (r <= 1.01)).sum()),
                             loss=int((r > 1.01).sum()), p=p))
        t = pd.DataFrame(rows).set_index("method")
        if len(t):
            order = t.p.sort_values().index
            adj, prev = {}, 0.0
            for i, k in enumerate(order):
                prev = max(prev, min(1.0, t.p[k] * (len(t) - i)))
                adj[k] = prev
            t["p_holm"] = pd.Series(adj)
        agg = agg.join(t, how="left")
        agg.insert(0, "m", m)
        out.append(agg.reset_index())
    return pd.concat(out)
