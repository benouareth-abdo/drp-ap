"""Five candidate improvements of DRP-AP-SA suggested by the greedy k-means++ best-of-10 baseline.

I1  hybrid pool:   DRP-AP-SA (or SA-opt) as anchor + (R-1) greedy k-means++ draws, keep the lowest
                   final SSE (built in experiments.best_of_pool).
I2  SA-opt:        optimal 1-D threshold (minimum two-group SSE along each candidate axis) instead
                   of the median.
I3  lookahead:     candidate splits (axes x {median, optimal threshold}) are ranked by the SSE of a
                   few global Lloyd iterations on the K'+1 clusters, not by the local gain.
I4  lookahead+D2:  I3 with extra candidates built from D^2-sampled seed pairs inside the node.
I5  halving:       successive halving over greedy k-means++ initialisations.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import pairwise_distances_argmin_min

from .core import DRPAPSA, centered_sse, kpp_indices, lloyd_in_space, spectrum


# ----------------------------------------------------------------------------
# optimal 1-D two-group split
# ----------------------------------------------------------------------------

def optimal_threshold_split(p, min_frac=0.0):
    """Indices (L, R) of the 2-group split of the 1-D array p that minimises the total within-group
    SSE (exact, O(m log m) with prefix sums). `min_frac` forbids groups smaller than that fraction."""
    m = len(p)
    o = np.argsort(p, kind="stable")
    x = p[o]
    s1, s2 = np.cumsum(x), np.cumsum(x * x)
    S1, S2 = s1[-1], s2[-1]
    j = np.arange(1, m)                     # left group = first j points
    left = s2[:-1] - s1[:-1] ** 2 / j
    right = (S2 - s2[:-1]) - (S1 - s1[:-1]) ** 2 / (m - j)
    cost = left + right
    jmin = max(1, int(np.ceil(min_frac * m)))
    ok = (j >= jmin) & (j <= m - jmin)
    if not ok.any():
        ok[:] = True
    cost = np.where(ok, cost, np.inf)
    k = int(np.argmin(cost)) + 1
    return o[:k], o[k:]


class DRPAPSAOpt(DRPAPSA):
    """I2: DRP-AP-SA with the optimal 1-D threshold along each of the top-q principal axes."""

    def __init__(self, min_frac=0.0, **kw):
        super().__init__(**kw)
        self.min_frac = min_frac

    def _split_arbitrate(self, Xc, Vt, r, rng=None):
        q = min(self.max_axes if self.max_axes else r, Vt.shape[0])
        P = Xc @ Vt[:q].T
        m = Xc.shape[0]
        best, bg, bj = None, -1.0, 0
        for j in range(q):
            L, R = optimal_threshold_split(P[:, j], self.min_frac)
            if len(L) == 0 or len(R) == 0:
                continue
            g = len(L) * len(R) / m * float(((Xc[L].mean(0) - Xc[R].mean(0)) ** 2).sum())
            if g > bg:
                best, bg, bj = (L, R), g, j
        if best is None:
            o = np.argsort(P[:, 0], kind="stable")
            best = (o[: m // 2], o[m // 2:])
        L, R = best
        self._axis = bj
        if self.local_iters > 0:
            c = np.stack([Xc[L].mean(0), Xc[R].mean(0)])
            lab = lloyd_in_space(Xc, c, self.local_iters)
            if lab is not None:
                L, R = np.where(lab == 0)[0], np.where(lab == 1)[0]
        return [L, R]


# ----------------------------------------------------------------------------
# I3 / I4: lookahead selection of the split
# ----------------------------------------------------------------------------

def _lloyd_sse(X, C, iters):
    """SSE after `iters` global Lloyd iterations started from C (empty clusters keep their centre)."""
    C = C.copy()
    for _ in range(iters):
        lab, _ = pairwise_distances_argmin_min(X, C)
        for k in range(len(C)):
            sel = lab == k
            if sel.any():
                C[k] = X[sel].mean(0)
    _, d = pairwise_distances_argmin_min(X, C)
    return float((d ** 2).sum())


class DRPAPLook:
    """Divisive initializer whose split is chosen by *lookahead*: every candidate split (top-q axes x
    {median, optimal threshold}, plus `n_d2` D^2-seeded 2-means splits) is refined by `local_iters`
    Lloyd iterations inside the node and then scored by the SSE reached after `look_iters` global
    Lloyd iterations over all current clusters (on a fixed subsample of `eval_n` points)."""

    def __init__(self, max_axes=3, local_iters=5, look_iters=2, n_d2=0, eval_n=4000, min_frac=0.0,
                 random_state=None):
        self.max_axes, self.local_iters, self.look_iters = max_axes, local_iters, look_iters
        self.n_d2, self.eval_n, self.min_frac, self.random_state = n_d2, eval_n, min_frac, random_state

    def _refine(self, Xc, L, R):
        if self.local_iters > 0:
            c = np.stack([Xc[L].mean(0), Xc[R].mean(0)])
            lab = lloyd_in_space(Xc, c, self.local_iters)
            if lab is not None:
                return np.where(lab == 0)[0], np.where(lab == 1)[0]
        return L, R

    def _candidates(self, Xn, rng):
        Xc = Xn - Xn.mean(0)
        lam, Vt = spectrum(Xc)
        q = max(1, min(self.max_axes, Vt.shape[0]))
        out = []
        for j in range(q):
            p = Xc @ Vt[j]
            o = np.argsort(p, kind="stable")
            h = max(len(p) // 2, 1)
            out.append((o[:h], o[h:]))
            out.append(optimal_threshold_split(p, self.min_frac))
        for _ in range(self.n_d2):
            seeds = kpp_indices(Xc, 2, rng)
            lab = lloyd_in_space(Xc, Xc[seeds].copy(), max(self.local_iters, 1))
            if lab is not None:
                out.append((np.where(lab == 0)[0], np.where(lab == 1)[0]))
        cands = [self._refine(Xc, L, R) for L, R in out if len(L) and len(R)]
        return cands

    def fit_init(self, X, K):
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        ev = np.arange(n) if n <= self.eval_n else np.sort(rng.choice(n, self.eval_n, replace=False))
        Xe = X[ev]
        clusters = [np.arange(n)]
        sses = [centered_sse(X - X.mean(0))]
        cents = [X.mean(0)]
        while len(clusters) < K:
            ok = [i for i, c in enumerate(clusters) if len(c) >= 2 and sses[i] > 1e-12]
            if not ok:
                break
            i = max(ok, key=lambda t: sses[t])
            idx = clusters.pop(i); sses.pop(i); cents.pop(i)
            Xn = X[idx]
            best, bv = None, np.inf
            for L, R in self._candidates(Xn, rng):
                C = np.vstack(cents + [Xn[L].mean(0), Xn[R].mean(0)])
                v = _lloyd_sse(Xe, C, self.look_iters)
                if v < bv:
                    best, bv = (L, R), v
            if best is None:
                o = np.argsort(Xn @ spectrum(Xn - Xn.mean(0))[1][0])
                best = (o[: len(o) // 2], o[len(o) // 2:])
            for part in best:
                sub = Xn[part]
                clusters.append(idx[part]); sses.append(centered_sse(sub - sub.mean(0)))
                cents.append(sub.mean(0))
        C = np.stack(cents)
        if len(C) < K:
            C = np.vstack([C, X[rng.choice(n, K - len(C), replace=n < K - len(C))]])
        return C


# ----------------------------------------------------------------------------
# I5: successive halving
# ----------------------------------------------------------------------------

def _lloyd_steps(X, C, iters):
    for _ in range(iters):
        lab, _ = pairwise_distances_argmin_min(X, C)
        C2 = C.copy()
        for k in range(len(C)):
            sel = lab == k
            if sel.any():
                C2[k] = X[sel].mean(0)
        if np.allclose(C2, C):
            break
        C = C2
    _, d = pairwise_distances_argmin_min(X, C)
    return C, float((d ** 2).sum())


def init_halving(X, K, rng, n0=20, eta=3, base_iters=2):
    """Successive halving: n0 greedy k-means++ initialisations, `base_iters` Lloyd iterations each;
    keep the best 1/eta, give them eta times more iterations, and so on until one is left.
    Returns the surviving centres (the evaluation then runs Lloyd to convergence)."""
    pool = [kmeans_plusplus(X, K, random_state=int(rng.integers(2 ** 31 - 1)))[0] for _ in range(n0)]
    iters = base_iters
    while True:
        scored = []
        for C in pool:
            C2, v = _lloyd_steps(X, C, iters)
            scored.append((v, C2))
        scored.sort(key=lambda t: t[0])
        keep = max(1, int(np.ceil(len(scored) / eta)))
        pool = [c for _, c in scored[:keep]]
        if len(pool) == 1:
            return pool[0]
        iters *= eta


# ----------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------

def improve_methods():
    from . import baselines as B
    from .experiments import _sa, best_of, best_of_pool, init_portfolio
    greedy = B.init_greedy_kmeanspp
    return {
        # references
        "PCA-Part (Su-Dy, mean split)": (lambda X, K, s: B.init_pca_part(X, K, split_at="mean"), True),
        "DRP-AP-SA": (_sa(), True),
        "Portfolio {PCA,Var,SA}": (lambda X, K, s: init_portfolio(X, K, s), True),
        "Greedy k-means++": (lambda X, K, s: greedy(X, K, np.random.default_rng(s)), False),
        "Greedy k-means++ best-of-3": (best_of(greedy, 3), False),
        "Greedy k-means++ best-of-10": (best_of(greedy, 10), False),
        # improvements
        "I1: SA + greedy best-of-2 (3 runs)": (best_of_pool(lambda X, K: DRPAPSA().fit_init(X, K), greedy, 3), False),
        "I1: SA + greedy best-of-9 (10 runs)": (best_of_pool(lambda X, K: DRPAPSA().fit_init(X, K), greedy, 10), False),
        "I2: SA-opt (optimal threshold)": (lambda X, K, s: DRPAPSAOpt().fit_init(X, K), True),
        "I1+I2: SA-opt + greedy best-of-9": (best_of_pool(lambda X, K: DRPAPSAOpt().fit_init(X, K), greedy, 10), False),
        "I3: SA lookahead": (lambda X, K, s: DRPAPLook(random_state=s).fit_init(X, K), True),
        "I4: SA lookahead + D2 candidates": (lambda X, K, s: DRPAPLook(n_d2=3, random_state=s).fit_init(X, K), False),
        "I5: successive halving (20 inits)": (lambda X, K, s: init_halving(X, K, np.random.default_rng(s)), False),
    }
