"""Baseline k-means initializers (protocol of the paper).

Every function has signature  f(X, K, rng) -> centroids (K, d)  with rng a numpy Generator,
except deterministic ones which ignore rng.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import pairwise_distances_argmin_min

from .core import centered_sse, median_split, spectrum


def _seed(rng) -> int:
    return int(rng.integers(0, 2 ** 31 - 1))


# --- random / k-means++ family ------------------------------------------------

def init_random(X, K, rng):
    return X[rng.choice(len(X), K, replace=len(X) < K)].astype(float)


def init_kmeanspp(X, K, rng):
    return kmeans_plusplus(X, K, random_state=_seed(rng), n_local_trials=1)[0]


def init_greedy_kmeanspp(X, K, rng):
    """scikit-learn default: best of 2 + floor(log K) candidates per step."""
    return kmeans_plusplus(X, K, random_state=_seed(rng))[0]


# --- deterministic ------------------------------------------------------------

def init_kkz(X, K, rng=None):
    mu = X.mean(0)
    i0 = int(np.argmax(((X - mu) ** 2).sum(1)))
    chosen = [i0]
    dmin = ((X - X[i0]) ** 2).sum(1)
    for _ in range(1, K):
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, ((X - X[i]) ** 2).sum(1))
    return X[chosen].astype(float)


def _lloyd2(sub, c0, T):
    """T Lloyd iterations of 2-means in the full space, starting from centres c0 (2,d)."""
    c = c0.copy()
    lab = None
    for _ in range(T):
        lab = ((sub[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        if lab.min() == lab.max():
            return None
        c = np.stack([sub[lab == 0].mean(0), sub[lab == 1].mean(0)])
    return lab


def _divisive(X, K, direction, split_at="median", local_iters=0):
    n = len(X)
    Xc0 = X - X.mean(0)
    clusters = [(np.arange(n), centered_sse(Xc0))]
    while len(clusters) < K:
        cand = [(i, c) for i, c in enumerate(clusters) if len(c[0]) >= 2 and c[1] > 1e-12]
        if not cand:
            break
        i = max(cand, key=lambda t: t[1][1])[0]
        idx, _ = clusters.pop(i)
        sub = X[idx]
        Xc = sub - sub.mean(0)
        proj = Xc @ direction(Xc)
        L, R = median_split(proj, split_at)
        if local_iters > 0:
            c0 = np.stack([sub[L].mean(0), sub[R].mean(0)])
            lab = _lloyd2(sub, c0, local_iters)
            if lab is not None:
                L, R = np.where(lab == 0)[0], np.where(lab == 1)[0]
        for p in (L, R):
            s = X[idx[p]]
            clusters.append((idx[p], centered_sse(s - s.mean(0))))
    cents = np.stack([X[c[0]].mean(0) for c in clusters])
    if len(cents) < K:
        cents = np.vstack([cents, X[np.random.default_rng(0).choice(n, K - len(cents))]])
    return cents


def init_pca_part(X, K, rng=None, split_at="median", local_iters=0):
    """PCA-Part (Su & Dy 2004): split the largest-SSE cluster along its first PC."""
    return _divisive(X, K, lambda Xc: spectrum(Xc)[1][0], split_at, local_iters)


def init_var_part(X, K, rng=None, split_at="median"):
    """Var-Part (Su & Dy 2007): split along the coordinate of largest variance."""
    def direction(Xc):
        e = np.zeros(Xc.shape[1])
        e[int(np.argmax(Xc.var(0)))] = 1.0
        return e
    return _divisive(X, K, direction, split_at)


# --- Bradley & Fayyad refinement ---------------------------------------------

def init_bradley_fayyad(X, K, rng, J=10, frac=0.1):
    n = len(X)
    m = int(min(n, max(frac * n, 5 * K)))
    sols = []
    for _ in range(J):
        sub = X[rng.choice(n, m, replace=False)]
        km = KMeans(K, init="random", n_init=1, max_iter=50, random_state=_seed(rng)).fit(sub)
        sols.append(km.cluster_centers_)
    CM = np.vstack(sols)
    best, best_in = None, np.inf
    for s in sols:
        km = KMeans(K, init=s, n_init=1, max_iter=50).fit(CM)
        if km.inertia_ < best_in:
            best, best_in = km.cluster_centers_, km.inertia_
    return best


# --- Mahalanobis seeding (K-means++ MD and variants) ---------------------------

def whitening(X, eps_rel=1e-6):
    """Global whitening W so that Euclidean distance in (X-mu) @ W is the Mahalanobis distance.

    The ad hoc isotropic term eps*I is the standard patch for singular covariances.
    """
    mu = X.mean(0)
    C = np.cov(X, rowvar=False, bias=True)
    C = np.atleast_2d(C)
    lbar = float(np.trace(C)) / C.shape[0]
    w, V = np.linalg.eigh(C + eps_rel * lbar * np.eye(C.shape[0]))
    w = np.maximum(w, 1e-300)
    return mu, V / np.sqrt(w)


def init_kmeanspp_md(X, K, rng, eps_rel=1e-6):
    mu, W = whitening(X, eps_rel)
    Z = (X - mu) @ W
    idx = kmeans_plusplus(Z, K, random_state=_seed(rng), n_local_trials=1)[1]
    return X[idx].astype(float)


def init_kmeanspp_md_best(X, K, rng, R=10, eps_rel=1e-6):
    """Best of R Mahalanobis seedings according to Mahalanobis inertia."""
    mu, W = whitening(X, eps_rel)
    Z = (X - mu) @ W
    best, best_in = None, np.inf
    for _ in range(R):
        idx = kmeans_plusplus(Z, K, random_state=_seed(rng), n_local_trials=1)[1]
        _, dmin = pairwise_distances_argmin_min(Z, Z[idx])
        inertia = float((dmin ** 2).sum())
        if inertia < best_in:
            best, best_in = idx, inertia
    return X[best].astype(float)


def init_kmeanspp_md_hybrid(X, K, rng, eps_rel=1e-6):
    """Hybrid metric: D^2 sampling under the average of the (trace-normalised) Euclidean and
    Mahalanobis squared distances.  NOTE: this is our operationalisation of 'Hybrid' - adapt it
    if the definition in the reference differs."""
    mu, W = whitening(X, eps_rel)
    Zm = (X - mu) @ W
    tr = float(((X - mu) ** 2).sum(1).mean())
    d = X.shape[1]
    Zh = np.hstack([(X - mu) / np.sqrt(2 * max(tr, 1e-300)), Zm / np.sqrt(2 * d)])
    idx = kmeans_plusplus(Zh, K, random_state=_seed(rng), n_local_trials=1)[1]
    return X[idx].astype(float)


# --- Model-based reference: GMM means as initial centroids ---------------------

def init_gmm(X, K, rng, max_iter=20):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    for reg in (1e-6, 1e-3, 1e-1):
        try:
            with warnings.catch_warnings():       # EM is deliberately capped (max_iter)
                warnings.simplefilter("ignore", ConvergenceWarning)
                g = GaussianMixture(K, covariance_type="full", init_params="k-means++",
                                    reg_covar=reg * float(X.var(0).mean()), max_iter=max_iter,
                                    random_state=_seed(rng)).fit(X)
            return g.means_
        except Exception:
            continue
    return init_kmeanspp(X, K, rng)
