"""Synthetic generators with controlled directional rank (Section 'Synthetic generators').

All generators return (X, y, K) and apply a random rotation so that no method benefits from
axis-aligned structure (Var-Part would otherwise be advantaged).
"""
from __future__ import annotations

import numpy as np


def _rotation(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def _assign(n, K, rng):
    y = np.arange(n) % K
    rng.shuffle(y)
    return y


def mono_directional(n=1500, d=10, K=4, sep=4.0, rng=None, seed=0):
    """Clusters separated along a single direction (regime of Theorem 1: r_delta = 1)."""
    rng = rng or np.random.default_rng(seed)
    y = _assign(n, K, rng)
    means = np.zeros((K, d))
    means[:, 0] = sep * (np.arange(K) - (K - 1) / 2) * 1.0
    X = means[y] + rng.standard_normal((n, d)) * 0.5
    return X @ _rotation(d, rng), y, K


def multi_axial(n=2000, d=20, r=2, ratio=0.6, sigma1=1.0, sigma2=0.3, noise=0.05,
                rng=None, seed=0):
    """Theorem-2 construction: K = 2^r clusters at hypercube vertices +-a on axes 2..r+1, each
    cluster elongated along axis 1.  Total spectrum: lam_1 = sigma1^2,
    lam_2..lam_{r+1} = sigma2^2 + a^2 = ratio * sigma1^2  (so lambda_2/lambda_1 = ratio).
    Remaining d-r-1 axes carry tiny variance `noise`^2 (ill-conditioning)."""
    rng = rng or np.random.default_rng(seed)
    d = max(d, r + 2)
    a = np.sqrt(max(ratio * sigma1 ** 2 - sigma2 ** 2, 1e-6))
    K = 2 ** r
    y = _assign(n, K, rng)
    signs = np.array([[1 if (k >> j) & 1 else -1 for j in range(r)] for k in range(K)], float)
    means = np.zeros((K, d))
    means[:, 1:r + 1] = a * signs
    std = np.full(d, noise)
    std[0] = sigma1
    std[1:r + 1] = sigma2
    X = means[y] + rng.standard_normal((n, d)) * std
    return X @ _rotation(d, rng), y, K


def anisotropic(n=2000, d=20, K=5, kappa=1e4, spread=6.0, rng=None, seed=0):
    """Gaussian clusters whose covariances have eigenvalues log-spaced in [1/kappa, 1],
    each with its own random orientation (ill-conditioned Mahalanobis metric)."""
    rng = rng or np.random.default_rng(seed)
    y = _assign(n, K, rng)
    ev = np.logspace(0, -np.log10(kappa), d)
    centers = rng.standard_normal((K, d)) * spread / np.sqrt(d) * 2
    X = np.empty((n, d))
    for k in range(K):
        sel = y == k
        Q = _rotation(d, rng)
        X[sel] = centers[k] + (rng.standard_normal((sel.sum(), d)) * np.sqrt(ev)) @ Q.T
    return X, y, K


def concentric_shells(n=1500, d=2, K=3, noise=0.05, rng=None, seed=0):
    """Concentric shells (sanity check; SSE-based methods are not expected to recover them)."""
    rng = rng or np.random.default_rng(seed)
    y = _assign(n, K, rng)
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    X = g * (1 + 2 * y)[:, None] + rng.standard_normal((n, d)) * noise
    return X, y, K


def high_dim_mixture(n=200, d=50, K=5, sep=3.0, rng=None, seed=0):
    """High-dimensional mixture with n comparable to d (small-sample conditioning regime):
    low-rank cluster covariances plus isotropic noise."""
    rng = rng or np.random.default_rng(seed)
    y = _assign(n, K, rng)
    centers = rng.standard_normal((K, d)) * sep / np.sqrt(d) * np.sqrt(d) / 3
    X = np.empty((n, d))
    for k in range(K):
        sel = y == k
        B = rng.standard_normal((d, 3)) * 1.5
        X[sel] = (centers[k] + rng.standard_normal((sel.sum(), 3)) @ B.T
                  + rng.standard_normal((sel.sum(), d)) * 0.3)
    return X, y, K


# ---------------------------------------------------------------------------
# Generators that are NOT the Theorem-2 construction (non-circular evaluation)
# ---------------------------------------------------------------------------

def subspace_gmm(n=2000, d=20, K=8, r=3, decay=0.7, noise=0.5, rng=None, seed=0):
    """Gaussian clusters whose centres are random in an r-dim subspace with geometrically decaying
    between-cluster spread (decay=1: equal spread on all r axes => high directional rank).
    No hypercube structure, random K, isotropic within-cluster noise."""
    rng = rng or np.random.default_rng(seed)
    d = max(d, r + 1)
    y = _assign(n, K, rng)
    scale = decay ** np.arange(r)
    centers = np.zeros((K, d))
    centers[:, :r] = rng.standard_normal((K, r)) * scale * 3.0
    X = centers[y] + rng.standard_normal((n, d)) * noise
    return X @ _rotation(d, rng), y, K


def random_gmm(n=2000, d=20, K=6, overlap=1.0, rng=None, seed=0):
    """Random-covariance Gaussian mixture: every cluster has its own random rotation, random
    spectrum decay exponent and random scale; centres are N(0, I) scaled by 1/overlap."""
    rng = rng or np.random.default_rng(seed)
    y = _assign(n, K, rng)
    centers = rng.standard_normal((K, d)) * 2.0 / overlap
    X = np.empty((n, d))
    for k in range(K):
        sel = y == k
        expo = rng.uniform(0.3, 2.0)
        ev = (np.arange(1, d + 1) ** -expo) * rng.uniform(0.5, 2.0)
        Q = _rotation(d, rng)
        X[sel] = centers[k] + (rng.standard_normal((sel.sum(), d)) * np.sqrt(ev)) @ Q.T
    return X, y, K


def student_mixture(n=2000, d=20, K=6, df=4, rng=None, seed=0):
    """Heavy-tailed (multivariate Student-t) clusters with random centres and spectra."""
    rng = rng or np.random.default_rng(seed)
    y = _assign(n, K, rng)
    centers = rng.standard_normal((K, d)) * 2.5
    X = np.empty((n, d))
    for k in range(K):
        sel = y == k
        m = int(sel.sum())
        ev = np.sort(rng.uniform(0.05, 1.0, d))[::-1]
        g = rng.standard_normal((m, d)) * np.sqrt(ev) / np.sqrt(rng.chisquare(df, (m, 1)) / df)
        X[sel] = centers[k] + g @ _rotation(d, rng).T
    return X, y, K


def ring_mixture(n=2000, d=20, K=8, radius=3.0, noise=0.6, rng=None, seed=0):
    """K clusters placed on a circle in a random 2-D plane: lambda_1 ~ lambda_2 (maximal 2-axis
    ambiguity for PCA-Part) without any hypercube or elongation structure."""
    rng = rng or np.random.default_rng(seed)
    y = _assign(n, K, rng)
    ang = 2 * np.pi * (np.arange(K) + rng.uniform(0, 1)) / K
    centers = np.zeros((K, d))
    centers[:, 0], centers[:, 1] = radius * np.cos(ang), radius * np.sin(ang)
    X = centers[y] + rng.standard_normal((n, d)) * noise
    return X @ _rotation(d, rng), y, K


FAMILIES = {
    "mono": mono_directional,
    "multi_axial": multi_axial,
    "anisotropic": anisotropic,
    "shells": concentric_shells,
    "highdim": high_dim_mixture,
    "subspace_gmm": subspace_gmm,
    "random_gmm": random_gmm,
    "student": student_mixture,
    "ring": ring_mixture,
}
