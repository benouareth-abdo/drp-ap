"""Fuzzy extension of DRP-AP-SA: F-DRP-AP-SA and a fuzzy c-means (FCM) toolbox.

Idea.  DRP-AP-SA splits clusters by a hard hyperplane.  Here every point keeps a membership
weight in each node of the divisive tree: a split of a node with weights ``w`` produces two
children with weights ``w*u`` and ``w*(1-u)``, where ``u`` is the membership of the point in
the left child given by two-cluster fuzzy c-means with fuzzifier ``m`` (the hard SSE-arbitrated
split along the best of the top-q principal axes only provides the starting centres).
Memberships stay a partition of unity at every level, so the leaves give a fuzzy partition
``U`` (n x K) usable directly as the starting point of FCM.  Weighted PCA, weighted SSE and
weighted medians replace their hard counterparts, and the node to split is the one with the
largest weighted SSE.  With ``m -> 1`` the construction is DRP-AP-SA.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _wmean(X, w):
    s = float(w.sum())
    return (w[:, None] * X).sum(0) / max(s, _EPS)


def _wsse(X, w):
    mu = _wmean(X, w)
    return float((w * ((X - mu) ** 2).sum(1)).sum())


def _wmedian(p, w):
    o = np.argsort(p, kind="stable")
    cw = np.cumsum(w[o])
    return float(p[o][np.searchsorted(cw, 0.5 * cw[-1])])


def memberships(X, C, m=2.0):
    """FCM memberships u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1)); hard assignment if m <= 1."""
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2) if X.shape[0] * C.shape[0] * X.shape[1] < 4e6 \
        else _d2_chunked(X, C)
    if m <= 1.0001:
        U = np.zeros_like(d2)
        U[np.arange(len(X)), d2.argmin(1)] = 1.0
        return U, d2
    d2 = np.maximum(d2, _EPS)
    inv = d2 ** (-1.0 / (m - 1.0))
    return inv / inv.sum(1, keepdims=True), d2


def _d2_chunked(X, C):
    x2 = (X ** 2).sum(1)[:, None]
    c2 = (C ** 2).sum(1)[None, :]
    return np.maximum(x2 + c2 - 2.0 * X @ C.T, 0.0)


def fcm(X, C0, m=2.0, max_iter=300, tol=1e-5):
    """Fuzzy c-means from initial centres C0. Returns (C, U, J, n_iter)."""
    C = np.asarray(C0, dtype=float).copy()
    J_old = np.inf
    for it in range(1, max_iter + 1):
        U, d2 = memberships(X, C, m)
        Um = U ** m
        C = (Um.T @ X) / np.maximum(Um.sum(0)[:, None], _EPS)
        J = float((Um * _d2_chunked(X, C)).sum())
        if abs(J_old - J) <= tol * max(J, _EPS):
            break
        J_old = J
    U, _ = memberships(X, C, m)
    J = float(((U ** m) * _d2_chunked(X, C)).sum())
    return C, U, J, it


def fcm_objective(X, C, m=2.0):
    U, _ = memberships(X, C, m)
    return float(((U ** m) * _d2_chunked(X, C)).sum())


class FuzzyDRPAPSA:
    """F-DRP-AP-SA: fuzzy, SSE-arbitrated divisive initializer.

    Parameters: local_iters (T) fuzzy 2-means iterations per split, max_axes (q) candidate
    principal axes, m fuzzifier (m <= 1 gives hard splits = DRP-AP-SA), min_weight to drop
    points whose node weight is negligible.
    Attributes after ``fit_init``: ``U_`` (n x K fuzzy partition), ``trace_``.
    """

    def __init__(self, local_iters=5, max_axes=3, m=2.0, min_weight=1e-4, random_state=None):
        self.local_iters, self.max_axes, self.m = local_iters, max_axes, m
        self.min_weight, self.random_state = min_weight, random_state

    def fit_init(self, X, K):
        X = np.asarray(X, dtype=float)
        n = len(X)
        nodes = [np.ones(n)]
        sse = [_wsse(X, nodes[0])]
        self.trace_ = []
        while len(nodes) < K:
            i = int(np.argmax(sse))
            if sse[i] <= _EPS:
                break
            w = nodes.pop(i)
            sse.pop(i)
            wl, wr = self._split(X, w)
            for c in (wl, wr):
                nodes.append(c)
                sse.append(_wsse(X, c))
        U = np.stack(nodes, 1)
        self.U_ = U
        Um = U ** self.m if self.m > 1.0001 else U
        C = (Um.T @ X) / np.maximum(Um.sum(0)[:, None], _EPS)
        if len(C) < K:  # degenerate data: pad with random points
            rng = np.random.default_rng(self.random_state)
            C = np.vstack([C, X[rng.choice(n, K - len(C), replace=n < K - len(C))]])
        return C

    def _split(self, X, w):
        idx = np.where(w > self.min_weight * w.max())[0]
        Xa, wa = X[idx], w[idx]
        mu = _wmean(Xa, wa)
        Xc = Xa - mu
        Z = np.sqrt(wa)[:, None] * Xc
        lam, V = np.linalg.eigh(Z.T @ Z)
        o = np.argsort(lam)[::-1]
        lam, V = lam[o], V[:, o]
        q = max(1, min(self.max_axes, int((lam > 1e-12 * max(lam[0], _EPS)).sum())))
        best, bg, bax = None, -1.0, 0
        for j in range(q):
            p = Xc @ V[:, j]
            t = _wmedian(p, wa)
            L = p <= t
            if L.all() or (~L).all():
                L = p < t
                if L.all() or (~L).all():
                    continue
            wl, wr = wa[L].sum(), wa[~L].sum()
            g = wl * wr / (wl + wr) * float(((_wmean(Xa[L], wa[L]) - _wmean(Xa[~L], wa[~L])) ** 2).sum())
            if g > bg:
                best, bg, bax = L, g, j
        if best is None:  # cannot split: duplicate with equal weights
            half = np.full(len(idx), 0.5)
            wl = np.zeros_like(w); wr = np.zeros_like(w)
            wl[idx] = wa * half; wr[idx] = wa * half
            return wl, wr
        L = best
        cL, cR = _wmean(Xa[L], wa[L]), _wmean(Xa[~L], wa[~L])
        m = self.m
        for _ in range(max(self.local_iters, 0)):
            u = self._uL(Xa, cL, cR, m)
            if m > 1.0001:
                wl_, wr_ = wa * u ** m, wa * (1 - u) ** m
            else:
                wl_, wr_ = wa * u, wa * (1 - u)
            if wl_.sum() <= _EPS or wr_.sum() <= _EPS:
                break
            cL, cR = _wmean(Xa, wl_), _wmean(Xa, wr_)
        u = self._uL(Xa, cL, cR, m)
        self.trace_.append(dict(size=float(wa.sum()), axis=bax, gain=bg))
        wl = np.zeros_like(w); wr = np.zeros_like(w)
        wl[idx] = wa * u
        wr[idx] = wa * (1 - u)
        return wl, wr

    @staticmethod
    def _uL(Xa, cL, cR, m):
        dl = ((Xa - cL) ** 2).sum(1)
        dr = ((Xa - cR) ** 2).sum(1)
        if m <= 1.0001:
            return (dl <= dr).astype(float)
        dl, dr = np.maximum(dl, _EPS), np.maximum(dr, _EPS)
        return 1.0 / (1.0 + (dl / dr) ** (1.0 / (m - 1.0)))
