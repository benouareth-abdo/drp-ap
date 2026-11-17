"""DRP-AP: Directional-Rank-Profile Adaptive Partitioning (k-means initializer).

Implements Algorithm 1 of the paper:
  * Mode A  (r_delta = 1)                       -> PCA-Part split along u_1
  * Mode B  (r_delta >= 2, well conditioned)    -> Mahalanobis 2-means in span(u_1..u_r)
  * Mode C  (r_delta >= 2, ill conditioned)     -> same, with shrinkage-regularised metric
Options expose every ablation of the protocol (A1-A5) and the DRP-AP+ variant.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.covariance import oas as _oas
from sklearn.metrics import pairwise_distances_argmin

# ----------------------------------------------------------------------------
# Spectral diagnostics
# ----------------------------------------------------------------------------


def centered_sse(Xc: np.ndarray) -> float:
    return float(np.einsum("ij,ij->", Xc, Xc))


def spectrum(Xc: np.ndarray):
    """Eigenvalues (biased covariance) and eigenvectors (rows of Vt) of a centred block.

    Uses the SVD of the data matrix, so it also works when m <= d (Gram regime).
    Returns lam (length min(m, d)) sorted decreasingly and Vt (min(m,d) x d).
    """
    m = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (s ** 2) / m, Vt


def directional_rank(lam: np.ndarray, delta: float) -> int:
    """r_delta(C): smallest r with sum_{j<=r} lam_j / sum_j lam_j >= 1 - delta."""
    tot = float(lam.sum())
    if tot <= 0.0:
        return 1
    c = np.cumsum(lam) / tot
    r = int(np.searchsorted(c, 1.0 - delta - 1e-12, side="left")) + 1
    return int(min(max(r, 1), len(lam)))


def conditioning_index(lam: np.ndarray, m: int, d: int, tiny: float = 1e-14) -> float:
    """kappa(C) = lam_1 / lam_min^+ ; infinity when the covariance is rank deficient (m <= d)."""
    if m <= d or lam[0] <= 0.0:
        return float("inf")
    lmin = lam[-1]
    if lmin <= tiny * lam[0]:
        return float("inf")
    return float(lam[0] / lmin)


def shrinkage_alpha(Xc: np.ndarray, rule: str) -> float:
    m = Xc.shape[0]
    if m < 3:
        return 1.0
    if rule == "lw":
        return float(ledoit_wolf_shrinkage(Xc, assume_centered=True))
    if rule == "oas":
        return float(_oas(Xc, assume_centered=True)[1])
    return 0.0


# ----------------------------------------------------------------------------
# Unimodality test (for DRP-AP+)
# ----------------------------------------------------------------------------
_DIP = None


def unimodality_pvalue(x: np.ndarray) -> float:
    """Hartigan dip-test p-value (package `diptest`); falls back to a 1-vs-2 Gaussian BIC test."""
    global _DIP
    if _DIP is None:
        try:
            import diptest  # type: ignore

            _DIP = diptest
        except Exception:  # pragma: no cover
            _DIP = False
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 8 or np.ptp(x) <= 0:
        return 1.0
    if _DIP:
        try:
            return float(_DIP.diptest(x)[1])
        except Exception:  # pragma: no cover
            pass
    from sklearn.mixture import GaussianMixture

    xs = x.reshape(-1, 1)
    b1 = GaussianMixture(1, random_state=0).fit(xs).bic(xs)
    b2 = GaussianMixture(2, random_state=0, n_init=2).fit(xs).bic(xs)
    return 0.01 if b2 < b1 - 10.0 else 1.0


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def median_split(proj: np.ndarray, how: str = "median"):
    """Split indices (relative to proj) into two non-empty parts."""
    if how == "mean":
        mask = proj > proj.mean()
        if 0 < mask.sum() < len(proj):
            return np.where(~mask)[0], np.where(mask)[0]
    order = np.argsort(proj, kind="stable")
    h = max(len(proj) // 2, 1)
    return order[:h], order[h:]


def kpp_indices(Z: np.ndarray, k: int, rng: np.random.Generator):
    """D^2 sampling (k-means++) of k indices under the Euclidean metric of Z."""
    n = len(Z)
    idx = [int(rng.integers(n))]
    d2 = ((Z - Z[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        s = float(d2.sum())
        i = int(rng.integers(n)) if s <= 0 else int(rng.choice(n, p=d2 / s))
        idx.append(i)
        d2 = np.minimum(d2, ((Z - Z[i]) ** 2).sum(1))
    return idx


def lloyd_in_space(Z: np.ndarray, centers: np.ndarray, T: int):
    """T Lloyd iterations. Returns labels (or None if a cluster becomes empty)."""
    k = len(centers)
    labels = pairwise_distances_argmin(Z, centers)
    for _ in range(T):
        new_centers = centers.copy()
        for j in range(k):
            sel = labels == j
            if not sel.any():
                return None
            new_centers[j] = Z[sel].mean(0)
        new_labels = pairwise_distances_argmin(Z, new_centers)
        centers = new_centers
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    if len(np.unique(labels)) < k:
        return None
    return labels


# ----------------------------------------------------------------------------
# DRP-AP
# ----------------------------------------------------------------------------


class DRPAP:
    """Directional-Rank-Profile Adaptive Partitioning.

    Parameters
    ----------
    delta        : variance fraction defining r_delta (default 0.05). delta=1 => PCA-Part (A1).
    kappa0       : conditioning threshold switching Mode B -> Mode C (default 1e3).
    shrinkage    : 'lw' (Ledoit-Wolf), 'oas', 'eps' (fixed ridge eps_rel * tr/d), 'none'.
    eps_rel      : ridge level for shrinkage='eps'.
    local_iters  : T, Lloyd iterations inside Modes B/C (default 5).
    adaptive_k   : DRP-AP+ (number of sub-clusters from the dip test on projections).
    metric       : 'mahalanobis' (default) or 'euclid' in Modes B/C (ablation A5).
    mode_policy  : 'drp' (default), 'random' (A4), 'A_only', 'B_only', 'C_only'.
    space        : 'subspace' (paper: project on top-r) or 'full' (Mahalanobis under the shrunk
                   covariance in the whole space, residual lumped in one coordinate).
    split_at     : 'median' (paper) or 'mean' for Mode A.
    axis_sampling: 'argmax' (default, deterministic) or 'gain' (randomised DRP-AP-SA-R: the split
                   axis is sampled with probability proportional to gain**rand_gamma).
    rand_gamma   : sharpness of that sampling (default 4).
    n_random_dirs: extra candidate directions drawn at random in the top-q principal subspace
                   (weighted by sqrt(lambda)); only used with axis_sampling='gain'.
    rho          : optional extension (None = paper's rule). If set, Mode A is also used whenever
                   lambda_2/lambda_1 <= rho, i.e. the second direction is negligible.
    """

    def __init__(self, delta=0.05, kappa0=1e3, shrinkage="lw", eps_rel=1e-3, local_iters=5,
                 adaptive_k=False, metric="mahalanobis", mode_policy="drp", space="subspace",
                 split_at="median", dip_alpha=0.05, rho=None, max_axes=None, random_state=None,
                 axis_sampling="argmax", rand_gamma=4.0, n_random_dirs=0):
        self.axis_sampling = axis_sampling
        self.rand_gamma = rand_gamma
        self.n_random_dirs = n_random_dirs
        self.rho = rho
        self.max_axes = max_axes
        self.delta = delta
        self.kappa0 = kappa0
        self.shrinkage = shrinkage
        self.eps_rel = eps_rel
        self.local_iters = local_iters
        self.adaptive_k = adaptive_k
        self.metric = metric
        self.mode_policy = mode_policy
        self.space = space
        self.split_at = split_at
        self.dip_alpha = dip_alpha
        self.random_state = random_state

    # -- public API ---------------------------------------------------------
    def fit_init(self, X: np.ndarray, K: int) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        self.trace_ = []
        Xc = X - X.mean(0)
        clusters = [(np.arange(n), centered_sse(Xc))]
        while len(clusters) < K:
            cand = [(i, c) for i, c in enumerate(clusters) if len(c[0]) >= 2 and c[1] > 1e-12]
            if not cand:
                break
            i = max(cand, key=lambda t: t[1][1])[0]
            idx, _ = clusters.pop(i)
            parts = self._split(X, idx, max_parts=K - len(clusters), rng=rng)
            for p in parts:
                sub = X[idx[p]]
                clusters.append((idx[p], centered_sse(sub - sub.mean(0))))
        cents = np.stack([X[c[0]].mean(0) for c in clusters])
        if len(cents) < K:  # degenerate data (fewer distinct points than K)
            extra = X[rng.choice(n, K - len(cents), replace=n < K - len(cents))]
            cents = np.vstack([cents, extra])
        return cents

    @property
    def mode_counts_(self):
        out = {"A": 0, "B": 0, "C": 0}
        for t in self.trace_:
            out[t["mode"]] = out.get(t["mode"], 0) + 1
        return out

    # -- internals ----------------------------------------------------------
    def _choose_mode(self, r, kappa, rng, ratio=0.0):
        p = self.mode_policy
        if p == "A_only":
            return "A"
        if p == "B_only":
            return "B"
        if p == "C_only":
            return "C"
        if p == "random":
            return str(rng.choice(["A", "B", "C"]))
        if r == 1 or (self.rho is not None and ratio <= self.rho):
            return "A"          # rho-gate (optional extension): negligible 2nd direction
        return "B" if kappa <= self.kappa0 else "C"

    def _split_arbitrate(self, Xc, Vt, r, rng=None):
        """SSE-arbitrated split (prototype 'DRP-AP-SA', deterministic).

        Candidates = median splits along u_1..u_q with q = min(max_axes or r_delta, rank).
        The candidate with the largest SSE reduction (between-group SS, exact in the full space)
        wins; u_1 is always a candidate, so each split removes at least as much SSE as PCA-Part's.
        The winner is then refined by `local_iters` Euclidean 2-means (Lloyd) iterations, which can
        only lower the split SSE.  No whitening, no randomness."""
        m = Xc.shape[0]
        q = min(self.max_axes if self.max_axes else r, Vt.shape[0])
        P = Xc @ Vt[:q].T
        dirs = [P[:, j] for j in range(q)]
        if self.axis_sampling == "gain" and self.n_random_dirs > 0 and rng is not None:
            lam_q = np.maximum((P ** 2).mean(0), 1e-300)
            for _ in range(self.n_random_dirs):
                w = rng.standard_normal(q) * np.sqrt(lam_q)
                w /= np.linalg.norm(w) + 1e-300
                dirs.append(P @ w)
        cands, gains = [], []
        for pj in dirs:
            L, R = median_split(pj, self.split_at)
            if len(L) == 0 or len(R) == 0:
                continue
            gains.append(len(L) * len(R) / m * float(((Xc[L].mean(0) - Xc[R].mean(0)) ** 2).sum()))
            cands.append((L, R))
        if not cands:
            L, R = median_split(Xc @ Vt[0], self.split_at)
            self._axis = 0
            return [L, R]
        gains = np.asarray(gains)
        if self.axis_sampling == "gain" and rng is not None:
            w = (gains / max(gains.max(), 1e-300)) ** self.rand_gamma
            j_ = int(rng.choice(len(cands), p=w / w.sum()))
            L, R = cands[j_]
            self._axis = j_
        else:
            L, R = cands[int(np.argmax(gains))]
            self._axis = int(np.argmax(gains))
        if self.local_iters > 0:
            c = np.stack([Xc[L].mean(0), Xc[R].mean(0)])
            lab = lloyd_in_space(Xc, c, self.local_iters)
            if lab is not None:
                L, R = np.where(lab == 0)[0], np.where(lab == 1)[0]
        return [L, R]

    def _split(self, X, idx, max_parts, rng):
        sub = X[idx]
        Xc = sub - sub.mean(0)
        m, d = Xc.shape
        lam, Vt = spectrum(Xc)
        r = directional_rank(lam, self.delta)
        kappa = conditioning_index(lam, m, d)
        ratio = float(lam[1] / lam[0]) if len(lam) > 1 and lam[0] > 0 else 0.0
        if self.mode_policy == "arbitrate":
            parts = self._split_arbitrate(Xc, Vt, r, rng)
            self.trace_.append(dict(size=m, r=r, kappa=kappa, mode="S", k_c=2, axis=self._axis,
                                    ratio=ratio))
            return parts
        mode = self._choose_mode(r, kappa, rng, ratio)
        k_c = 2
        parts = None

        if mode != "A" and len(lam) >= 2:
            rr = min(max(r, 2), len(lam))
            lam_t = lam.copy()
            lam_null = 0.0
            if mode == "C" and self.shrinkage != "none":
                lbar = float(lam.sum()) / d
                if self.shrinkage == "eps":
                    lam_t = lam + self.eps_rel * lbar
                    lam_null = self.eps_rel * lbar
                else:
                    a = shrinkage_alpha(Xc, self.shrinkage)
                    lam_t = (1.0 - a) * lam + a * lbar
                    lam_null = a * lbar
            floor = max(1e-12 * float(lam_t[0]), 1e-300)
            if self.metric == "mahalanobis":
                if self.space == "full" and mode == "C":
                    proj = Xc @ Vt.T
                    Z = proj / np.sqrt(np.maximum(lam_t, floor))
                    if lam_null > 0 and Vt.shape[0] < d:
                        resid = np.sqrt(np.maximum((Xc ** 2).sum(1) - (proj ** 2).sum(1), 0.0))
                        Z = np.hstack([Z, (resid / np.sqrt(lam_null))[:, None]])
                else:
                    Z = (Xc @ Vt[:rr].T) / np.sqrt(np.maximum(lam_t[:rr], floor))
            else:
                Z = Xc @ Vt[:rr].T

            if self.adaptive_k:
                n_sig = sum(unimodality_pvalue(Xc @ Vt[j]) < self.dip_alpha for j in range(rr))
                k_c = int(min(max(1 + n_sig, 2), max_parts))
            k_c = max(2, min(k_c, max_parts, m))
            seeds = kpp_indices(Z, k_c, rng)
            labels = lloyd_in_space(Z, Z[seeds].copy(), self.local_iters)
            if labels is not None:
                parts = [np.where(labels == j)[0] for j in range(k_c)]
            else:
                mode, k_c = "A", 2  # fallback: robust PCA-Part split

        if parts is None:
            L, R = median_split(Xc @ Vt[0], self.split_at)
            parts = [L, R]
            k_c = 2
        self.trace_.append(dict(size=m, r=r, kappa=kappa, mode=mode, k_c=k_c))
        return parts


class DRPAPSA(DRPAP):
    """DRP-AP-SA: the recommended, deterministic SSE-arbitrated variant.

    Defaults follow the full benchmark (results4):
      * local_iters=5  - per-split Lloyd refinement. T=0 is statistically indistinguishable from
                         PCA-Part on real data; T=5 carries the real gains (20NG, Isolet, Iris,
                         Olivetti, synthetic multi-axial). Known cost: +3.6% on COIL-20
                         (Portfolio avoids it). Use local_iters=0 for the cheaper variant;
      * max_axes=3     - arbitrate among the top-3 principal axes only (identical SSE to the
                         uncapped version on all 11 datasets, much cheaper in high dimension).
    """

    def __init__(self, local_iters=5, max_axes=3, **kw):
        kw.pop("mode_policy", None)
        super().__init__(mode_policy="arbitrate", local_iters=local_iters, max_axes=max_axes, **kw)


class DRPAPSAR(DRPAPSA):
    """DRP-AP-SA-R: randomised SSE-arbitrated splitting, meant to be used inside a restart pool.

    The split axis is sampled with probability proportional to (gain / best gain)**rand_gamma over
    the top-q principal axes plus `n_random_dirs` random directions of the top-q subspace, so
    the best axis is the most likely but not the only one.  With a different seed each call gives a
    different, still SSE-informed, initialisation."""

    def __init__(self, rand_gamma=4.0, n_random_dirs=3, **kw):
        kw.pop("axis_sampling", None)
        super().__init__(axis_sampling="gain", rand_gamma=rand_gamma,
                         n_random_dirs=n_random_dirs, **kw)
