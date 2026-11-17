"""Metrics of the protocol: SSE, ARI, NMI, silhouette, seed stability, failure rate."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             pairwise_distances_argmin_min, silhouette_score)


def sse(X, C) -> float:
    _, d = pairwise_distances_argmin_min(X, C)
    return float((d ** 2).sum())


def external_scores(y, labels):
    if y is None:
        return np.nan, np.nan
    return (float(adjusted_rand_score(y, labels)),
            float(normalized_mutual_info_score(y, labels)))


def silhouette(X, labels, sample=2000, seed=0) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(X, labels, sample_size=min(sample, len(X) - 1), random_state=seed)
                 if len(X) > sample else silhouette_score(X, labels))


def diameter(X) -> float:
    """Bounding-box diagonal (cheap upper bound of the data diameter)."""
    return float(np.linalg.norm(X.max(0) - X.min(0)))


def matched_distance(A, B) -> float:
    """Mean Euclidean distance between two centroid sets after optimal (Hungarian) matching."""
    D = cdist(A, B)
    r, c = linear_sum_assignment(D)
    return float(D[r, c].mean())


def seed_stability(centroid_sets, diam) -> float:
    """Average pairwise matched distance between initial centroids across resamples,
    normalised by the data diameter (lower = more stable)."""
    B = len(centroid_sets)
    vals = [matched_distance(centroid_sets[i], centroid_sets[j])
            for i in range(B) for j in range(i + 1, B)]
    return float(np.mean(vals) / max(diam, 1e-300)) if vals else np.nan
