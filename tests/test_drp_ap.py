import numpy as np
import pytest

from drp_ap import DRPAP, directional_rank, conditioning_index
from drp_ap import baselines as B
from drp_ap import synthetic as S
from drp_ap.core import shrinkage_alpha, spectrum
from drp_ap.metrics import matched_distance


def test_directional_rank_definition():
    lam = np.array([0.7, 0.2, 0.05, 0.05])
    assert directional_rank(lam, 0.05) == 3      # 0.95 reached at r=3
    assert directional_rank(lam, 0.5) == 1
    assert directional_rank(lam, 1.0) == 1       # delta=1 -> always rank one
    assert directional_rank(lam, 1e-9) == 4


def test_conditioning_infinite_when_rank_deficient():
    assert conditioning_index(np.array([2.0, 1.0]), m=5, d=10) == float("inf")
    assert conditioning_index(np.array([4.0, 1.0]), m=50, d=2) == pytest.approx(4.0)


def test_proposition_containment_delta_one_equals_pca_part():
    X, y, K = S.multi_axial(n=600, r=2, ratio=0.6, seed=1)
    c1 = DRPAP(delta=1.0, random_state=0).fit_init(X, K)
    c2 = B.init_pca_part(X, K)
    np.testing.assert_allclose(np.sort(c1, axis=0), np.sort(c2, axis=0), atol=1e-9)


def test_mode_a_only_policy_equals_pca_part():
    X, y, K = S.anisotropic(n=500, d=8, K=4, seed=2)
    c1 = DRPAP(mode_policy="A_only", random_state=0).fit_init(X, K)
    np.testing.assert_allclose(np.sort(c1, axis=0), np.sort(B.init_pca_part(X, K), axis=0), atol=1e-9)


@pytest.mark.parametrize("kw", [{}, {"adaptive_k": True}, {"space": "full"},
                                {"shrinkage": "oas"}, {"shrinkage": "eps"}, {"metric": "euclid"},
                                {"mode_policy": "random"}])
def test_shapes_and_finiteness(kw):
    X, y, K = S.high_dim_mixture(n=120, d=60, K=5, seed=3)   # n ~ d: small-sample regime
    C = DRPAP(random_state=0, **kw).fit_init(X, K)
    assert C.shape == (K, 60) and np.isfinite(C).all()


def test_modes_used_in_deficiency_region():
    X, y, K = S.multi_axial(n=2000, d=20, r=2, ratio=0.8, seed=0)
    m = DRPAP(random_state=0)
    m.fit_init(X, K)
    counts = m.mode_counts_
    assert counts["B"] + counts["C"] >= 1          # multi-axial -> Modes B/C triggered


def test_mono_directional_is_mode_a_dominated():
    X, y, K = S.mono_directional(n=1200, d=6, K=4, seed=0)
    m = DRPAP(delta=0.2, random_state=0)
    m.fit_init(X, K)
    assert m.mode_counts_["A"] >= 1


def test_multi_axial_spectrum_matches_construction():
    r, ratio = 2, 0.6
    X, y, K = S.multi_axial(n=40000, d=10, r=r, ratio=ratio, seed=0)
    lam, _ = spectrum(X - X.mean(0))
    assert lam[1] / lam[0] == pytest.approx(ratio, rel=0.08)
    assert K == 4


def test_theorem2_construction_pca_part_worse_than_u2_split():
    """Corrected Theorem 2: PCA-Part is suboptimal iff a^2 > (2/pi) sigma1^2 while lam2 < lam1,
    i.e. only once lam2/lam1 exceeds ~2/pi (here ratio=0.9, sigma2=0.2 -> a^2=0.86 > 0.637)."""
    X, y, K = S.multi_axial(n=4000, d=2 + 1, r=1, ratio=0.9, sigma1=1.0, sigma2=0.2, noise=0.01, seed=0)
    Xc = X - X.mean(0)
    lam, Vt = spectrum(Xc)

    def split_sse(u):
        p = Xc @ u
        L, R = Xc[p <= np.median(p)], Xc[p > np.median(p)]
        return sum(((A - A.mean(0)) ** 2).sum() for A in (L, R))
    assert split_sse(Vt[1]) < split_sse(Vt[0])


def test_theorem2_below_threshold_pca_part_is_fine():
    """Below ratio ~ 2/pi the construction gives NO failure of PCA-Part (documents the regime)."""
    X, y, K = S.multi_axial(n=4000, d=3, r=1, ratio=0.5, sigma1=1.0, sigma2=0.3, noise=0.01, seed=0)
    Xc = X - X.mean(0)
    lam, Vt = spectrum(Xc)

    def split_sse(u):
        p = Xc @ u
        L, R = Xc[p <= np.median(p)], Xc[p > np.median(p)]
        return sum(((A - A.mean(0)) ** 2).sum() for A in (L, R))
    assert split_sse(Vt[0]) <= split_sse(Vt[1])


def test_shrinkage_alpha_in_unit_interval():
    X = np.random.default_rng(0).standard_normal((30, 40))
    a = shrinkage_alpha(X - X.mean(0), "lw")
    assert 0.0 <= a <= 1.0


def test_baselines_shapes():
    X, y, K = S.anisotropic(n=300, d=6, K=3, kappa=1e3, seed=0)
    rng = np.random.default_rng(0)
    for f in (B.init_random, B.init_kmeanspp, B.init_greedy_kmeanspp, B.init_bradley_fayyad,
              B.init_kmeanspp_md, B.init_kmeanspp_md_best, B.init_kmeanspp_md_hybrid, B.init_gmm):
        C = f(X, K, rng)
        assert C.shape == (K, 6), f.__name__
    for f in (B.init_kkz, B.init_pca_part, B.init_var_part):
        assert f(X, K).shape == (K, 6), f.__name__


def test_matched_distance_permutation_invariant():
    A = np.random.default_rng(0).standard_normal((5, 3))
    assert matched_distance(A, A[::-1]) == pytest.approx(0.0, abs=1e-12)


def test_arbitrate_first_split_dominates_pca_part_and_is_deterministic():
    from drp_ap.core import DRPAP as D
    X, y, K = S.multi_axial(n=1500, d=20, r=2, ratio=0.95, seed=3)
    c1 = D(mode_policy="arbitrate", local_iters=0).fit_init(X, 2)
    c2 = B.init_pca_part(X, 2)
    def sse2(C):
        d = ((X[:, None, :] - C[None]) ** 2).sum(2)
        return d.min(1).sum()
    assert sse2(c1) <= sse2(c2) + 1e-9          # K=2: best-axis split never worse than u1 split
    a = D(mode_policy="arbitrate").fit_init(X, 5)
    b = D(mode_policy="arbitrate").fit_init(X, 5)
    np.testing.assert_array_equal(a, b)


def test_max_axes_cap_is_valid_and_matches_when_not_binding():
    X, y, K = S.multi_axial(n=1200, d=20, r=2, ratio=0.9, seed=1)
    full = DRPAP(mode_policy="arbitrate", random_state=0).fit_init(X, K)
    cap = DRPAP(mode_policy="arbitrate", max_axes=50, random_state=0).fit_init(X, K)   # cap >= r
    np.testing.assert_allclose(full, cap)
    c1 = DRPAP(mode_policy="arbitrate", max_axes=1, random_state=0).fit_init(X, K)    # = PCA-Part axis
    assert c1.shape == (K, 20) and np.isfinite(c1).all()


def test_portfolio_never_worse_than_its_candidates_after_lloyd():
    from sklearn.cluster import KMeans
    from drp_ap.experiments import init_portfolio
    X, y, K = S.multi_axial(n=1500, d=15, r=2, ratio=0.95, seed=4)

    def final(C):
        return KMeans(K, init=C, n_init=1, algorithm="lloyd", random_state=0).fit(X).inertia_
    port = final(init_portfolio(X, K))
    assert port <= final(B.init_pca_part(X, K)) + 1e-6
    assert port <= final(B.init_var_part(X, K)) + 1e-6
    assert port <= final(DRPAP(mode_policy="arbitrate").fit_init(X, K)) + 1e-6


def test_drpapsa_defaults_and_equivalences():
    from drp_ap import DRPAPSA
    X, y, K = S.multi_axial(n=1200, d=20, r=2, ratio=0.9, seed=2)
    m = DRPAPSA()
    assert m.local_iters == 5 and m.max_axes == 3 and m.mode_policy == "arbitrate"
    a = m.fit_init(X, K)
    b = DRPAP(mode_policy="arbitrate", local_iters=5, max_axes=3).fit_init(X, K)
    np.testing.assert_allclose(a, b)                       # same as the explicit configuration
    np.testing.assert_array_equal(a, DRPAPSA().fit_init(X, K))   # deterministic
    assert DRPAPSA(mode_policy="random").mode_policy == "arbitrate"   # cannot be overridden


def test_best_of_restarts_never_worse_than_single_run_and_is_seeded():
    from sklearn.cluster import KMeans
    from drp_ap.experiments import best_of
    X, y, K = S.anisotropic(n=500, d=8, K=5, kappa=1e2, seed=1)

    def final(C):
        return KMeans(K, init=C, n_init=1, algorithm="lloyd", random_state=0).fit(X).inertia_
    f1 = best_of(B.init_greedy_kmeanspp, 1)
    f10 = best_of(B.init_greedy_kmeanspp, 10)
    assert final(f10(X, K, 0)) <= final(f1(X, K, 0)) + 1e-9          # first draw is shared (same seed)
    np.testing.assert_array_equal(f10(X, K, 3), f10(X, K, 3))
    assert f10(X, K, 0).shape == (K, 8)


def test_registry_contains_new_baselines():
    from drp_ap.experiments import main_methods, ablation_methods
    mm = main_methods()
    for k in ["PCA-Part (Su-Dy, mean split)", "Var-Part (Su-Dy, mean split)", "DRP-AP-SA (mean split)",
              "DRP-AP-SA", "DRP-AP-SA (T=0)", "PCA-Part + local Lloyd (A3)",
              "Greedy k-means++ best-of-3", "Greedy k-means++ best-of-10", "Portfolio {PCA,Var,SA}"]:
        assert k in mm, k
    assert "A8: SA (T=0, q<=3)" in ablation_methods()


def test_randomised_sa_is_seeded_diverse_and_valid():
    from drp_ap import DRPAPSAR
    X, y, K = S.multi_axial(n=1200, d=20, r=3, ratio=0.95, seed=1)
    a = DRPAPSAR(random_state=1).fit_init(X, K)
    np.testing.assert_array_equal(a, DRPAPSAR(random_state=1).fit_init(X, K))   # seeded
    assert not np.allclose(a, DRPAPSAR(random_state=2).fit_init(X, K))          # diverse
    assert a.shape == (K, 20) and np.isfinite(a).all()


def test_pool_never_worse_than_its_deterministic_anchor():
    from drp_ap import DRPAPSA
    from drp_ap.experiments import best_of_pool, _sar, _final_sse
    X, y, K = S.multi_axial(n=900, d=15, r=2, ratio=0.9, seed=5)
    anchor = lambda X, K: DRPAPSA().fit_init(X, K)
    pool = best_of_pool(anchor, _sar, 4)
    assert _final_sse(X, K, pool(X, K, 0)) <= _final_sse(X, K, anchor(X, K)) + 1e-9


@pytest.mark.parametrize("fam", ["subspace_gmm", "random_gmm", "student", "ring"])
def test_new_generators_shapes(fam):
    X, y, K = S.FAMILIES[fam](seed=0)
    assert X.shape[0] == len(y) and np.isfinite(X).all() and K == len(np.unique(y))


def test_lodo_selects_without_using_heldout_dataset():
    import pandas as pd
    from drp_ap.tuning import lodo
    rows = []
    for d, (a, b) in {"d1": (1.0, 1.1), "d2": (1.0, 1.1), "d3": (1.5, 1.0)}.items():
        rows += [dict(dataset=d, method="A", nsse=a), dict(dataset=d, method="B", nsse=b),
                 dict(dataset=d, method="PCA-Part", nsse=b)]
    t, s = lodo(pd.DataFrame(rows), default="A", pool=["A", "B"])
    assert t.loc["d3", "selected"] == "A"      # d3 held out: A wins on d1, d2
    assert t.loc["d1", "selected"] == "B"      # d1 held out: B wins on d2+d3 (mean 1.05 vs 1.25)


def test_diagnostics_columns_and_extended_registry():
    from drp_ap.diagnostics import diagnose
    from drp_ap.data import EXTENDED, ALL_NAMES
    X, y, K = S.multi_axial(n=800, d=15, r=2, ratio=0.9, seed=2)
    d = diagnose(X, K)
    for k in ("root_r", "root_ratio", "frac_not_u1", "init_ratio", "final_ratio", "gain_pct"):
        assert k in d and np.isfinite(d[k])
    assert len(EXTENDED) == 18 and len(ALL_NAMES) == 30
    from drp_ap.experiments import main_methods
    for k in ("SA-R best-of-10", "SA pool-3", "SA pool-10"):
        assert k in main_methods()


def test_fuzzy_partition_of_unity_and_hard_limit():
    from drp_ap.fuzzy import FuzzyDRPAPSA, fcm
    from drp_ap import DRPAPSA
    X, y, K = S.FAMILIES["subspace_gmm"](seed=0, r=3, decay=0.7)
    f = FuzzyDRPAPSA(m=2.0)
    C = f.fit_init(X, K)
    assert C.shape == (K, X.shape[1])
    assert np.allclose(f.U_.sum(1), 1.0) and (f.U_ >= -1e-12).all()
    # m -> 1 recovers hard SSE-arbitrated centres up to the weighted-median tie-breaking
    Ch = FuzzyDRPAPSA(m=1.0).fit_init(X, K)
    assert Ch.shape == C.shape
    C2, U2, J, it = fcm(X, C, m=2.0)
    assert np.isfinite(J) and np.allclose(U2.sum(1), 1.0)


def test_improvements_shapes_and_optimal_threshold():
    from drp_ap.improve import (DRPAPLook, DRPAPSAOpt, init_halving, optimal_threshold_split)
    rng = np.random.default_rng(0)
    p = rng.standard_normal(40)
    L, R = optimal_threshold_split(p)
    o = np.sort(p)
    sse = lambda a: ((a - a.mean()) ** 2).sum()
    best = min(sse(o[:j]) + sse(o[j:]) for j in range(1, 40))
    assert abs(sse(p[L]) + sse(p[R]) - best) < 1e-9
    X, y, K = S.FAMILIES["subspace_gmm"](seed=0, r=3, decay=0.7)
    for f in (DRPAPSAOpt(), DRPAPLook(), DRPAPLook(n_d2=2, random_state=0)):
        assert f.fit_init(X, K).shape == (K, X.shape[1])
    assert init_halving(X, K, np.random.default_rng(0)).shape == (K, X.shape[1])
