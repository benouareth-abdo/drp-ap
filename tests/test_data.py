import numpy as np

from drp_ap import data as D


def test_manual_npz_and_stratified_subsample(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((900, 7))
    y = np.repeat(np.array(["a", "b", "c"]), 300)
    np.savez(tmp_path / "Toy.npz", X=X, y=y)
    Xs, ys, K = D.load_dataset("Toy", data_dir=str(tmp_path), max_n=150)
    assert K == 3 and 140 <= len(Xs) <= 160
    assert set(np.bincount(ys)) <= set(range(45, 56))      # roughly stratified


def test_tfidf_lsa_pipeline():
    topics = [["goal match team score league"], ["stock market price trade bank"],
              ["gene protein cell biology dna"]]
    texts, labels = [], []
    rng = np.random.default_rng(0)
    for k, t in enumerate(topics):
        words = t[0].split()
        for _ in range(60):
            texts.append(" ".join(rng.choice(words, 8)) + " common filler words here")
            labels.append(k)
    Z = D._tfidf_lsa(texts, n_comp=5)
    assert Z.shape[0] == 180 and Z.shape[1] <= 5
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-6)


def test_bundled_datasets_load():
    for name, d in [("Iris", 4), ("Wine", 13), ("BC", 30), ("Digits", 64)]:
        X, y, K = D.load_dataset(name)
        assert X.shape[1] == d and K >= 2


def test_pca_reduction_dimension():
    X = np.random.default_rng(0).standard_normal((120, 200))
    assert D._pca(X, 50).shape == (120, 50)
