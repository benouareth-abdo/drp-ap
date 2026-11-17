"""The twelve benchmark datasets of Table 1 (tabular, image, text, audio, sensor).

Bundled with scikit-learn (no download): iris, wine, breast_cancer, digits.
Downloaded on first use (internet needed, fine on Colab): MNIST, Fashion-MNIST, COIL-20,
Olivetti, 20 Newsgroups, Reuters R8, Isolet, UCI HAR.

Any dataset that cannot be obtained is skipped with a warning; you can also drop a file
`<data_dir>/<name>.npz` containing arrays X and y to supply it manually.
"""
from __future__ import annotations

import io
import os
import urllib.request
import warnings
import zipfile

import numpy as np
from sklearn import datasets as skd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler

DEFAULT_DIR = os.environ.get("DRP_AP_DATA", "./data_cache")

# name -> (modality, description)
DATASETS = {
    "Iris": "tabular", "Wine": "tabular", "BC": "tabular",
    "Digits": "image", "MNIST": "image", "F-MNIST": "image", "COIL-20": "image",
    "Olivetti": "image", "20NG": "text", "R8": "text", "Isolet": "audio", "HAR": "sensor",
}


# Extended protocol: 18 additional datasets from OpenML (downloaded on first use;
# the names are the OpenML dataset names and were NOT checked offline: a failing name is skipped with
# a warning, drop <data_dir>/<name>.npz to supply it). Together with the 12 core ones: the 30-dataset benchmark.
EXTENDED = {
    "glass": "tabular", "ecoli": "tabular", "yeast": "tabular", "vehicle": "tabular",
    "segment": "image-features", "satimage": "remote-sensing", "pendigits": "pen-trajectory",
    "letter": "image-features", "waveform-5000": "signal", "spambase": "text-features",
    "ionosphere": "signal", "sonar": "signal", "vowel": "audio", "page-blocks": "document",
    "mfeat-karhunen": "image-features", "mfeat-fourier": "image-features", "optdigits": "image",
    "dermatology": "tabular",
}
ALL_NAMES = list(DATASETS) + list(EXTENDED)

# Confirmation set: chosen BEFORE running the confirmation experiment, disjoint from the 30 above.
CONFIRM = {
    "mfeat-zernike": "tabular", "mfeat-factors": "tabular", "mfeat-pixel": "tabular",
    "mfeat-morphological": "tabular", "texture": "tabular", "wall-robot-navigation": "sensor",
    "cardiotocography": "tabular", "steel-plates-fault": "tabular",
    "one-hundred-plants-margin": "tabular", "one-hundred-plants-texture": "tabular",
    "one-hundred-plants-shape": "tabular", "phoneme": "audio", "balance-scale": "tabular",
    "JapaneseVowels": "audio", "collins": "text", "cnae-9": "text", "semeion": "image",
}


def _openml_tab(name, data_dir):
    """OpenML tabular dataset: one-hot for categorical columns, median imputation, z-scoring, PCA to
    <=50 dimensions when d is larger."""
    import pandas as pd

    last = None
    for ver in (1, "active"):                      # version 1 first, then the active version
        for _ in range(2):                         # one retry on transient network errors
            try:
                b = skd.fetch_openml(name, version=ver, as_frame=True, data_home=data_dir)
                break
            except Exception as e:                 # noqa: BLE001
                last = e
        else:
            continue
        break
    else:
        raise last
    X = pd.get_dummies(b.data, dtype=float)
    X = X.fillna(X.median()).to_numpy(float)
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)
    if X.shape[1] > 50:
        X = _pca(X)
    return X, _factorize(b.target)


def _pca(X, d=50, seed=0):
    d = min(d, X.shape[1], X.shape[0] - 1)
    return PCA(d, random_state=seed).fit_transform(X)


def _factorize(y):
    return np.unique(np.asarray(y), return_inverse=True)[1]


def _openml(name, data_dir, version=1):
    b = skd.fetch_openml(name, version=version, as_frame=False, data_home=data_dir)
    return np.asarray(b.data, dtype=float), _factorize(b.target)


def _tfidf_lsa(texts, n_comp=100, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer

    V = TfidfVectorizer(max_features=20000, stop_words="english", min_df=5, sublinear_tf=True)
    T = V.fit_transform(texts)
    Z = TruncatedSVD(min(n_comp, T.shape[1] - 1), random_state=seed).fit_transform(T)
    return Normalizer().fit_transform(Z)


def _load_coil20(data_dir):
    url = ("https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/"
           "coil-20-proc.zip")
    from PIL import Image

    raw = urllib.request.urlopen(url, timeout=60).read()
    X, y = [], []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for nm in sorted(z.namelist()):
            if nm.lower().endswith(".png"):
                lab = int(os.path.basename(nm).split("__")[0].replace("obj", ""))
                im = Image.open(z.open(nm)).convert("L").resize((32, 32))
                X.append(np.asarray(im, dtype=float).ravel() / 255.0)
                y.append(lab)
    return np.array(X), _factorize(y)


def _load_r8(data_dir):
    base = "https://www.cs.umb.edu/~smimarog/textmining/datasets/"
    texts, labels = [], []
    for f in ("r8-train-no-stop.txt", "r8-test-no-stop.txt"):
        local = os.path.join(data_dir, f)          # manual copy of the raw file takes priority
        if os.path.exists(local):
            raw = open(local, encoding="utf-8", errors="ignore").read()
        else:
            raw = urllib.request.urlopen(base + f, timeout=60).read().decode("utf-8", "ignore")
        for line in raw.splitlines():
            if "\t" in line:
                lab, txt = line.split("\t", 1)
                labels.append(lab)
                texts.append(txt)
    return _tfidf_lsa(texts), _factorize(labels)


def _raw(name, data_dir):
    if name == "Iris":
        b = skd.load_iris(); return StandardScaler().fit_transform(b.data), b.target
    if name == "Wine":
        b = skd.load_wine(); return StandardScaler().fit_transform(b.data), b.target
    if name == "BC":
        b = skd.load_breast_cancer(); return StandardScaler().fit_transform(b.data), b.target
    if name == "Digits":
        b = skd.load_digits(); return b.data.astype(float), b.target
    if name == "MNIST":
        X, y = _openml("mnist_784", data_dir); return _pca(X / 255.0), y
    if name == "F-MNIST":
        X, y = _openml("Fashion-MNIST", data_dir); return _pca(X / 255.0), y
    if name == "COIL-20":
        X, y = _load_coil20(data_dir); return _pca(X), y
    if name == "Olivetti":
        b = skd.fetch_olivetti_faces(data_home=data_dir); return _pca(b.data), b.target
    if name == "20NG":
        b = skd.fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                                   data_home=data_dir)
        return _tfidf_lsa(b.data), b.target
    if name == "R8":
        return _load_r8(data_dir)
    if name == "Isolet":
        X, y = _openml("isolet", data_dir); return _pca(StandardScaler().fit_transform(X)), y
    if name == "HAR":
        X, y = _openml("har", data_dir); return _pca(StandardScaler().fit_transform(X)), y
    if name in EXTENDED or name in CONFIRM:
        return _openml_tab(name, data_dir)
    raise KeyError(name)


def load_dataset(name, data_dir=DEFAULT_DIR, max_n=None, seed=0):
    """Return (X, y, K). `max_n` draws a stratified subsample (for speed on Colab)."""
    os.makedirs(data_dir, exist_ok=True)
    manual = os.path.join(data_dir, f"{name}.npz")
    if os.path.exists(manual):
        z = np.load(manual, allow_pickle=True)
        X, y = z["X"].astype(float), _factorize(z["y"])
    else:
        X, y = _raw(name, data_dir)
    X = np.asarray(X, dtype=float)
    y = _factorize(y)
    if max_n and len(X) > max_n:
        rng = np.random.default_rng(seed)
        keep = []
        for c in np.unique(y):
            ids = np.where(y == c)[0]
            keep.append(rng.choice(ids, max(int(round(max_n * len(ids) / len(X))), 2),
                                   replace=False))
        keep = np.sort(np.concatenate(keep))
        X, y = X[keep], y[keep]
    return X, y, int(len(np.unique(y)))


def manifest(data):
    """One row per loaded dataset: modality, n, d, K, smallest class size (for the paper's table)."""
    import pandas as pd
    mod = {**DATASETS, **EXTENDED, **CONFIRM}
    rows = []
    for n, (X, y, K) in data.items():
        rows.append(dict(dataset=n, modality=mod.get(n, "?"), n=len(X), d=X.shape[1], K=K,
                         min_class=int(np.bincount(y).min()), n_per_class=round(len(X) / K, 1)))
    return pd.DataFrame(rows)


def load_many(names, data_dir=DEFAULT_DIR, max_n=None, seed=0, verbose=True):
    out = {}
    for n in names:
        try:
            out[n] = load_dataset(n, data_dir, max_n, seed)
            if verbose:
                X, y, K = out[n]
                print(f"[data] {n:9s} n={X.shape[0]:6d} d={X.shape[1]:4d} K={K}")
        except Exception as e:  # network failure, missing file...
            warnings.warn(f"Dataset {n} skipped: {type(e).__name__}: {str(e)[:120]}. "
                          f"Provide {os.path.join(data_dir, n + '.npz')} (arrays X, y) to add it.")
    return out
