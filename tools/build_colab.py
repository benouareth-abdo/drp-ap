"""Generate DRP_AP_Colab.ipynb: a single self-contained notebook that writes the whole package
(%%writefile cells) and runs the protocol.  Usage:  python tools/build_colab.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["drp_ap/__init__.py", "drp_ap/core.py", "drp_ap/baselines.py", "drp_ap/metrics.py",
         "drp_ap/stats.py", "drp_ap/synthetic.py", "drp_ap/data.py", "drp_ap/experiments.py",
         "drp_ap/reporting.py", "drp_ap/tuning.py", "drp_ap/diagnostics.py", "run_experiments.py",
         "tools/merge_results.py",
         "tests/test_drp_ap.py", "tests/test_data.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [md("""
# DRP-AP: Directional-Rank-Profile Adaptive Partitioning — reproducible Colab

Implements the paper *Directional-Rank Deficiency: A Unified Failure Mode of PCA-Part and
Mahalanobis k-means++ with a Principled Fix*: the **DRP-AP** initializer, 10 baselines, the
12-dataset protocol (tabular, image, text, audio, sensor), synthetic generators with controlled
directional rank, ablations A1–A7, sensitivity grid, seed stability, and Friedman/Nemenyi +
Wilcoxon–Holm statistics.

**How to use:** *Runtime → Run all.* The notebook is self-contained (it writes the package files
itself). Start with `QUICK = True` (≈ 1 min); then set `QUICK = False` for the full protocol.
GPU is not needed.
"""),
         code("""
!pip -q install diptest pillow
!mkdir -p drp_ap tests tools results data_cache
"""),
         md("## 1. Package source (written to the Colab disk)")]

for f in FILES:
    src = open(os.path.join(ROOT, f)).read()
    cells.append(code(f"%%writefile {f}\n" + src))

cells += [
    md("### Check that every file was written (re-run the cells above if anything is missing)"),
    code("""
import os
need = %s
missing = [f for f in need if not os.path.exists(f)]
assert not missing, "Missing files: %%s  -> re-run the writefile cells above (Runtime > Run all)" %% missing
print("OK: all", len(need), "files present")
""" % repr(FILES)),
    md("## 2. Self-check (unit tests: Proposition 1 containment, Theorem 2 construction, shapes…)"),
    code("!python -m pytest -q tests"),
    md("## 3. Quick demo — the deficiency region on one synthetic dataset"),
    code("""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from drp_ap import DRPAP, DRPAPSA
from drp_ap import baselines as B, synthetic as S, metrics as M

RATIO = 0.95          # lambda2/lambda1 of the Theorem-2 construction (try 0.2, 0.5, 0.8, 0.95)
rows = []
for draw in range(10):                       # one draw is noisy: average over 10
    X, y, K = S.multi_axial(n=2000, d=20, r=2, ratio=RATIO, seed=draw)
    inits = {
      "k-means++":            lambda: B.init_kmeanspp(X, K, np.random.default_rng(draw)),
      "PCA-Part":             lambda: B.init_pca_part(X, K),
      "k-means++ MD":         lambda: B.init_kmeanspp_md(X, K, np.random.default_rng(draw)),
      "DRP-AP (paper)":       lambda: DRPAP(random_state=draw).fit_init(X, K),
      "DRP-AP-SA (T=0)":      lambda: DRPAPSA(local_iters=0).fit_init(X, K), # no split refinement
      "DRP-AP-SA (default)":  lambda: DRPAPSA().fit_init(X, K),              # T=5, top-3 axes (recommended)
    }
    for name, f in inits.items():
        C = f()
        km = KMeans(K, init=C, n_init=1).fit(X)
        rows.append((draw, name, M.sse(X, C), km.inertia_, km.n_iter_))
df = pd.DataFrame(rows, columns=["draw", "method", "init SSE", "final SSE", "iterations"])
df["final / best"] = df["final SSE"] / df.groupby("draw")["final SSE"].transform("min")
print(f"Mean over 10 draws, lambda2/lambda1 = {RATIO}")
print(df.groupby("method")[["init SSE", "final SSE", "final / best", "iterations"]].mean().round(3))

X, y, K = S.multi_axial(n=2000, d=20, r=2, ratio=RATIO, seed=0)
m = DRPAP(random_state=0); m.fit_init(X, K)
print("\\nMode usage at each split (draw 0):", m.mode_counts_)
print(pd.DataFrame(m.trace_).round(3))
"""),
    md("""
## 4. Configuration
* `QUICK=True`: 3 seeds, offline datasets (Iris, Wine, Breast Cancer, Digits) — smoke test.
* `QUICK=False`: 30 seeds on all 12 datasets (MNIST, Fashion-MNIST, COIL-20, Olivetti, 20 Newsgroups,
  Reuters R8, Isolet, UCI HAR are downloaded on first use; needs internet, available on Colab).
  `MAX_N` caps large datasets by stratified subsampling to keep runtime reasonable (set `0` for full size).
* If a dataset cannot be downloaded it is skipped with a warning; upload `data_cache/<Name>.npz`
  (arrays `X`, `y`) to supply it by hand.
* Optionally mount Google Drive and point `DATA_DIR` there to cache downloads between sessions.
"""),
    code("""
QUICK      = True        # False = full protocol
DATASETS   = "offline" if QUICK else "benchmark30"   # 30 datasets; or "all" (12 core) or a list
# To spread the 30 datasets over several Colab sessions use a list per session and a different OUT, e.g.
#   session 1: DATASETS = ["Iris","Wine","BC","Digits","MNIST","F-MNIST","COIL-20","Olivetti","20NG","R8","Isolet","HAR"]; OUT = "results_A"
#   session 2: DATASETS = ["glass","ecoli","yeast","vehicle","segment","satimage","pendigits","letter","waveform-5000"]; OUT = "results_B"
#   session 3: DATASETS = ["spambase","ionosphere","sonar","vowel","page-blocks","mfeat-karhunen","mfeat-fourier","optdigits","dermatology"]; OUT = "results_C"
# then merge them in section 7b (tools/merge_results.py).
OUT        = "results"
SEEDS      = 3 if QUICK else 30
MAX_N      = 20000       # 0 = no cap
EXTRAS     = True        # synthetic (x2) + ablations + sensitivity + stability + LODO tuning + diagnostics
RESTARTS   = True        # equal-compute baselines Greedy k-means++ best-of-3/10 (heavy: +13 Lloyd runs/seed)
DATA_DIR   = "data_cache"   # e.g. "/content/drive/MyDrive/drp_ap_data" after mounting Drive

# from google.colab import drive; drive.mount('/content/drive')
"""),
    md("""
### Check which of the 30 datasets can be loaded (run this before the long run)
Downloads are cached in `DATA_DIR`. A dataset that fails (wrong name, network) is listed here;
drop `DATA_DIR/<name>.npz` (arrays `X`, `y`) to supply it by hand. The table is the paper's dataset table.
"""),
    code("""
import warnings; warnings.filterwarnings("ignore")
from drp_ap.data import ALL_NAMES, load_many, manifest
_names = ALL_NAMES if DATASETS in ("benchmark30", "extended") else (DATASETS if isinstance(DATASETS, list) else None)
if _names:
    _d = load_many(_names, DATA_DIR, MAX_N or None, verbose=False)
    print(f"loaded {len(_d)}/{len(_names)}; missing: {[n for n in _names if n not in _d]}")
    import pandas as pd; display(manifest(_d))
"""),
    md("## 5. Run the experimental protocol"),
    code("""
import subprocess, shlex
cmd = f"python run_experiments.py --datasets {DATASETS if isinstance(DATASETS, str) else ' '.join(DATASETS)} " \\
      f"--seeds {SEEDS} --max-n {MAX_N} --data-dir {DATA_DIR} --out {OUT} " \\
      + ("--quick " if QUICK else "") + ("" if RESTARTS else "--no-restarts ") + ("--all-extras" if EXTRAS else "")
print(cmd)
!{cmd}
"""),
    md("## 6. Results: tables, statistics and figures"),
    code("""
import pandas as pd, os
from IPython.display import Image, display
pd.set_option("display.width", 200)

print("Aggregate (mean over datasets and seeds; NSSE = normalised SSE, 1 = best observed)")
display(pd.read_csv("results/aggregate.csv", index_col=0).round(4))
print("Average ranks (Friedman/Nemenyi)")
display(pd.read_csv("results/average_ranks.csv", index_col=0).round(2)) if os.path.exists("results/average_ranks.csv") else None
print("Mode usage")
display(pd.read_csv("results/mode_usage.csv").round(3))
for f in ["cd_diagram.png", "synthetic_multiaxial.png"]:
    if os.path.exists("results/" + f):
        display(Image("results/" + f))
if os.path.exists("results/synthetic_verdicts.txt"):
    print(open("results/synthetic_verdicts.txt").read())
for f in ["ablation_table.csv", "stability.csv"]:
    if os.path.exists("results/" + f):
        print(f); display(pd.read_csv("results/" + f).round(4))
"""),
    md("""
## 6b. Finish / resume the ablations and the sensitivity grid
If the main run stopped during the ablation stage (timeout, disconnect), do NOT re-run everything: the
main results are already saved. Run only the missing stages. Progress is saved after every dataset in
`ablation_raw.csv` / `sensitivity_raw.csv`, so the cell can be re-run until it prints `Done`.
The sensitivity grid (4 x 4 x 3 configurations x 5 seeds) is the slow one: restrict it to a few datasets.
"""),
    code("""
# ablations A1-A8 on all datasets (resumable):
!python run_experiments.py --datasets {DATASETS if isinstance(DATASETS, str) else ' '.join(DATASETS)} --max-n {MAX_N} --data-dir {DATA_DIR} --out {OUT} --skip-main --ablation
# sensitivity grid on a representative subset (resumable):
!python run_experiments.py --datasets Iris Wine BC Digits COIL-20 Olivetti glass vehicle segment --max-n {MAX_N} --data-dir {DATA_DIR} --out {OUT} --skip-main --sensitivity
"""),
    md("""
## 7b. (optional) Merge several sessions into one benchmark
If you ran the datasets in groups (`results_A`, `results_B`, `results_C`), upload the three folders to
the same session and merge them: the statistics (Friedman, Nemenyi, Wilcoxon-Holm, LODO) are then computed over all datasets.
"""),
    code("""
# !python tools/merge_results.py --out results_all results_A results_B results_C
"""),
    md("""
## 7. Export
`results/tab_sse.tex` and `results/tab_agg.tex` are ready-to-`\\input` LaTeX tables that replace the
empty `Tables 2–3` of the paper (bold = best, † = significantly worse than DRP-AP, Wilcoxon–Holm).
"""),
    code("""
!zip -qr results.zip results
try:
    from google.colab import files
    files.download("results.zip")
except ImportError:
    pass
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "colab": {"provenance": []}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "DRP_AP_Colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, f"({len(cells)} cells)")
