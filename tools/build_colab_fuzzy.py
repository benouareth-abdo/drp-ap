"""Generate DRP_AP_Fuzzy_Colab.ipynb: self-contained notebook for the fuzzy benchmark.
Usage: python tools/build_colab_fuzzy.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["drp_ap/__init__.py", "drp_ap/core.py", "drp_ap/baselines.py", "drp_ap/metrics.py",
         "drp_ap/stats.py", "drp_ap/synthetic.py", "drp_ap/data.py", "drp_ap/experiments.py",
         "drp_ap/reporting.py", "drp_ap/tuning.py", "drp_ap/diagnostics.py", "drp_ap/fuzzy.py",
         "drp_ap/fuzzy_experiments.py", "run_fuzzy.py", "tests/test_drp_ap.py"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [md("""
# F-DRP-AP-SA — fuzzy benchmark on 30 datasets (Colab)

Compares initializers of **fuzzy c-means** (FCM): Random, k-means++, greedy k-means++, PCA-Part,
DRP-AP-SA, F-DRP-AP-SA (T=0 and T=5) and equal-budget FCM restarts (best-of-3 / best-of-10),
for fuzzifiers m = 1.5 and m = 2. *Runtime → Run all.* The run is **resumable**: if Colab
disconnects, re-run the last cell and finished datasets are skipped.
"""),
         code("!pip -q install diptest pillow\n!mkdir -p drp_ap tests tools results_fuzzy data_cache"),
         md("## 1. Package source")]
for f in FILES:
    cells.append(code(f"%%writefile {f}\n" + open(os.path.join(ROOT, f)).read()))
cells += [
    code("""
import os
need = %s
missing = [f for f in need if not os.path.exists(f)]
assert not missing, "Missing files: %%s -> re-run the writefile cells" %% missing
print("OK:", len(need), "files")
""" % repr(FILES)),
    md("## 2. Self-check"),
    code("!python -m pytest -q tests"),
    md("""## 3. Optional: R8 (manual files)
If you have `r8-train-no-stop.txt` and `r8-test-no-stop.txt`, upload them to `data_cache/` first
(otherwise R8 is skipped; 29 datasets run)."""),
    md("## 4. Smoke test (≈ 1 min)"),
    code("!python run_fuzzy.py --datasets Iris Wine --seeds 3 --max-n 1000 --out results_fuzzy_smoke"),
    md("""## 5. Full run on the 30 benchmark datasets
`--max-n 5000` subsamples large datasets (FCM is expensive); `--seeds 10`. Expect roughly
30–90 min on a free Colab CPU. Lower `--max-n` (e.g. 3000) if it is too slow."""),
    code("!python run_fuzzy.py --datasets benchmark30 --seeds 10 --max-n 5000 --out results_fuzzy"),
    md("## 6. Package and download the results (send me this zip)"),
    code("""
!zip -qr results_fuzzy.zip results_fuzzy
from google.colab import files
files.download("results_fuzzy.zip")
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "DRP_AP_Fuzzy_Colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
