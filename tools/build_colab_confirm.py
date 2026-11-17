"""Generate DRP_AP_Confirm_Colab.ipynb (five improvements of DRP-AP-SA vs greedy k-means++ restarts)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["drp_ap/__init__.py", "drp_ap/core.py", "drp_ap/baselines.py", "drp_ap/metrics.py",
         "drp_ap/stats.py", "drp_ap/synthetic.py", "drp_ap/data.py", "drp_ap/experiments.py",
         "drp_ap/reporting.py", "drp_ap/tuning.py", "drp_ap/diagnostics.py", "drp_ap/fuzzy.py",
         "drp_ap/improve.py", "run_improve.py", "run_confirm.py", "tests/test_drp_ap.py"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [md("""
# Out-of-sample confirmation of I2 (SA-opt) on datasets not used before

Pre-registered hypotheses (fixed before the run, seeds disjoint from earlier runs):

* **H1** I2 has lower NSSE than DRP-AP-SA (one-sided Wilcoxon over datasets)
* **H2** I2 is not worse than greedy k-means++ best-of-3 (two-sided Wilcoxon + bootstrap CI)
* **H3** I1+I2 (SA-opt + greedy best-of-9) vs greedy best-of-10 (two-sided + bootstrap CI)
* **H4** I2 is worse than greedy best-of-10 (one-sided)

17 candidate OpenML datasets are listed in `drp_ap/data.py` (`CONFIRM`); those that cannot be
downloaded are reported and skipped. *Runtime → Run all.* Resumable.
"""),
         code("!pip -q install diptest pillow\n!mkdir -p drp_ap tests results_confirm data_cache"),
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
    md("## 3. Smoke test (≈ 1 min)"),
    code("!python run_confirm.py --datasets Iris Wine --seeds 2 --max-n 1000 --out results_confirm_smoke"),
    md("## 4. Confirmation run (17 candidate datasets, 30 seeds)"),
    code("!python run_confirm.py --datasets confirm --seeds 30 --max-n 10000 --out results_confirm"),
    md("## 5. Download (send me this zip)"),
    code("""
!zip -qr results_confirm.zip results_confirm
from google.colab import files
files.download("results_confirm.zip")
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "DRP_AP_Confirm_Colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
