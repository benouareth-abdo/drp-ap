"""Generate DRP_AP_Improve_Colab.ipynb (five improvements of DRP-AP-SA vs greedy k-means++ restarts)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["drp_ap/__init__.py", "drp_ap/core.py", "drp_ap/baselines.py", "drp_ap/metrics.py",
         "drp_ap/stats.py", "drp_ap/synthetic.py", "drp_ap/data.py", "drp_ap/experiments.py",
         "drp_ap/reporting.py", "drp_ap/tuning.py", "drp_ap/diagnostics.py", "drp_ap/fuzzy.py",
         "drp_ap/improve.py", "run_improve.py", "tests/test_drp_ap.py"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [md("""
# Five improvements of DRP-AP-SA vs greedy k-means++ best-of-10 (Colab)

| id | idea |
|----|------|
| I1 | hybrid pool: DRP-AP-SA + greedy k-means++ restarts (3 and 10 Lloyd runs) |
| I2 | SA-opt: optimal 1-D threshold instead of the median |
| I1+I2 | SA-opt anchor + 9 greedy restarts |
| I3 | lookahead: split chosen by the SSE after a few global Lloyd iterations |
| I4 | lookahead + D² seeded candidates |
| I5 | successive halving over 20 greedy k-means++ initialisations |

*Runtime → Run all.* Resumable: if Colab disconnects, re-run the last run cell.
"""),
         code("!pip -q install diptest pillow\n!mkdir -p drp_ap tests results_improve data_cache"),
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
    code("!python run_improve.py --datasets Iris Wine Digits --seeds 2 --max-n 2000 --out results_improve_smoke"),
    md("""## 4. Full run on the 30 benchmark datasets
`--seeds 10` (use 30 for the paper protocol), `--max-n 10000` (subsample of large datasets).
Optional: upload `r8-train-no-stop.txt` and `r8-test-no-stop.txt` to `data_cache/` first (else 29 datasets)."""),
    code("!python run_improve.py --datasets benchmark30 --seeds 10 --max-n 10000 --out results_improve"),
    md("## 5. Download (send me this zip)"),
    code("""
!zip -qr results_improve.zip results_improve
from google.colab import files
files.download("results_improve.zip")
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "DRP_AP_Improve_Colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
