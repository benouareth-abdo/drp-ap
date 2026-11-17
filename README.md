# drp-ap

Code for the paper **When Can PCA-Part and Mahalanobis Seeding Fail? Diagnosing Directional-Rank Deficiency in k-means
Initialization** (*Abdallah BENOUARETH*, Université Badji Mokhtar-Annaba).

* **DRP-AP**: *Directional-Rank-Profile Adaptive Partitioning*, a divisive initializer that switches between
  partitioning modes according to the measured directional rank.
* **DRP-AP-SA**: DRP-AP with *SSE-Arbitrated* splitting; a deterministic Euclidean initializer that chooses the
  split axis among the leading principal axes by the SSE each split removes.
* A reproducible **equal-budget protocol**: 24 initializers, 30 real datasets, two synthetic suites, ablations,
  leave-one-dataset-out tuning, and a pre-registered confirmation on 17 further OpenML datasets.

## What the results say
DRP-AP-SA is statistically indistinguishable from PCA-Part (Wilcoxon–Holm p = 1.0) and better than the
Mahalanobis seeders. At equal Lloyd budget, restarts of greedy k-means++ reach the lowest SSE (best-of-10:
Friedman rank 3.13 vs 12.53 for DRP-AP-SA). Axis arbitration helps on the generator built from the theorem, not
on unrelated generators; no pre-split diagnostic predicts where it helps. See the paper in `paper/`.

## Install
```bash
git clone https://github.com/[user]/drp-ap.git && cd drp-ap
pip install -e ".[dev]"        # add ,dip for the optional Hartigan dip test
python -m pytest -q tests
```

## Use
```python
from drp_ap import DRPAPSA, DRPAP
from sklearn.cluster import KMeans

C = DRPAPSA().fit_init(X, K)                  # deterministic DRP-AP-SA (T=5, top-3 axes)
C = DRPAP(delta=0.05, kappa0=1e3, shrinkage="lw", random_state=0).fit_init(X, K)   # original design
labels = KMeans(K, init=C, n_init=1).fit_predict(X)
```

## Reproduce the experiments
```bash
python run_experiments.py --quick --all-extras                      # smoke run, offline datasets
python run_experiments.py --datasets all --seeds 30 --all-extras    # main benchmark (30 datasets)
python run_fuzzy.py                                                 # fuzzy variant (F-DRP-AP-SA)
python run_improve.py                                               # restart-inspired variants I1-I5
python run_confirm.py                                               # out-of-sample confirmation (17 datasets)
```
Runs are resumable. Datasets are downloaded on first use (OpenML / scikit-learn) into `data_cache/`
(override with `DRP_AP_DATA`). Self-contained **Google Colab** notebooks are in `notebooks/`; regenerate them
with `python tools/build_colab*.py`.

## Layout
| path | content |
|---|---|
| `drp_ap/core.py` | DRP-AP and DRP-AP-SA |
| `drp_ap/baselines.py` | Random, k-means++, greedy k-means++, KKZ, PCA-Part, Var-Part, Bradley–Fayyad, k-means++ MD, GMM-init |
| `drp_ap/improve.py` | variants I1–I5 (hybrid pool, optimal threshold, lookahead, D² candidates, successive halving) |
| `drp_ap/fuzzy.py`, `fuzzy_experiments.py` | F-DRP-AP-SA and fuzzy c-means experiments |
| `drp_ap/data.py`, `synthetic.py` | real datasets (30 + 17 confirmation) and synthetic generators |
| `drp_ap/metrics.py`, `stats.py`, `diagnostics.py`, `tuning.py` | metrics, Friedman/Nemenyi/Wilcoxon–Holm, diagnostics, LODO tuning |
| `results/` | CSV outputs of the runs reported in the paper (main, fuzzy, improve, confirm)  | LaTeX source, figures and tables of the manuscript, highlights |

## Citation
See `CITATION.cff`
## License
MIT.
