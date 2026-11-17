#!/usr/bin/env python
"""Run the full experimental protocol of the DRP-AP paper.

Examples
--------
  python run_experiments.py --quick                      # fast smoke run (offline datasets only)
  python run_experiments.py --datasets all --seeds 30    # full protocol on 12 datasets
  python run_experiments.py --datasets Iris Wine Digits --seeds 10 --synthetic --ablation
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

_REQUIRED = ["__init__", "core", "baselines", "metrics", "stats", "synthetic", "data",
             "experiments", "reporting", "tuning", "diagnostics"]
_here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drp_ap")
_missing = [f"drp_ap/{m}.py" for m in _REQUIRED if not os.path.exists(os.path.join(_here, m + ".py"))]
if _missing:
    raise SystemExit("Missing package files: " + ", ".join(_missing) +
                     "\nIn Colab, run every '%%writefile' cell (Runtime > Run all) so that they are "
                     "written to /content/drp_ap/ before launching this script.")

from drp_ap import experiments as E
from drp_ap import reporting as R
from drp_ap.data import ALL_NAMES, DATASETS, DEFAULT_DIR, load_many, manifest

OFFLINE = ["Iris", "Wine", "BC", "Digits"]


def _resumable(path, fn, data, **kw):
    """Run fn(one dataset, **kw) dataset by dataset, appending to `path` after each one and skipping
    the datasets already in the file, so that a crashed / timed-out Colab session can be resumed."""
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    have = set(done["dataset"]) if len(done) else set()
    for name, d in data.items():
        if name in have:
            print(f"[resume] {name}: already in {os.path.basename(path)}")
            continue
        part = fn({name: d}, **kw)
        done = pd.concat([done, part], ignore_index=True)
        done.to_csv(path, index=False)            # checkpoint after every dataset
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"],
                    help="names from %s, 'all' (12 core), 'extended'/'benchmark30' (30) or 'offline'" % ALL_NAMES)
    ap.add_argument("--methods", nargs="+", default=None, help="subset of method names")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--max-n", type=int, default=20000,
                    help="stratified subsample cap per dataset (0 = no cap)")
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="few seeds / small grids")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--no-restarts", action="store_true",
                    help="skip the equal-compute 'Greedy k-means++ best-of-3/10' baselines (they add "
                         "3 + 10 full Lloyd runs per seed and dominate the run time)")
    ap.add_argument("--synthetic2", action="store_true",
                    help="second synthetic suite (generators unrelated to the Theorem-2 construction)")
    ap.add_argument("--tune", action="store_true",
                    help="leave-one-dataset-out choice of (T, q_max) on a deterministic grid")
    ap.add_argument("--diagnostics", action="store_true",
                    help="measured directional rank per dataset and correlation with the SA gain")
    ap.add_argument("--skip-main", action="store_true",
                    help="do not re-run the main real benchmark / mode usage (use it to finish or "
                         "resume the extras: --skip-main --ablation --sensitivity)")
    ap.add_argument("--all-extras", action="store_true",
                    help="synthetic + synthetic2 + ablation + sensitivity + stability + tune + diagnostics")
    a = ap.parse_args()

    if a.all_extras:
        a.synthetic = a.ablation = a.sensitivity = a.stability = True
        a.synthetic2 = a.tune = a.diagnostics = True
    if a.quick:
        a.seeds = min(a.seeds, 3)
        if a.datasets == ["all"]:
            a.datasets = ["offline"]
    names = (list(DATASETS) if a.datasets == ["all"] else
             ALL_NAMES if a.datasets in (["extended"], ["benchmark30"]) else
             OFFLINE if a.datasets == ["offline"] else a.datasets)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    data = load_many(names, a.data_dir, a.max_n or None)
    if not data:
        raise SystemExit("No dataset could be loaded.")
    man = manifest(data)
    man.to_csv(os.path.join(a.out, "datasets_manifest.csv"), index=False)
    miss = [n for n in names if n not in data]
    print(f"\n=== Loaded {len(data)}/{len(names)} datasets ===\n", man.to_string(index=False))
    if miss:
        print("NOT LOADED:", miss, "-> add <data-dir>/<name>.npz (arrays X, y) and re-run to include them")

    reg = E.main_methods()
    methods = reg if a.methods is None else {k: reg[k] for k in a.methods}
    if a.no_restarts:
        methods = {k: v for k, v in methods.items() if "best-of" not in k}

    df = None
    if not a.skip_main:
        # ---------------- real benchmarks (Tables sse / agg, statistics) -------------
        df = E.run_real(data, methods, a.seeds)
        df.to_csv(os.path.join(a.out, "real_raw.csv"), index=False)
        R.sse_table(df, path=os.path.join(a.out, "tab_sse.tex"))
        agg = R.aggregate_table(df, path=os.path.join(a.out, "tab_agg.tex"))
        agg.to_csv(os.path.join(a.out, "aggregate.csv"))
        stat = R.statistical_summary(df, a.out)
        print("\n=== Aggregate (mean over datasets and seeds) ===\n", agg.round(4).to_string())
        if "friedman_p" in stat:
            print(f"\nFriedman chi2={stat['friedman_stat']:.2f}  p={stat['friedman_p']:.3g}  "
                  f"Nemenyi CD={stat['cd']:.2f}\n", stat["ranks"].round(2).to_string())
        for k_, v_ in stat.items():
            if k_ == "wilcoxon" or k_.startswith("wilcoxon::"):
                who = "DRP-AP" if k_ == "wilcoxon" else k_.split("::", 1)[1]
                print(f"\nWilcoxon-Holm vs {who} (normalised SSE across datasets; diff>0 => other worse)\n",
                      v_.round(4).to_string())

        # ---------------- mode usage + P6 -------------------------------------------
        modes = E.run_modes(data)
        modes.to_csv(os.path.join(a.out, "mode_usage.csv"), index=False)
        print("\n=== Mode usage (fraction of splits) ===\n", modes.round(3).to_string(index=False))
        corr = R.mode_correlation(modes, df)
        if corr:
            print("P6 Spearman(frac B/C, gain over PCA-Part):", corr)
            json.dump(corr, open(os.path.join(a.out, "p6_correlation.json"), "w"))

    # ---------------- optional extras ---------------------------------------------
    if a.stability:
        stab_methods = {k: reg[k] for k in ["k-means++", "PCA-Part", "Var-Part", "k-means++ MD",
                                            "k-means++ MD Hybrid", "DRP-AP", "DRP-AP-SA", "DRP-AP-SA (T=0)"]}
        st = E.run_stability(data, stab_methods, n_boot=8 if a.quick else 15)
        st.to_csv(os.path.join(a.out, "stability.csv"), index=False)
        print("\n=== Seed stability (lower = better) ===\n",
              st.pivot(index="dataset", columns="method", values="stability").round(4).to_string())
    if a.synthetic:
        sdf = E.run_synthetic(n_draws=5 if a.quick else 30, quick=a.quick)
        sdf.to_csv(os.path.join(a.out, "synthetic_raw.csv"), index=False)
        print("\n=== Synthetic verdicts ===")
        print("\n".join(R.synthetic_report(sdf, a.out)))
    if a.synthetic2:
        s2 = E.run_synthetic2(n_draws=5 if a.quick else 30, quick=a.quick)
        s2.to_csv(os.path.join(a.out, "synthetic2_raw.csv"), index=False)
        print("\n=== Synthetic suite 2 (non-Theorem-2 generators) ===")
        print("\n".join(R.synthetic2_report(s2, a.out)))
    if a.tune:
        from drp_ap import tuning as TU
        gr = TU.run_grid(data)
        gr.to_csv(os.path.join(a.out, "tuning_grid_raw.csv"), index=False)
        lt, ls = TU.lodo(gr)
        lt.to_csv(os.path.join(a.out, "lodo_table.csv"))
        json.dump(ls, open(os.path.join(a.out, "lodo_summary.json"), "w"), indent=1)
        print("\n=== Leave-one-dataset-out choice of (T, q_max) ===\n", lt.round(4).to_string(),
              "\n", ls)
    if a.diagnostics:
        from drp_ap import diagnostics as DG
        dg = DG.run_diagnostics(data)
        dg.to_csv(os.path.join(a.out, "diagnostics.csv"), index=False)
        print("\n=== Measured directional rank and SA gain over PCA-Part ===\n", dg.round(3).to_string(index=False))
        print("Spearman (rho, p):", DG.correlations(dg))
        json.dump(DG.correlations(dg), open(os.path.join(a.out, "diagnostics_corr.json"), "w"))
    if a.ablation:
        ab = _resumable(os.path.join(a.out, "ablation_raw.csv"), E.run_ablation, data,
                        n_seeds=3 if a.quick else 10)
        tab = ab.groupby(["dataset", "method"])["rel_sse"].mean().unstack()
        tab.to_csv(os.path.join(a.out, "ablation_table.csv"))
        print("\n=== Ablations (SSE relative to full DRP-AP; <1 is better) ===\n",
              tab.round(4).to_string())
    if a.sensitivity:
        se = _resumable(os.path.join(a.out, "sensitivity_raw.csv"), E.run_sensitivity, data,
                        n_seeds=2 if a.quick else 5,
                        deltas=(0.05, 0.2) if a.quick else (0.01, 0.05, 0.1, 0.2),
                        kappas=(1e3, 1e6) if a.quick else (1e2, 1e3, 1e4, 1e6),
                        Ts=(0, 5) if a.quick else (0, 5, 10))
        print("\n=== Sensitivity: mean normalised SSE by delta / kappa0 / T ===")
        for k in ("delta", "kappa0", "T"):
            print(se.groupby(k)["nsse"].mean().round(4).to_string(), "\n")

    print(f"Done in {time.time() - t0:.0f}s. Results in ./{a.out}/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
