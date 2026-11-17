"""Merge several run_experiments outputs (e.g. one Colab session per group of datasets) into ONE
benchmark: concatenates real_raw.csv, rebuilds the tables, the Friedman/Nemenyi/Wilcoxon-Holm
statistics, the leave-one-dataset-out analysis, and the dataset manifest.

    python tools/merge_results.py --out results_all results_part1 results_part2 results_part3
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drp_ap import reporting as R          # noqa: E402
from drp_ap import tuning as TU            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results_all")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = [pd.read_csv(os.path.join(d, "real_raw.csv")) for d in a.dirs]
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(["dataset", "method", "seed"]).sum()
    if dup:
        print(f"warning: {dup} duplicated (dataset, method, seed) rows -> keeping the last")
        df = df.drop_duplicates(["dataset", "method", "seed"], keep="last")
    # a method missing on some datasets would make Friedman drop it: report it
    cnt = df.groupby("method")["dataset"].nunique()
    full = cnt.max()
    if (cnt < full).any():
        print("methods not run on every dataset (dropped from the rank statistics):",
              cnt[cnt < full].to_dict())
    df.to_csv(os.path.join(a.out, "real_raw.csv"), index=False)
    for d in a.dirs:
        m = os.path.join(d, "datasets_manifest.csv")
        if os.path.exists(m):
            pd.read_csv(m).to_csv(os.path.join(a.out, f"datasets_manifest_{os.path.basename(d.rstrip('/'))}.csv"),
                                  index=False)
    R.sse_table(df, path=os.path.join(a.out, "tab_sse.tex"))
    agg = R.aggregate_table(df, path=os.path.join(a.out, "tab_agg.tex"))
    agg.to_csv(os.path.join(a.out, "aggregate.csv"))
    stat = R.statistical_summary(df, a.out)
    print(f"{df.dataset.nunique()} datasets, {df.method.nunique()} methods, {df.seed.nunique()} seeds")
    print(agg.round(4).to_string())
    if "friedman_p" in stat:
        print(f"\nFriedman chi2={stat['friedman_stat']:.2f} p={stat['friedman_p']:.3g} CD={stat['cd']:.2f}")
        print(stat["ranks"].round(2).to_string())
    pool = [c for c in ("DRP-AP-SA", "DRP-AP-SA (T=0)", "DRP-AP-SA (mean split)") if c in set(df.method)]
    if len(pool) >= 2 and df.dataset.nunique() >= 6:
        t, s = TU.lodo(df, default="DRP-AP-SA", pool=pool, ref="PCA-Part")
        t.to_csv(os.path.join(a.out, "lodo_table.csv"))
        json.dump(s, open(os.path.join(a.out, "lodo_summary.json"), "w"), indent=1)
        print("\nLODO:", s)
    print("written to", a.out)


if __name__ == "__main__":
    main()
