"""Fuzzy benchmark: F-DRP-AP-SA as an initializer for fuzzy c-means on the 30 datasets.

Usage (Colab):  python run_fuzzy.py --datasets benchmark30 --seeds 10 --max-n 5000
Resumable: finished (m, dataset) pairs are stored in <out>/fuzzy_raw.csv and skipped on re-run."""
import argparse
import os
import time
import warnings

import pandas as pd

from drp_ap.data import ALL_NAMES, DEFAULT_DIR, load_many, manifest
from drp_ap.fuzzy_experiments import run_fuzzy, summarize

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["benchmark30"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-n", type=int, default=5000)
    ap.add_argument("--m", nargs="+", type=float, default=[1.5, 2.0])
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default="results_fuzzy")
    a = ap.parse_args()
    names = ALL_NAMES if a.datasets == ["benchmark30"] else a.datasets
    os.makedirs(a.out, exist_ok=True)
    data = load_many(names, a.data_dir, a.max_n)
    manifest(data).to_csv(os.path.join(a.out, "datasets_manifest.csv"), index=False)
    print(f"=== Loaded {len(data)}/{len(names)} datasets ===")
    raw = os.path.join(a.out, "fuzzy_raw.csv")
    done = pd.read_csv(raw) if os.path.exists(raw) else pd.DataFrame()
    t0 = time.time()
    for m in a.m:
        for name, d in data.items():
            if len(done) and ((done.m == m) & (done.dataset == name)).any():
                continue
            df = run_fuzzy({name: d}, m=m, n_seeds=a.seeds, verbose=False)
            df["m"] = m
            done = pd.concat([done, df], ignore_index=True)
            done.to_csv(raw, index=False)
            print(f"[fuzzy m={m}] {name:15s} done  ({time.time() - t0:6.0f} s)", flush=True)
    # normalise per (m, dataset) over everything that was run
    done["nJ"] = done.J / done.groupby(["m", "dataset"]).J.transform("min")
    done["nSSE"] = done.km_sse / done.groupby(["m", "dataset"]).km_sse.transform("min")
    done.to_csv(raw, index=False)
    s = summarize(done)
    s.to_csv(os.path.join(a.out, "fuzzy_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    for m, g in s.groupby("m"):
        print(f"\n=== m = {m} ===")
        print(g.drop(columns="m").sort_values("rank_J").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
