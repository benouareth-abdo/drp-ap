"""Test the five improvements of DRP-AP-SA against greedy k-means++ best-of-3/10 (30 datasets).
Usage:  python run_improve.py --datasets benchmark30 --seeds 10 --max-n 10000
Resumable: finished datasets are stored in <out>/improve_raw.csv and skipped on re-run."""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from drp_ap.data import ALL_NAMES, DEFAULT_DIR, load_many, manifest
from drp_ap.experiments import evaluate
from drp_ap.improve import improve_methods

warnings.filterwarnings("ignore")
REFS = ["DRP-AP-SA", "Greedy k-means++ best-of-3", "Greedy k-means++ best-of-10"]


def holm(p):
    order = sorted(p, key=p.get)
    out, prev = {}, 0.0
    for i, k in enumerate(order):
        prev = max(prev, min(1.0, p[k] * (len(order) - i)))
        out[k] = prev
    return out


def summarize(df):
    df = df.copy()
    df["nsse"] = df.sse / df.groupby("dataset").sse.transform("min")
    pv = df.groupby(["dataset", "method"]).nsse.mean().unstack()
    agg = pd.DataFrame({"NSSE": pv.mean(), "fail>1.02": (pv > 1.02).mean(),
                        "time_s": df.groupby("method").t_init.mean(),
                        "ARI": df.groupby("method").ari.mean()})
    agg["rank"] = pd.Series(rankdata(pv.values, axis=1).mean(0), index=pv.columns)
    for ref in REFS:
        if ref not in pv:
            continue
        w, t, l, ps = {}, {}, {}, {}
        for b in pv.columns:
            if b == ref:
                continue
            r = pv[b] / pv[ref]
            w[b], t[b], l[b] = int((r < .99).sum()), int(((r >= .99) & (r <= 1.01)).sum()), int((r > 1.01).sum())
            try:
                ps[b] = wilcoxon(pv[b], pv[ref]).pvalue
            except ValueError:
                ps[b] = 1.0
        h = holm(ps)
        tag = ref.replace("Greedy k-means++ ", "G").replace("DRP-AP-SA", "SA")
        agg[f"W/T/L vs {tag}"] = pd.Series({b: f"{w[b]}/{t[b]}/{l[b]}" for b in w})
        agg[f"Holm vs {tag}"] = pd.Series(h)
    return agg.sort_values("rank"), pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["benchmark30"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-n", type=int, default=10000)
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default="results_improve")
    a = ap.parse_args()
    names = ALL_NAMES if a.datasets == ["benchmark30"] else a.datasets
    os.makedirs(a.out, exist_ok=True)
    data = load_many(names, a.data_dir, a.max_n)
    manifest(data).to_csv(os.path.join(a.out, "datasets_manifest.csv"), index=False)
    print(f"=== Loaded {len(data)}/{len(names)} datasets ===")
    raw = os.path.join(a.out, "improve_raw.csv")
    done = pd.read_csv(raw) if os.path.exists(raw) else pd.DataFrame()
    methods = improve_methods()
    t0 = time.time()
    for name, (X, y, K) in data.items():
        if len(done) and (done.dataset == name).any():
            continue
        rows = evaluate(name, X, y, K, methods, a.seeds, verbose=False, with_sil=False)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(raw, index=False)
        print(f"[improve] {name:15s} n={len(X):6d} K={K:3d} done ({time.time() - t0:6.0f} s)", flush=True)
    agg, pv = summarize(done)
    agg.to_csv(os.path.join(a.out, "improve_summary.csv"))
    pv.to_csv(os.path.join(a.out, "improve_per_dataset_nsse.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== Summary (NSSE: mean normalised SSE; rank: Friedman average rank, lower is better) ===")
    print(agg.round(4).to_string())
    if pv.shape[1] > 2:
        print("\nFriedman:", friedmanchisquare(*[pv[c] for c in pv.columns]))
    print("\n=== Per-dataset mean NSSE (key methods) ===")
    key = [c for c in pv.columns if c in REFS or c.startswith("I")]
    print(pv[key].round(3).to_string())


if __name__ == "__main__":
    main()
