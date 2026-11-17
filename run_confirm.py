"""Out-of-sample confirmation of I2 (SA-opt) and the hybrid pool on datasets NOT used before.

Pre-registered hypotheses (fixed before the run; NSSE = SSE / best SSE on the dataset):
  H1  I2 (SA-opt) has lower NSSE than DRP-AP-SA            (one-sided Wilcoxon over datasets)
  H2  I2 is not worse than greedy k-means++ best-of-3       (two-sided Wilcoxon + bootstrap CI)
  H3  I1+I2 (SA-opt + greedy best-of-9) vs greedy best-of-10 (two-sided Wilcoxon + bootstrap CI)
  H4  I2 is worse than greedy best-of-10                    (one-sided Wilcoxon)
Seeds are disjoint from the earlier runs (offset 1000).
Usage: python run_confirm.py --datasets confirm --seeds 30 --max-n 10000
Resumable: finished datasets are stored in <out>/confirm_raw.csv."""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from drp_ap.data import CONFIRM, DEFAULT_DIR, load_many, manifest
from drp_ap.experiments import evaluate
from drp_ap.improve import improve_methods

warnings.filterwarnings("ignore")
KEEP = ["PCA-Part (Su-Dy, mean split)", "DRP-AP-SA", "I2: SA-opt (optimal threshold)",
        "I1+I2: SA-opt + greedy best-of-9", "Greedy k-means++", "Greedy k-means++ best-of-3",
        "Greedy k-means++ best-of-10"]
SA, I2, I12 = KEEP[1], KEEP[2], KEEP[3]
G3, G10 = KEEP[5], KEEP[6]


def boot_ci(diff, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff)
    m = rng.choice(d, (B, len(d))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def wtl(r):
    return int((r < .99).sum()), int(((r >= .99) & (r <= 1.01)).sum()), int((r > 1.01).sum())


def pval(a, b, alt):
    try:
        return wilcoxon(a, b, alternative=alt).pvalue
    except ValueError:
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["confirm"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--max-n", type=int, default=10000)
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default="results_confirm")
    a = ap.parse_args()
    names = list(CONFIRM) if a.datasets == ["confirm"] else a.datasets
    os.makedirs(a.out, exist_ok=True)
    data = load_many(names, a.data_dir, a.max_n, seed=a.seed_offset)
    manifest(data).to_csv(os.path.join(a.out, "datasets_manifest.csv"), index=False)
    print(f"=== Loaded {len(data)}/{len(names)} datasets ===")
    missing = [n for n in names if n not in data]
    if missing:
        print("NOT LOADED:", missing)
    off = a.seed_offset
    base = improve_methods()
    methods = {k: ((lambda f: (lambda X, K, s: f(X, K, s + off)))(base[k][0]), base[k][1]) for k in KEEP}
    raw = os.path.join(a.out, "confirm_raw.csv")
    done = pd.read_csv(raw) if os.path.exists(raw) else pd.DataFrame()
    t0 = time.time()
    for name, (X, y, K) in data.items():
        if len(done) and (done.dataset == name).any():
            continue
        rows = evaluate(name, X, y, K, methods, a.seeds, verbose=False, with_sil=False)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(raw, index=False)
        print(f"[confirm] {name:28s} n={len(X):6d} d={X.shape[1]:3d} K={K:3d} ({time.time() - t0:5.0f} s)", flush=True)
    done["nsse"] = done.sse / done.groupby("dataset").sse.transform("min")
    pv = done.groupby(["dataset", "method"]).nsse.mean().unstack()[KEEP]
    pv.to_csv(os.path.join(a.out, "confirm_per_dataset_nsse.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    agg = pd.DataFrame({"NSSE": pv.mean(), "time_s": done.groupby("method").t_init.mean(),
                        "ARI": done.groupby("method").ari.mean()}).loc[KEEP]
    print(f"\n=== {len(pv)} confirmation datasets: mean NSSE ===")
    print(agg.round(4).to_string())
    print("\n=== Per-dataset NSSE ===")
    print(pv.round(3).to_string())
    rows = []
    for tag, x, y, alt in [("H1: I2 < DRP-AP-SA", I2, SA, "less"), ("H2: I2 vs G-3", I2, G3, "two-sided"),
                           ("H3: I1+I2 vs G-10", I12, G10, "two-sided"), ("H4: I2 > G-10", I2, G10, "greater"),
                           ("(extra) I2 vs PCA-Part Su-Dy", I2, KEEP[0], "two-sided")]:
        d = pv[x] - pv[y]
        lo, hi = boot_ci(d)
        w, t, l = wtl(pv[x] / pv[y])
        rows.append(dict(test=tag, W_T_L=f"{w}/{t}/{l}", mean_diff=d.mean(), ci95=f"[{lo:.4f}, {hi:.4f}]",
                         p=pval(pv[x], pv[y], alt)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "confirm_tests.csv"), index=False)
    print("\n=== Pre-registered tests (W/T/L: first method better/equal/worse at 1%) ===")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
