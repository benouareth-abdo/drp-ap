"""Turn raw result tables into the tables / figures / statistics of Section 'Results'."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import stats as S

ORDER_REAL = ["Iris", "Wine", "BC", "Digits", "MNIST", "F-MNIST", "COIL-20", "Olivetti",
              "20NG", "R8", "Isolet", "HAR"]
HEADER = {"Iris": "Iris", "Wine": "Wine", "BC": "BC", "Digits": "Digits", "MNIST": "MNIST",
          "F-MNIST": "F-MNIST", "COIL-20": "COIL", "Olivetti": "Olivetti", "20NG": "20NG",
          "R8": "R8", "Isolet": "Isolet", "HAR": "HAR"}


def _tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("+", r"$^{+}$") \
        if "+" in str(s) and "k-means" not in str(s) else str(s).replace("_", r"\_").replace("&", r"\&")


def sse_table(df, ref="DRP-AP", path=None):
    """Table 'tab:sse' of the paper: mean +- std of normalised SSE, bold best, dagger = significantly
    worse than `ref` (paired Wilcoxon over seeds, Holm across methods)."""
    from scipy import stats
    datasets = [d for d in ORDER_REAL if d in df.dataset.unique()] + \
               [d for d in df.dataset.unique() if d not in ORDER_REAL]
    methods = list(df.method.unique())
    mean = df.groupby(["method", "dataset"])["nsse"].mean().unstack()
    std = df.groupby(["method", "dataset"])["nsse"].std().unstack()
    sig = pd.DataFrame(False, index=methods, columns=datasets)
    for d in datasets:
        p = {}
        sub = df[df.dataset == d]
        a = sub[sub.method == ref].sort_values("seed")["nsse"].values
        for m in methods:
            if m == ref:
                continue
            b = sub[sub.method == m].sort_values("seed")["nsse"].values
            if len(a) == len(b) and len(a) > 5 and not np.allclose(a, b) and b.mean() > a.mean():
                try:
                    p[m] = stats.wilcoxon(b, a, alternative="greater").pvalue
                except ValueError:
                    p[m] = 1.0
        adj = S.holm(p) if p else {}
        for m, v in adj.items():
            sig.loc[m, d] = v < 0.05
    lines = [r"\begin{tabular}{l" + "c" * len(datasets) + "}", r"\toprule",
             "Method & " + " & ".join(HEADER.get(d, d) for d in datasets) + r" \\", r"\midrule"]
    for m in methods:
        cells = []
        for d in datasets:
            if d not in mean.columns or pd.isna(mean.loc[m, d]):
                cells.append("--")
                continue
            c = f"{mean.loc[m, d]:.3f}" + (f"$\\pm${std.loc[m, d]:.3f}" if not pd.isna(std.loc[m, d]) else "")
            if np.isclose(mean.loc[m, d], mean[d].min(), atol=5e-4):
                c = r"\textbf{" + c + "}"
            if sig.loc[m, d]:
                c += r"$^\dagger$"
            cells.append(c)
        lines.append(_tex(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    if path:
        open(path, "w").write(tex)
    return mean


def aggregate_table(df, path=None):
    g = df.groupby("method").agg(ARI=("ari", "mean"), NMI=("nmi", "mean"),
                                 Failure=("fail", "mean"), Fail2pct=("fail2", "mean"), Iterations=("n_iter", "mean"),
                                 Time_s=("t_init", "mean"), NSSE=("nsse", "mean"))
    if path:
        lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
                 r"Method & NSSE & ARI & NMI & Fail$>$1.5 & Fail$>$1.02 & Iterations & Init time (s) \\", r"\midrule"]
        for m, r in g.iterrows():
            lines.append(f"{_tex(m)} & {r.NSSE:.3f} & {r.ARI:.3f} & {r.NMI:.3f} & {r.Failure:.3f} & {r.Fail2pct:.3f} "
                         f"& {r.Iterations:.1f} & {r.Time_s:.3f} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(path, "w").write("\n".join(lines))
    return g


def statistical_summary(df, out_dir, ref="DRP-AP"):
    """Friedman + Nemenyi (critical-difference diagram) + Wilcoxon-Holm vs reference."""
    pivot = df.groupby(["dataset", "method"])["nsse"].mean().unstack().dropna(axis=1)
    res = {}
    if pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
        stat, p = S.friedman(pivot)
        ranks = S.average_ranks(pivot)
        cd = S.nemenyi_cd(pivot.shape[1], pivot.shape[0])
        S.cd_diagram(ranks, cd, os.path.join(out_dir, "cd_diagram.png"),
                     f"Average ranks of normalised SSE (Friedman p={p:.3g})")
        ranks.to_csv(os.path.join(out_dir, "average_ranks.csv"), header=["avg_rank"])
        res.update(friedman_stat=stat, friedman_p=p, cd=cd, ranks=ranks)
    for r_ in dict.fromkeys([ref, "DRP-AP-SA", "Portfolio {PCA,Var,SA}"]):
        if r_ in pivot.columns:
            w = S.wilcoxon_vs_ref(pivot, r_)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in r_)
            w.to_csv(os.path.join(out_dir, f"wilcoxon_holm_vs_{safe}.csv"))
            res["wilcoxon" if r_ == ref else f"wilcoxon::{r_}"] = w
    return res


def mode_correlation(modes_df, df, base="PCA-Part", ref="DRP-AP"):
    """P6: Spearman correlation between fraction of B/C splits and SSE gain over PCA-Part."""
    from scipy.stats import spearmanr
    m = df.groupby(["dataset", "method"])["nsse"].mean().unstack()
    if base not in m.columns or ref not in m.columns:
        return None          # needs both methods to have been run
    gain = (m[base] - m[ref]).rename("gain")
    j = modes_df.set_index("dataset").join(gain, how="inner").dropna()
    if len(j) < 4 or j["frac_BC"].nunique() < 2 or j["gain"].nunique() < 2:
        return None          # correlation undefined for constant input
    rho, p = spearmanr(j["frac_BC"], j["gain"])
    return dict(spearman_rho=float(rho), p=float(p), n=len(j))


def synthetic_report(df, out_dir):
    """Verdicts for P1-P3 and summary plots."""
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    piv = df.pivot_table(index=["family", "config", "draw"], columns="method", values="nsse")
    summ = df.groupby(["family", "config", "method"])["nsse"].agg(["mean", "std"]).unstack()
    summ.to_csv(os.path.join(out_dir, "synthetic_summary.csv"))

    def wil(cfg_mask, a, b, alt):
        sub = piv[cfg_mask]
        if len(sub) < 6:
            return float("nan")          # too few draws for a meaningful test
        if np.allclose(sub[a], sub[b]):
            return 1.0
        return float(stats.wilcoxon(sub[a], sub[b], alternative=alt).pvalue)

    fam = piv.index.get_level_values("family")
    cfg = piv.index.get_level_values("config")
    cands = [c for c in ("DRP-AP", "DRP-AP-SA", "DRP-AP-SA (T=0)") if c in piv.columns]
    na = lambda p: np.isnan(p)
    for cand in cands:
        # P1: mono-directional: the candidate must not be significantly worse than PCA-Part
        p1 = wil(fam == "mono", cand, "PCA-Part", "greater")
        out.append(f"[{cand}] P1 (mono): p(worse than PCA-Part)={p1:.3g} -> "
                   f"{'n/a (need >= 6 draws)' if na(p1) else ('REFUTED' if p1 < 0.05 else 'not refuted')}")
        # P2: multi-axial above the failure threshold of the construction
        # (a^2 > (2/pi) sigma1^2  <=>  ratio > 2/pi + sigma2^2 ~ 0.73 for sigma2=0.3):
        # the candidate must be significantly better than PCA-Part there.
        for c in sorted(set(cfg[fam == "multi_axial"])):
            ratio = float(c.split("ratio=")[1].split(",")[0]) if "ratio=" in c else 0
            p = wil(cfg == c, cand, "PCA-Part", "less")
            verdict = ("n/a (need >= 6 draws)" if na(p) else
                       "supported" if p < 0.05 else
                       ("NOT supported" if ratio > 0.75 else "n.s. (below threshold)"))
            out.append(f"[{cand}] P2 ({c}): p(better than PCA-Part)={p:.3g} -> {verdict}")
        # P3: SSE on anisotropic data vs K-means++ MD (seed stability is in stability.csv)
        for c in sorted(set(cfg[fam == "anisotropic"])):
            p = wil(cfg == c, cand, "k-means++ MD", "less")
            out.append(f"[{cand}] P3 SSE ({c}): p(better than k-means++ MD)={p:.3g}")
        # small-sample regime (n ~ d): is the candidate WORSE than PCA-Part?
        for c in sorted(set(cfg[fam == "highdim"])):
            p = wil(cfg == c, cand, "PCA-Part", "greater")
            out.append(f"[{cand}] small-sample ({c}): p(worse than PCA-Part)={p:.3g}"
                       + ("  <- significantly worse" if (not na(p) and p < 0.05) else ""))
    open(os.path.join(out_dir, "synthetic_verdicts.txt"), "w").write("\n".join(out))

    # plot: nsse vs lambda2/lambda1 for multi-axial
    sub = df[df.family == "multi_axial"].copy()
    if len(sub):
        sub["r"] = sub.config.str.extract(r"r=(\d+)").astype(int)
        sub["ratio"] = sub.config.str.extract(r"ratio=([\d.]+)").astype(float)
        rs = sorted(sub.r.unique())
        fig, axs = plt.subplots(1, len(rs), figsize=(4.2 * len(rs), 3.4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, r in zip(axs, rs):
            for m, g in sub[sub.r == r].groupby("method"):
                a = g.groupby("ratio")["nsse"].mean()
                ax.plot(a.index, a.values, marker="o", label=m, lw=2.2 if m.startswith("DRP-AP") else 1)
            ax.set_title(f"r = {r}  (K = {2**r})")
            ax.set_xlabel(r"$\lambda_2/\lambda_1$")
        axs[0].set_ylabel("normalised SSE")
        axs[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "synthetic_multiaxial.png"), dpi=150)
        plt.close(fig)
    return out


def save_all(out_dir, **tables):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in tables.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)


def synthetic2_report(df, out_dir):
    """Non-circular synthetic suite: per config, mean NSSE of each method, and Wilcoxon p-values of
    DRP-AP-SA (T=5) and the SA pool against PCA-Part and against Greedy k-means++ best-of-3."""
    from scipy import stats
    piv = df.pivot_table(index=["family", "config", "draw"], columns="method", values="nsse")
    summ = df.groupby(["family", "config", "method"])["nsse"].mean().unstack()
    summ.to_csv(os.path.join(out_dir, "synthetic2_summary.csv"))
    out = []
    cfgs = sorted(set(piv.index.get_level_values("config")))
    fam = piv.index.get_level_values("family")
    cfg = piv.index.get_level_values("config")
    pairs = [("DRP-AP-SA", "PCA-Part"), ("DRP-AP-SA", "PCA-Part (Su-Dy, mean split)"), ("DRP-AP-SA", "PCA-Part + local Lloyd (A3)"),
             ("DRP-AP-SA", "Greedy k-means++"), ("SA pool-3", "Greedy k-means++ best-of-3")]
    rows = []
    for c in cfgs:
        sub = piv[cfg == c]
        for a, b in pairs:
            if a not in sub or b not in sub or len(sub) < 6:
                continue
            d = (sub[a] - sub[b]).to_numpy()
            if np.allclose(d, 0):
                p_less = p_gr = 1.0
            else:
                p_less = float(stats.wilcoxon(sub[a], sub[b], alternative="less").pvalue)
                p_gr = float(stats.wilcoxon(sub[a], sub[b], alternative="greater").pvalue)
            rows.append(dict(config=c, a=a, b=b, mean_diff=float(d.mean()), p_a_better=p_less,
                             p_a_worse=p_gr))
            out.append(f"[{c}] {a} vs {b}: mean dNSSE={d.mean():+.4f}  "
                       f"p(a better)={p_less:.3g}  p(a worse)={p_gr:.3g}")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "synthetic2_tests.csv"), index=False)
    return out
