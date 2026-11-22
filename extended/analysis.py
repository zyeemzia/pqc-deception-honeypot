"""Statistics, LaTeX tables and figures for the journal paper.

Input : extended/results/episodes.csv (from evaluate_all.py)
Output: extended/results/summary.csv          mean + bootstrap 95% CI per cell
        extended/results/tests.csv            pairwise tests vs reference policy
        extended/results/table_main.tex       main results table (eval_env=adaptive)
        extended/results/table_cross.tex      robustness matrix (return)
        extended/results/fig_cross_env.pdf    return per eval env, per policy

Statistics (per-seed means are the unit of analysis, so seeds are not
pseudo-replicated by episodes):
  - 95% CI: percentile bootstrap over episodes, 10,000 resamples
  - Two-sided Mann-Whitney U on episode returns vs the reference policy,
    Holm-Bonferroni corrected across all comparisons in one table
  - Effect size: Cliff's delta (|d|<0.147 negligible, <0.33 small, <0.474 medium)

Usage: python extended/analysis.py [--ref static_rule] [--csv path]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METRICS = ["ret", "recon_deflection_rate", "downgrade_capture_rate",
           "legit_fp_rate", "flood_on_real_rate", "engaged_steps"]
LABEL = dict(ret="Return", recon_deflection_rate="Recon defl.",
             downgrade_capture_rate="Downgr. capt.", legit_fp_rate="Legit FP",
             flood_on_real_rate="Flood on real", engaged_steps="Engaged steps")


def bootstrap_ci(x, n=10_000, alpha=0.05, rng=None):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def label(row):
    return row.policy if row.train_env == "-" else f"{row.policy}@{row.train_env}"


def summarise(d):
    rng = np.random.default_rng(0)
    out = []
    for (agent, env), g in d.groupby(["agent", "eval_env"]):
        rec = dict(agent=agent, eval_env=env, n=len(g))
        for m in METRICS:
            if m in g:
                lo, hi = bootstrap_ci(g[m], rng=rng)
                rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = g[m].mean(), lo, hi
        out.append(rec)
    return pd.DataFrame(out)


def pairwise(d, ref):
    rows = []
    for env, ge in d.groupby("eval_env"):
        r = ge[ge.agent == ref]["ret"]
        if r.empty:
            continue
        for agent, g in ge.groupby("agent"):
            if agent == ref:
                continue
            _, p = mannwhitneyu(g["ret"], r, alternative="two-sided")
            rows.append(dict(eval_env=env, agent=agent, ref=ref, p=p,
                             cliffs_delta=cliffs_delta(g["ret"].to_numpy(), r.to_numpy()),
                             mean_diff=g["ret"].mean() - r.mean()))
    t = pd.DataFrame(rows)
    if not t.empty:
        t["p_holm"] = holm(t.p)
        t["significant"] = t.p_holm < 0.05
    return t


def fmt(v, lo, hi, pct=False):
    if pct:
        return f"{100*v:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{v:.1f} [{lo:.1f}, {hi:.1f}]"


def table_main(s, env="adaptive"):
    s = s[s.eval_env == env].sort_values("ret", ascending=False)
    cols = [m for m in METRICS if m in s]
    lines = [r"\begin{table*}[t]\centering",
             rf"\caption{{Held-out results on the \texttt{{{env}}} environment "
             r"(mean, bootstrap 95\% CI; rates in \%).}}",
             r"\label{tab:main}", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Policy & " + " & ".join(LABEL[c] for c in cols) + r" \\", r"\midrule"]
    for _, r in s.iterrows():
        cells = [fmt(r[c], r[f"{c}_lo"], r[f"{c}_hi"], pct=c.endswith("rate")) for c in cols]
        lines.append(r.agent.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_cross(s):
    piv = s.pivot(index="agent", columns="eval_env", values="ret")
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Mean return of each policy on each evaluation environment.}",
             r"\label{tab:cross}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(piv.columns) + "}", r"\toprule",
             "Policy & " + " & ".join(c.replace("_", r"\_") for c in piv.columns) + r" \\",
             r"\midrule"]
    for agent, row in piv.iterrows():
        lines.append(agent.replace("_", r"\_") + " & "
                     + " & ".join(f"{v:.1f}" for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def fig_cross(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    envs = list(dict.fromkeys(s.eval_env))
    agents = list(s.sort_values("ret", ascending=False).agent.unique())
    w = 0.8 / max(len(agents), 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, ag in enumerate(agents):
        g = s[s.agent == ag].set_index("eval_env").reindex(envs)
        x = np.arange(len(envs)) + i * w
        ax.bar(x, g.ret, w, yerr=[g.ret - g.ret_lo, g.ret_hi - g.ret], capsize=2, label=ag)
    ax.set_xticks(np.arange(len(envs)) + w * (len(agents) - 1) / 2, envs, rotation=15)
    ax.set_ylabel("Episode return")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=6, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(RESULTS / "episodes.csv"))
    p.add_argument("--ref", default="static_rule")
    p.add_argument("--main-env", default="adaptive")
    a = p.parse_args(argv)
    d = pd.read_csv(a.csv)
    d["agent"] = d.apply(label, axis=1)
    RESULTS.mkdir(exist_ok=True)
    s = summarise(d)
    s.to_csv(RESULTS / "summary.csv", index=False)
    t = pairwise(d, a.ref)
    t.to_csv(RESULTS / "tests.csv", index=False)
    (RESULTS / "table_main.tex").write_text(table_main(s, a.main_env))
    (RESULTS / "table_cross.tex").write_text(table_cross(s))
    fig_cross(s, RESULTS / "fig_cross_env.pdf")
    print(f"wrote summary.csv, tests.csv, table_main.tex, table_cross.tex, fig_cross_env.pdf to {RESULTS}")


if __name__ == "__main__":
    main()
