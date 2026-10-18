"""Aggregate results across seeds, run significance tests, and make paper figures.

Unit of analysis = seed (n=5): per-seed means are compared, so tests are not
inflated by treating 200 correlated episodes as independent samples.
Outputs: eval/summary_table.md, eval/stats_tests.md, figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

METRICS = ["ret", "recon_deflection_rate", "downgrade_capture_rate",
           "flood_on_real_rate", "legit_fp_rate", "engaged_steps", "decoy_detected"]
ORDER = ["no_deception", "random", "static_rule", "dqn", "qrdqn"]


def main():
    d = pd.read_csv(ROOT / "eval" / "results_episodes.csv")
    order = [p for p in ORDER if p in d.policy.unique()]
    per_seed = d.groupby(["policy", "seed"])[METRICS].mean().reset_index()

    agg = per_seed.groupby("policy")[METRICS].agg(["mean", "std"]).loc[order]
    table = pd.DataFrame({m: [f"{agg.loc[p, (m, 'mean')]:.3f} ± {agg.loc[p, (m, 'std')]:.3f}"
                              for p in order] for m in METRICS}, index=order)
    (ROOT / "eval" / "summary_table.md").write_text(table.to_markdown())
    print(table.to_markdown(), "\n")

    lines = ["| comparison | metric | test | statistic | p-value |", "|---|---|---|---|---|"]
    target = "qrdqn" if "qrdqn" in order else order[-1]
    for base in [p for p in order if p != target]:
        for m in ["ret", "recon_deflection_rate", "legit_fp_rate"]:
            a = per_seed[per_seed.policy == target].sort_values("seed")[m].to_numpy()
            b = per_seed[per_seed.policy == base].sort_values("seed")[m].to_numpy()
            diff = a - b
            if np.allclose(diff, 0):
                lines.append(f"| {target} vs {base} | {m} | identical | - | 1.0 |")
                continue
            normal = stats.shapiro(diff).pvalue > 0.05
            res = stats.ttest_rel(a, b) if normal else stats.wilcoxon(a, b)
            name = "paired t" if normal else "Wilcoxon"
            lines.append(f"| {target} vs {base} | {m} | {name} | {res.statistic:.3f} | {res.pvalue:.4f} |")
    lines.append("\nn = 5 seeds (paired by seed). With n=5, the smallest possible "
                 "two-sided Wilcoxon p-value is 0.0625, so the paired t-test is used when "
                 "differences pass Shapiro-Wilk.")
    (ROOT / "eval" / "stats_tests.md").write_text("\n".join(lines))
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, m, title in zip(axes, ["ret", "recon_deflection_rate", "downgrade_capture_rate", "legit_fp_rate"],
                            ["Episode return", "Recon deflection rate", "Downgrade capture rate",
                             "Legit false-positive rate"]):
        means = [agg.loc[p, (m, "mean")] for p in order]
        stds = [agg.loc[p, (m, "std")] for p in order]
        ax.bar(order, means, yerr=stds, capsize=4, color="#4C72B0")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(FIG / "policy_comparison.png", dpi=200)

    prof = d[d.policy.isin(["static_rule", target])].groupby(["profile", "policy"]).ret.mean().unstack()
    prof.plot.bar(figsize=(7, 3.6), rot=0, title="Return by attacker profile")
    plt.tight_layout()
    plt.savefig(FIG / "return_by_profile.png", dpi=200)

    hw = pd.read_csv(ROOT / "eval" / "esp32_kyber.csv")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.boxplot([hw.keygen_us / 1e3, hw.encaps_us / 1e3, hw.decaps_us / 1e3])
    ax.set_xticks([1, 2, 3], ["KeyGen", "Encaps", "Decaps"])
    ax.set_ylabel("ms (ESP32 @ 240 MHz)")
    ax.set_title("ML-KEM-768 on ESP32 (n=100)")
    plt.tight_layout()
    plt.savefig(FIG / "esp32_mlkem_latency.png", dpi=200)
    print("figures saved to", FIG)


if __name__ == "__main__":
    main()
