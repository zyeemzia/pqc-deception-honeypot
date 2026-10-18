"""Sensitivity analysis: does QR-DQN still beat the static rule when the
hand-set reward weights / decoy-detection probabilities / traffic mix change?

Each condition retrains QR-DQN from scratch (3 seeds, 200k steps) and evaluates
it against static_rule and no_deception in the SAME modified environment.
Absolute returns are not comparable across conditions; rankings and the
operational metrics (deflection, capture, false positives) are.
Writes eval/sensitivity_episodes.csv and eval/sensitivity_table.md
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "rl-env"))
from pqc_deception_env import PQCDeceptionEnv, ROUTE_REAL, ROUTE_DECOY, REQ_HANDSHAKE  # noqa: E402
from train import COMMON  # noqa: E402

torch.set_num_threads(2)
SEEDS = [42, 123, 456]
STEPS = 200_000
N_EP = 200
CONDITIONS = {
    "rewards_pos_x0.5":   dict(reward_scale={"pos": 0.5}),
    "penalties_x1.5":     dict(reward_scale={"neg": 1.5}),
    "detect_p_x1.5":      dict(detect_scale=1.5),
    "detect_p_x0.5":      dict(detect_scale=0.5),
    "legit_rate_0.5":     dict(legit_rate=0.5),
}


def evaluate(policy, kwargs, seed, name, cond):
    env = PQCDeceptionEnv(**kwargs)
    rows = []
    for ep in range(N_EP):
        obs, _ = env.reset(seed=1_000_000 + seed * 1000 + ep)
        done, ret, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            done = term or trunc
        rows.append(dict(condition=cond, policy=name, seed=seed, episode=ep, ret=ret, **info))
    return rows


def main():
    only = sys.argv[1:] or list(CONDITIONS)
    out = ROOT / "eval" / "sensitivity_episodes.csv"
    for cond in only:
        kwargs = CONDITIONS[cond]
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            env = Monitor(PQCDeceptionEnv(**kwargs))
            env.reset(seed=seed)
            model = QRDQN("MlpPolicy", env, seed=seed,
                          **{**COMMON, "policy_kwargs": dict(net_arch=[128, 128], n_quantiles=50)})
            model.learn(total_timesteps=STEPS)
            print(f"{cond} seed={seed} trained in {(time.time() - t0) / 60:.1f} min", flush=True)
            pols = {"no_deception": lambda o: ROUTE_REAL,
                    "static_rule": lambda o: ROUTE_REAL if o[REQ_HANDSHAKE] == 1 else ROUTE_DECOY,
                    "qrdqn": lambda o, m=model: int(m.predict(o, deterministic=True)[0])}
            for name, p in pols.items():
                rows += evaluate(p, kwargs, seed, name, cond)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    summarize()


def summarize():
    d = pd.read_csv(ROOT / "eval" / "sensitivity_episodes.csv")
    d["recon_deflection_rate"] = d.recon_deflected / d.recon_total.clip(lower=1)
    d["legit_fp_rate"] = d.legit_misrouted / d.legit_total.clip(lower=1)
    per_seed = d.groupby(["condition", "policy", "seed"])[["ret", "recon_deflection_rate", "legit_fp_rate"]].mean()
    agg = per_seed.groupby(["condition", "policy"]).agg(["mean", "std"]).round(3)
    wins = per_seed.ret.unstack("policy")
    win = (wins.qrdqn > wins.static_rule).groupby("condition").agg(["sum", "count"])
    text = agg.to_markdown() + "\n\nQR-DQN return > static_rule (seeds won / total):\n\n" + win.to_markdown()
    (ROOT / "eval" / "sensitivity_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
