"""Evaluate all policies on held-out episodes for each of the 5 seeds.

Eval episode seeds (1_000_000 + seed*1000 + ep) never occur during training,
so results measure generalisation, not memorised episodes.
Writes eval/results_episodes.csv (one row per policy x seed x episode).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import QRDQN
from stable_baselines3 import DQN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "rl-env"))
from pqc_deception_env import PQCDeceptionEnv, ROUTE_REAL, ROUTE_DECOY, REQ_HANDSHAKE  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]
N_EPISODES = 200


def run(policy, seed):
    env = PQCDeceptionEnv()
    rows = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=1_000_000 + seed * 1000 + ep)
        done, ret, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            done = term or trunc
        rows.append(dict(seed=seed, episode=ep, ret=ret, **info))
    return rows


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        policies = {
            "no_deception": lambda o: ROUTE_REAL,
            "random": lambda o, rng=rng: int(rng.integers(0, 5)),
            "static_rule": lambda o: ROUTE_REAL if o[REQ_HANDSHAKE] == 1 else ROUTE_DECOY,
        }
        for algo, cls in [("dqn", DQN), ("qrdqn", QRDQN)]:
            path = HERE / "models" / f"{algo}_seed{seed}.zip"
            if path.exists():
                model = cls.load(path)
                policies[algo] = lambda o, m=model: int(m.predict(o, deterministic=True)[0])
        for name, pol in policies.items():
            for r in run(pol, seed):
                r["policy"] = name
                rows.append(r)
            print(f"seed {seed} {name} done")

    d = pd.DataFrame(rows)
    d["recon_deflection_rate"] = d.recon_deflected / d.recon_total.clip(lower=1)
    d["downgrade_capture_rate"] = d.downgrade_captured / d.downgrade_total.clip(lower=1)
    d["legit_fp_rate"] = d.legit_misrouted / d.legit_total.clip(lower=1)
    d["flood_on_real_rate"] = d.flood_on_real / d.flood_total.clip(lower=1)
    out = ROOT / "eval" / "results_episodes.csv"
    d.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
