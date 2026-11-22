"""Cross-evaluate every policy on every environment variant.

For each (train_env, algo, seed) model in extended/models/ and every hand-written
baseline, run EVAL_EPISODES held-out episodes on each env in EVAL_ENVS.
Eval seeds (EVAL_SEED_BASE + seed*1000 + ep) never occur during training.

Usage
  python extended/evaluate_all.py                  # everything found on disk
  python extended/evaluate_all.py --episodes 20    # quick check
Output: extended/results/episodes.csv  (one row per policy x eval_env x seed x episode)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "envs"))

import experiments as X  # noqa: E402
from adaptive_env import make_env  # noqa: E402
from pqc_deception_env import (  # noqa: E402
    ROUTE_REAL, ROUTE_DECOY, ROUTE_DECOY_HI, THROTTLE, REQ_HANDSHAKE, REQ_FLOOD,
)

MODELS = HERE / "models"
RESULTS = HERE / "results"


# ---------- hand-written baselines ----------
def baseline(name, rng):
    if name == "no_deception":
        return lambda o: ROUTE_REAL
    if name == "random":
        return lambda o: int(rng.integers(0, 5))
    if name == "static_rule":  # letter baseline: handshakes real, everything else decoy
        return lambda o: ROUTE_REAL if o[REQ_HANDSHAKE] >= 0.5 else ROUTE_DECOY
    if name == "always_decoy_hi":  # non-handshakes to the high-fidelity decoy
        return lambda o: ROUTE_REAL if o[REQ_HANDSHAKE] >= 0.5 else ROUTE_DECOY_HI
    if name == "threshold_rule":
        # stronger expert rule: floods throttled; handshakes from sources with a
        # suspicious history (prior downgrade/flood or high rate) go to hi-fi decoy
        def pol(o):
            if o[REQ_FLOOD] >= 0.5:
                return THROTTLE
            if o[REQ_HANDSHAKE] >= 0.5:
                suspicious = o[6] > 0 or o[7] > 0 or o[9] > 0.5 or o[8] > 0.5
                return ROUTE_DECOY_HI if suspicious else ROUTE_REAL
            return ROUTE_DECOY_HI
        return pol
    raise ValueError(name)


def load_model(path: Path):
    from stable_baselines3 import A2C, DQN, PPO
    from sb3_contrib import QRDQN, RecurrentPPO
    cls = dict(qrdqn=QRDQN, dqn=DQN, ppo=PPO, a2c=A2C, recurrent_ppo=RecurrentPPO)
    algo = path.stem.rsplit("_seed", 1)[0]
    return algo, cls[algo].load(path, device="cpu")


class ModelPolicy:
    """Wraps an SB3 model; keeps LSTM state for recurrent policies."""

    def __init__(self, model):
        self.model, self.state, self.start = model, None, True

    def reset(self):
        self.state, self.start = None, True

    def __call__(self, obs):
        a, self.state = self.model.predict(obs, state=self.state,
                                           episode_start=np.array([self.start]),
                                           deterministic=True)
        self.start = False
        return int(a)


# ---------- rollout ----------
def run(policy, env_name, seed, n_ep):
    env = make_env(env_name)
    rows = []
    for ep in range(n_ep):
        if hasattr(policy, "reset"):
            policy.reset()
        obs, _ = env.reset(seed=X.EVAL_SEED_BASE + seed * 1000 + ep)
        done, ret, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            done = term or trunc
        rows.append(dict(seed=seed, episode=ep, ret=ret, **info))
    return rows


def add_rates(d: pd.DataFrame) -> pd.DataFrame:
    d["recon_deflection_rate"] = d.recon_deflected / d.recon_total.clip(lower=1)
    d["downgrade_capture_rate"] = d.downgrade_captured / d.downgrade_total.clip(lower=1)
    d["legit_fp_rate"] = d.legit_misrouted / d.legit_total.clip(lower=1)
    d["flood_on_real_rate"] = d.flood_on_real / d.flood_total.clip(lower=1)
    return d


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=X.EVAL_EPISODES)
    p.add_argument("--envs", nargs="+", default=X.EVAL_ENVS)
    p.add_argument("--no-baselines", action="store_true")
    a = p.parse_args(argv)

    rows = []
    for env_name in a.envs:
        for seed in X.SEEDS:
            if not a.no_baselines:
                for b in X.BASELINES:
                    pol = baseline(b, np.random.default_rng(seed))
                    for r in run(pol, env_name, seed, a.episodes):
                        rows.append(dict(r, policy=b, train_env="-", eval_env=env_name))
            for path in sorted(MODELS.glob(f"*/*_seed{seed}.zip")):
                train_env = path.parent.name
                algo, model = load_model(path)
                for r in run(ModelPolicy(model), env_name, seed, a.episodes):
                    rows.append(dict(r, policy=algo, train_env=train_env, eval_env=env_name))
            print(f"eval_env={env_name} seed={seed} done")

    d = add_rates(pd.DataFrame(rows))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "episodes.csv"
    d.to_csv(out, index=False)
    print("saved", out, len(d), "rows")


if __name__ == "__main__":
    main()
