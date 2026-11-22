"""Train defender agents for the journal experiment grid.

Examples
  python extended/train_all.py --algo qrdqn --env adaptive --seed 42
  python extended/train_all.py --all                 # full grid (long!)
  python extended/train_all.py --all --algos qrdqn ppo --envs adaptive --steps 50000

Models:  extended/models/<env>/<algo>_seed<seed>.zip   (existing ones are skipped,
         so an interrupted --all run can simply be restarted)
Log:     extended/results/train_log.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "envs"))

import experiments as X  # noqa: E402
from adaptive_env import make_env  # noqa: E402

MODELS = HERE / "models"
RESULTS = HERE / "results"


def build_model(algo, env, seed):
    import torch
    torch.set_num_threads(2)
    from stable_baselines3 import A2C, DQN, PPO
    from sb3_contrib import QRDQN, RecurrentPPO

    if algo == "qrdqn":
        return QRDQN("MlpPolicy", env, seed=seed, verbose=0, **X.OFF_POLICY,
                     policy_kwargs=dict(net_arch=X.NET, n_quantiles=50))
    if algo == "dqn":
        return DQN("MlpPolicy", env, seed=seed, verbose=0, **X.OFF_POLICY,
                   policy_kwargs=dict(net_arch=X.NET))
    if algo == "ppo":
        return PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=2048, batch_size=256,
                   ent_coef=0.01, **X.ON_POLICY, policy_kwargs=dict(net_arch=X.NET))
    if algo == "a2c":
        return A2C("MlpPolicy", env, seed=seed, verbose=0, n_steps=16, ent_coef=0.01,
                   learning_rate=7e-4, gamma=X.ON_POLICY["gamma"],
                   policy_kwargs=dict(net_arch=X.NET))
    if algo == "recurrent_ppo":
        # LSTM policy: can remember a source's behaviour across rotations
        return RecurrentPPO("MlpLstmPolicy", env, seed=seed, verbose=0, n_steps=1024,
                            batch_size=256, ent_coef=0.01, **X.ON_POLICY,
                            policy_kwargs=dict(net_arch=X.NET, lstm_hidden_size=64))
    raise ValueError(f"unknown algo {algo}")


def train_one(algo, env_name, seed, steps):
    out = MODELS / env_name / f"{algo}_seed{seed}.zip"
    if out.exists():
        print(f"skip {out.relative_to(HERE)} (exists)")
        return
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(make_env(env_name))
    env.reset(seed=seed)
    model = build_model(algo, env, seed)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    dt = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out.with_suffix(""))
    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / "train_log.csv"
    new = not log.exists()
    with open(log, "a") as f:
        if new:
            f.write("env,algo,seed,timesteps,wall_s\n")
        f.write(f"{env_name},{algo},{seed},{steps},{dt:.1f}\n")
    print(f"{env_name}/{algo} seed={seed} steps={steps}: {dt/60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo"), p.add_argument("--env"), p.add_argument("--seed", type=int)
    p.add_argument("--all", action="store_true", help="run the whole grid from experiments.py")
    p.add_argument("--algos", nargs="+", default=X.ALGOS)
    p.add_argument("--envs", nargs="+", default=X.TRAIN_ENVS)
    p.add_argument("--seeds", nargs="+", type=int, default=X.SEEDS)
    p.add_argument("--steps", type=int, default=X.TIMESTEPS)
    a = p.parse_args(argv)
    if a.all:
        jobs = [(al, e, s) for e in a.envs for al in a.algos for s in a.seeds]
        print(f"{len(jobs)} training jobs")
        for al, e, s in jobs:
            train_one(al, e, s, a.steps)
    else:
        if not (a.algo and a.env and a.seed is not None):
            p.error("give --algo, --env and --seed, or use --all")
        train_one(a.algo, a.env, a.seed, a.steps)


if __name__ == "__main__":
    main()
