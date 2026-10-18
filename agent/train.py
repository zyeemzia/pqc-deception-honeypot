"""Train a defender agent on PQCDeceptionEnv.

Usage: python train.py <algo: qrdqn|dqn> <seed> [timesteps]
Saves agent/models/<algo>_seed<seed>.zip and appends wall time to agent/train_log.csv
"""
import sys
import time
from pathlib import Path

import torch
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl-env"))
from pqc_deception_env import PQCDeceptionEnv  # noqa: E402

torch.set_num_threads(2)  # i7-6500U: 2 physical cores

COMMON = dict(learning_rate=5e-4, buffer_size=100_000, learning_starts=5_000,
              batch_size=128, gamma=0.97, train_freq=4, target_update_interval=2_000,
              exploration_fraction=0.3, exploration_final_eps=0.02,
              policy_kwargs=dict(net_arch=[128, 128]), verbose=0)
ALGOS = {"qrdqn": QRDQN, "dqn": DQN}


def main():
    algo, seed = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    env = Monitor(PQCDeceptionEnv())
    env.reset(seed=seed)
    kwargs = dict(COMMON)
    if algo == "qrdqn":
        kwargs["policy_kwargs"] = dict(net_arch=[128, 128], n_quantiles=50)
    model = ALGOS[algo]("MlpPolicy", env, seed=seed, **kwargs)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    dt = time.time() - t0
    (HERE / "models").mkdir(exist_ok=True)
    model.save(HERE / "models" / f"{algo}_seed{seed}")
    log = HERE / "train_log.csv"
    new = not log.exists()
    with open(log, "a") as f:
        if new:
            f.write("algo,seed,timesteps,wall_s\n")
        f.write(f"{algo},{seed},{steps},{dt:.1f}\n")
    print(f"{algo} seed={seed} steps={steps} done in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
