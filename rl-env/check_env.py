"""Sanity checks: Gymnasium API compliance, determinism under a seed, baseline behaviour."""
import numpy as np
from stable_baselines3.common.env_checker import check_env

from pqc_deception_env import PQCDeceptionEnv, ROUTE_REAL, ROUTE_DECOY, REQ_HANDSHAKE


def rollout(policy, seed, n=200):
    env = PQCDeceptionEnv()
    rets, infos = [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            done = term or trunc
        rets.append(ret)
        infos.append(info)
    return np.array(rets), infos


def main():
    check_env(PQCDeceptionEnv(), warn=True)
    print("check_env: OK")

    a, _ = rollout(lambda o: 0, seed=42, n=5)
    b, _ = rollout(lambda o: 0, seed=42, n=5)
    print("deterministic under seed:", np.allclose(a, b))

    rng = np.random.default_rng(0)
    policies = {
        "no_deception": lambda o: ROUTE_REAL,
        "random": lambda o: int(rng.integers(0, 5)),
        "static_rule": lambda o: ROUTE_REAL if o[REQ_HANDSHAKE] == 1 else ROUTE_DECOY,
    }
    for name, p in policies.items():
        rets, infos = rollout(p, seed=42)
        fp = sum(i["legit_misrouted"] for i in infos) / max(1, sum(i["legit_total"] for i in infos))
        print(f"{name:13s} return={rets.mean():7.2f} ± {rets.std():5.2f}  legit_FP_rate={fp:.3f}")


if __name__ == "__main__":
    main()
