"""Gymnasium environment: RL-adaptive deception for a PQC (ML-KEM-768) IoT gateway.

Each step, one request arrives at the gateway from either a legitimate client or
the episode's attacker. The defender (agent) decides where it goes.

Defender actions
  0 ROUTE_REAL      serve on the real ESP32 ML-KEM service
  1 ROUTE_DECOY     serve on the low-fidelity decoy
  2 ROUTE_DECOY_HI  serve on the high-fidelity decoy (harder to detect, costs more)
  3 THROTTLE        serve on real, rate-limited (blunts floods, delays legit users)
  4 BLOCK           drop the source

Attacker kill chain (MITRE ATT&CK)
  RECON  (TA0043 Reconnaissance: banner grab / scanning)
  PROBE  (TA0043: timing-probe handshakes to fingerprint the device)
  DOWNGRADE (TA0005 Defense Evasion: request classical ECDH fallback)
  FLOOD  (TA0040 Impact: handshake-flood DoS)

Latency costs are sampled from real ESP32 measurements (eval/network_handshake.csv),
not assumed.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]

ROUTE_REAL, ROUTE_DECOY, ROUTE_DECOY_HI, THROTTLE, BLOCK = range(5)
ACTIONS = ["route_real", "route_decoy", "route_decoy_hi", "throttle", "block"]

REQ_BANNER, REQ_HANDSHAKE, REQ_DOWNGRADE, REQ_FLOOD = range(4)
REQ_NAMES = ["banner", "handshake", "downgrade", "flood"]

# attacker profiles: request probabilities per kill-chain stage
PROFILES = {
    "scanner":    dict(stages=[REQ_BANNER], stealthy=False),
    "downgrader": dict(stages=[REQ_BANNER, REQ_DOWNGRADE], stealthy=False),
    "flooder":    dict(stages=[REQ_BANNER, REQ_FLOOD], stealthy=False),
    "apt":        dict(stages=[REQ_BANNER, REQ_HANDSHAKE, REQ_DOWNGRADE, REQ_FLOOD], stealthy=False),
    "stealthy":   dict(stages=[REQ_HANDSHAKE, REQ_HANDSHAKE, REQ_DOWNGRADE], stealthy=True),
}

# per-interaction probability that the attacker recognises a decoy
DETECT_P = {ROUTE_DECOY: 0.25, ROUTE_DECOY_HI: 0.05}

R = dict(
    engage=1.0,             # attacker interacting with a decoy (per step)
    recon_deflect=2.0,      # recon/probe answered by decoy
    downgrade_capture=4.0,  # downgrade attempt captured at decoy (IoC)
    recon_leak=-3.0,        # recon/probe answered by the real device
    downgrade_real=-2.0,    # downgrade attempt reached the real device (refused, but leaks policy)
    flood_real=-5.0,        # flood reached the real device
    flood_throttled=-1.0,
    legit_served=0.5,
    legit_misrouted=-4.0,   # legit client sent to a decoy (false positive)
    legit_blocked=-6.0,
    attacker_blocked=1.0,   # blocking ends engagement early, so small reward
    decoy_hi_cost=-0.3,     # extra resources for the high-fidelity decoy
    latency_weight=-1.0,    # per second of added legit latency
)


def load_calibration():
    d = pd.read_csv(ROOT / "eval" / "network_handshake.csv")
    if "timeout" in d:
        d = d[d.timeout == 0]
    compute = (d.server_keygen_ms + d.server_decaps_ms).astype(float).to_numpy()
    total = d.total_ms.astype(float).to_numpy()
    return compute, total


class PQCDeceptionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=60, legit_rate=0.35, profile=None, reward_scale=None, detect_scale=1.0):
        super().__init__()
        self.max_steps = max_steps
        self.legit_rate = legit_rate
        self.fixed_profile = profile
        # sensitivity analysis: reward_scale={"pos": k1, "neg": k2} multiplies positive/negative terms
        rs = reward_scale or {}
        self.R = {k: v * (rs.get("pos", 1.0) if v > 0 else rs.get("neg", 1.0)) for k, v in R.items()}
        self.detect_p = {k: min(1.0, v * detect_scale) for k, v in DETECT_P.items()}
        self.compute_ms, self.total_ms = load_calibration()
        # obs: req one-hot(4), src_prev_banner, src_prev_handshake, src_prev_downgrade,
        #      src_prev_flood, src_seen_decoy, src_rate, step_frac, attacker_suspected_decoy
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    # ---------- episode ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.profile_name = self.fixed_profile or rng.choice(list(PROFILES))
        self.profile = PROFILES[self.profile_name]
        self.stage = 0
        self.stage_len = int(rng.integers(3, 8))
        self.stage_step = 0
        self.attacker_active = True
        self.attacker_hist = np.zeros(4)
        self.attacker_seen_decoy = 0.0
        self.attacker_suspects = 0.0
        self.t = 0
        self.m = dict(profile=self.profile_name, engaged_steps=0, recon_total=0,
                      recon_deflected=0, downgrade_total=0, downgrade_captured=0,
                      flood_total=0, flood_on_real=0, legit_total=0, legit_misrouted=0,
                      legit_blocked=0, decoy_detected=0, legit_latency_ms=0.0)
        self._next_request()
        return self._obs(), {}

    def _next_request(self):
        rng = self.np_random
        if not self.attacker_active or rng.random() < self.legit_rate:
            self.src = "legit"
            self.req = REQ_HANDSHAKE
            self.src_hist = np.array([0, rng.integers(0, 3), 0, 0], dtype=float)
            self.src_rate = rng.uniform(0.0, 0.3)
            self.src_seen_decoy = 0.0
        else:
            self.src = "attacker"
            self.req = self.profile["stages"][self.stage]
            self.src_hist = self.attacker_hist.copy()
            self.src_rate = rng.uniform(0.7, 1.0) if self.req == REQ_FLOOD else (
                rng.uniform(0.0, 0.3) if self.profile["stealthy"] else rng.uniform(0.2, 0.6))
            self.src_seen_decoy = self.attacker_seen_decoy

    def _obs(self):
        onehot = np.eye(4)[self.req]
        hist = np.minimum(self.src_hist / 10.0, 1.0)
        extra = [self.src_seen_decoy, self.src_rate, self.t / self.max_steps,
                 self.attacker_suspects if self.src == "attacker" else 0.0]
        return np.concatenate([onehot, hist, extra]).astype(np.float32)

    # ---------- dynamics ----------
    def step(self, action):
        action = int(action)
        rng = self.np_random
        r = 0.0
        if self.src == "legit":
            r += self._legit(action, rng)
        else:
            r += self._attacker(action, rng)
        self.t += 1
        terminated = False
        truncated = self.t >= self.max_steps
        if not truncated:
            self._next_request()
        info = dict(self.m) if (terminated or truncated) else {}
        return self._obs(), float(r), terminated, truncated, info

    def _legit(self, a, rng):
        self.m["legit_total"] += 1
        if a in (ROUTE_DECOY, ROUTE_DECOY_HI):
            self.m["legit_misrouted"] += 1
            return self.R["legit_misrouted"]
        if a == BLOCK:
            self.m["legit_blocked"] += 1
            return self.R["legit_blocked"]
        lat = float(rng.choice(self.total_ms))
        if a == THROTTLE:
            lat += 500.0
        self.m["legit_latency_ms"] += lat
        return self.R["legit_served"] + self.R["latency_weight"] * lat / 1000.0

    def _attacker(self, a, rng):
        req, m, r = self.req, self.m, 0.0
        self.attacker_hist[req] += 1
        if req in (REQ_BANNER, REQ_HANDSHAKE):
            m["recon_total"] += 1
        elif req == REQ_DOWNGRADE:
            m["downgrade_total"] += 1
        else:
            m["flood_total"] += 1

        if a == BLOCK:
            self.attacker_active = False
            return self.R["attacker_blocked"]

        if a in (ROUTE_DECOY, ROUTE_DECOY_HI):
            self.attacker_seen_decoy = 1.0
            m["engaged_steps"] += 1
            r += self.R["engage"] + (self.R["decoy_hi_cost"] if a == ROUTE_DECOY_HI else 0.0)
            if req in (REQ_BANNER, REQ_HANDSHAKE):
                m["recon_deflected"] += 1
                r += self.R["recon_deflect"]
            elif req == REQ_DOWNGRADE:
                m["downgrade_captured"] += 1
                r += self.R["downgrade_capture"]
            if rng.random() < self.detect_p[a]:
                self.attacker_suspects = 1.0
                m["decoy_detected"] += 1
                self.attacker_active = False  # attacker recognises the decoy and leaves
                return r
        else:  # real service (optionally throttled)
            if req in (REQ_BANNER, REQ_HANDSHAKE):
                r += self.R["recon_leak"]
            elif req == REQ_DOWNGRADE:
                r += self.R["downgrade_real"]
            else:
                m["flood_on_real"] += 1
                r += self.R["flood_throttled"] if a == THROTTLE else self.R["flood_real"]

        self.stage_step += 1
        if self.stage_step >= self.stage_len and self.stage < len(self.profile["stages"]) - 1:
            self.stage += 1
            self.stage_step = 0
            self.stage_len = int(rng.integers(3, 8))
        return r
