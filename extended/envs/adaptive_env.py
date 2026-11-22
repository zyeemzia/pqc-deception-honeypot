"""Adaptive-attacker extension of PQCDeceptionEnv (journal version).

The letter version (rl-env/pqc_deception_env.py) models decoy detection as a
fixed per-interaction probability (DETECT_P). Reviewers of a full journal paper
will ask: "what if the attacker actively tries to tell the decoy apart?"
This module answers that with three mechanisms, each switchable so that every
one can be ablated separately:

1. Timing side channel (``timing_attack``)
   Every handshake/probe answered by a server reveals its compute time.
   - real service:      sampled from the measured ESP32 keygen+decaps times
   - high-fidelity decoy: resampled from the same measured distribution
   - low-fidelity decoy:  measured times scaled by ``lofi_bias`` (+ jitter),
                          i.e. a cheaper host that does not emulate the MCU well
   The attacker keeps a running sample of log-latencies and runs a one-sample
   z-test against its reference model of real ESP32 timing. When
   |z| > ``z_thresh`` after ``min_samples`` observations it flags a decoy.
   Detection is therefore *emergent* from calibrated data, not a constant.

2. Source rotation (``rotate_on_detect``)
   Instead of leaving after spotting a decoy, the adaptive attacker rotates to
   a fresh source address: per-source history features reset, so the defender
   loses the evidence it had accumulated. Up to ``max_rotations`` times.

3. Traffic mimicry (``mimicry``)
   Attacker request rates are drawn from the legitimate-client distribution,
   removing the rate feature as an easy giveaway.

Optional ``obs_noise`` adds Gaussian noise to observations (sensor/telemetry
imperfection). The observation and action spaces are identical to the base
environment, so agents trained on the base env can be evaluated here
unchanged (cross-environment robustness).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl-env"))
from pqc_deception_env import (  # noqa: E402
    PROFILES, ROUTE_DECOY, ROUTE_DECOY_HI, REQ_BANNER, REQ_HANDSHAKE, REQ_DOWNGRADE,
    PQCDeceptionEnv,
)

# New attacker profiles for the journal version (base profiles are kept).
EXT_PROFILES = dict(PROFILES)
EXT_PROFILES["adaptive"] = dict(
    stages=[REQ_BANNER, REQ_HANDSHAKE, REQ_HANDSHAKE, REQ_DOWNGRADE], stealthy=True)
EXT_PROFILES["mimic"] = dict(
    stages=[REQ_HANDSHAKE, REQ_HANDSHAKE, REQ_DOWNGRADE], stealthy=True)

TIMING_REQS = (REQ_HANDSHAKE, REQ_DOWNGRADE)  # requests whose reply exposes compute time


class AdaptivePQCDeceptionEnv(PQCDeceptionEnv):
    def __init__(self, *, timing_attack=True, rotate_on_detect=True, mimicry=True,
                 lofi_bias=0.7, lofi_jitter=0.05, z_thresh=3.0, min_samples=3,
                 max_rotations=2, residual_detect=None, obs_noise=0.0,
                 profiles=None, **base_kwargs):
        super().__init__(**base_kwargs)
        self.timing_attack = timing_attack
        self.rotate_on_detect = rotate_on_detect
        self.mimicry = mimicry
        self.lofi_bias = lofi_bias
        self.lofi_jitter = lofi_jitter
        self.z_thresh = z_thresh
        self.min_samples = min_samples
        self.max_rotations = max_rotations
        self.obs_noise = obs_noise
        self.profiles = profiles or EXT_PROFILES
        # With the timing channel on, the fixed DETECT_P is replaced by a small
        # residual (non-timing behavioural cues); off -> fall back to base values.
        if residual_detect is None:
            residual_detect = ({ROUTE_DECOY: 0.05, ROUTE_DECOY_HI: 0.01}
                               if timing_attack else dict(self.detect_p))
        self.detect_p = residual_detect
        log_c = np.log(self.compute_ms)
        self.ref_mu, self.ref_sd = float(log_c.mean()), float(max(log_c.std(ddof=1), 1e-6))

    # ---------- episode ----------
    def reset(self, seed=None, options=None):
        # base reset draws the profile from our extended table (see super_reset)
        obs, info = super_reset(self, seed, options)
        self.timing_samples: list[float] = []
        self.rotations = 0
        self.m.update(timing_detected=0, rotations=0, residual_detected=0)
        return self._noisy(obs), info

    def _next_request(self):
        super()._next_request()
        if self.src == "attacker" and self.mimicry and self.profile.get("stealthy"):
            # rates indistinguishable from legitimate clients
            self.src_rate = self.np_random.uniform(0.0, 0.3)

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if info:
            info["rotations"] = self.rotations
        return self._noisy(obs), r, term, trunc, info

    def _noisy(self, obs):
        if self.obs_noise <= 0:
            return obs
        noise = self.np_random.normal(0.0, self.obs_noise, size=obs.shape)
        return np.clip(obs + noise, 0.0, 1.0).astype(np.float32)

    # ---------- attacker ----------
    def _observed_compute(self, action):
        rng = self.np_random
        base = float(rng.choice(self.compute_ms))
        if action == ROUTE_DECOY:
            return base * self.lofi_bias * (1.0 + rng.normal(0.0, self.lofi_jitter))
        return base  # real service and high-fidelity decoy share the measured distribution

    def _timing_flag(self):
        n = len(self.timing_samples)
        if n < self.min_samples:
            return False
        z = (np.mean(self.timing_samples) - self.ref_mu) / (self.ref_sd / np.sqrt(n))
        return abs(z) > self.z_thresh

    def _attacker(self, a, rng):
        req = self.req
        was_active = self.attacker_active
        detected_before = self.m["decoy_detected"]
        r = super()._attacker(a, rng)  # residual (behavioural) detection happens in here
        residual_hit = self.m["decoy_detected"] > detected_before

        timing_hit = False
        if self.timing_attack and self.attacker_active and req in TIMING_REQS and a != 4:
            self.timing_samples.append(float(np.log(max(self._observed_compute(a), 1e-3))))
            if a in (ROUTE_DECOY, ROUTE_DECOY_HI) and self._timing_flag():
                timing_hit = True
                self.m["timing_detected"] += 1
                self.m["decoy_detected"] += 1
                self.attacker_suspects = 1.0
                self.attacker_active = False
        if residual_hit:
            self.m["residual_detected"] += 1

        if (timing_hit or residual_hit) and was_active and self.rotate_on_detect \
                and self.rotations < self.max_rotations:
            self._rotate_source()
        return r

    def _rotate_source(self):
        """Attacker reappears from a fresh address: per-source evidence is lost."""
        self.rotations += 1
        self.m["rotations"] = self.rotations
        self.attacker_active = True
        self.attacker_hist = np.zeros(4)
        self.attacker_seen_decoy = 0.0
        self.attacker_suspects = 0.0
        self.timing_samples = []


def super_reset(env, seed, options):
    """Call the base reset while letting it pick from our extended profile table."""
    import pqc_deception_env as base
    saved = base.PROFILES
    base.PROFILES = env.profiles
    try:
        return PQCDeceptionEnv.reset(env, seed=seed, options=options)
    finally:
        base.PROFILES = saved


# Named environment variants used throughout the journal experiments.
ENV_VARIANTS = {
    "base":        lambda **kw: PQCDeceptionEnv(**kw),
    "adaptive":    lambda **kw: AdaptivePQCDeceptionEnv(**kw),
    # ablations: switch off one adaptive mechanism at a time
    "no_timing":   lambda **kw: AdaptivePQCDeceptionEnv(timing_attack=False, **kw),
    "no_rotation": lambda **kw: AdaptivePQCDeceptionEnv(rotate_on_detect=False, **kw),
    "no_mimicry":  lambda **kw: AdaptivePQCDeceptionEnv(mimicry=False, **kw),
    "noisy_obs":   lambda **kw: AdaptivePQCDeceptionEnv(obs_noise=0.05, **kw),
}


def make_env(name: str, **kwargs):
    return ENV_VARIANTS[name](**kwargs)
