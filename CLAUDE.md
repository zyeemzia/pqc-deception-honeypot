# Project guide for Claude Code

Repo: pqc-deception-honeypot — RL-adaptive deception for post-quantum (ML-KEM-768) ESP32 IoT gateways.
Author and only contributor: Zyeem Bin Zia (GitHub: zyeemzia). Commit as `zyeemzia`; no co-author or assistant attribution lines.

## Branches
- `main` = published letter version (IEEE ESL submission, Zenodo DOI 10.5281/zenodo.24030629). Do not change results or code there.
- `journal-extension` = work for the extended full-journal paper. All new work goes in `extended/`.

## Goal of the extension (≥30% new content over the letter)
1. Adaptive attacker: timing side channel, source rotation, mimicry (`extended/envs/adaptive_env.py`).
2. More algorithms: QR-DQN, DQN, PPO, A2C, RecurrentPPO, plus stronger hand-written baselines.
3. Cross-evaluation: train on base/adaptive, test on every variant and ablation.
4. Statistics: bootstrap 95% CIs, Mann-Whitney U with Holm correction, Cliff's delta.
5. Hardware: ESP32 energy per handshake (INA219), second board (ESP32-S3), live attack traffic on the physical decoy.

## Layout
- `rl-env/pqc_deception_env.py` — letter environment (keep unchanged; extension subclasses it).
- `extended/experiments.py` — the experiment grid (seeds, steps, algos, envs). Change numbers only here.
- `extended/train_all.py` — training (`--all` resumes, skipping existing models).
- `extended/evaluate_all.py` — cross-evaluation → `extended/results/episodes.csv` (TODO).
- `extended/analysis.py` — stats, LaTeX tables, figures (TODO).
- `extended/hardware/` — energy/serial log parsers (TODO).

## Machine
Laptop: i7-6500U, 2 cores, CPU only → `torch.set_num_threads(2)`. Full grid is long; test first with `--steps 20000`.

## Rules
- Every number in the paper must come from a CSV produced by these scripts. Never invent results.
- Keep observation/action spaces identical to the base env so cross-evaluation works.
- Verify references before citing (see reference-verification practice used for the letter).
