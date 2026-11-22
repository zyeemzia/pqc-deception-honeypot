"""Single source of truth for the journal experiment grid.

Change numbers here, not inside the scripts, so the paper's Methods section
can cite one file.
"""
SEEDS = [42, 123, 456, 789, 1024]          # same seeds as the letter
TIMESTEPS = 300_000                         # letter used 200k; more for harder env
EVAL_EPISODES = 200                         # held-out episodes per seed
EVAL_SEED_BASE = 1_000_000                  # eval seeds never seen in training

# Algorithms compared (value-based, distributional, policy-gradient, recurrent).
ALGOS = ["qrdqn", "dqn", "ppo", "a2c", "recurrent_ppo"]

# Environments agents are TRAINED on.
TRAIN_ENVS = ["base", "adaptive"]

# Environments every trained agent is EVALUATED on (cross-evaluation matrix).
EVAL_ENVS = ["base", "adaptive", "no_timing", "no_rotation", "no_mimicry", "noisy_obs"]

# Hand-written baselines (no training).
BASELINES = ["no_deception", "random", "static_rule", "always_decoy_hi", "threshold_rule"]

# Hyper-parameters. Off-policy shares the letter's settings for comparability.
OFF_POLICY = dict(learning_rate=5e-4, buffer_size=100_000, learning_starts=5_000,
                  batch_size=128, gamma=0.97, train_freq=4, target_update_interval=2_000,
                  exploration_fraction=0.3, exploration_final_eps=0.02)
ON_POLICY = dict(learning_rate=3e-4, gamma=0.97)
NET = [128, 128]
