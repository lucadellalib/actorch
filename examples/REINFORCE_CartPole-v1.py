# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Train REINFORCE on CartPole-v1."""

# Navigate to `<path-to-repository>`, open a terminal and run:
# cd examples
# actorch run REINFORCE_CartPole-v1.py

import gym
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=REINFORCE,
    stop={"training_iteration": 30},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=REINFORCE.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **config: gym.make("CartPole-v1", **config),
            config,
            num_workers=2,
        ),
        train_num_episodes=10,
        eval_interval_iterations=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes=10,
        policy_network_model_builder=FCNet,
        policy_network_model_config={
            "torso_fc_configs": [
                {"out_features": 64, "bias": True}
            ],
        },
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 1e-1},
        discount=0.99,
        entropy_coeff=0.001,
        max_grad_l2_norm=0.5,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=False,
    ),
)
