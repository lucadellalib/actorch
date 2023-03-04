# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train Advantage-Weighted Regression on Pendulum-v1."""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# actorch run AWR_Pendulum-v1.py

import gymnasium as gym
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=AWR,
    stop={"timesteps_total": int(1e5)},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=AWR.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: gym.make("Pendulum-v1", **kwargs),
            config,
            num_workers=2,
        ),
        train_num_timesteps_per_iter=400,
        eval_freq=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes_per_iter=10,
        policy_network_model_builder=FCNet,
        policy_network_model_config={
            "torso_fc_configs": [
                {"out_features": 256, "bias": True},
                {"out_features": 256, "bias": True},
            ],
            "independent_heads": ["action/log_scale"],
        },
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 5e-3},
        value_network_model_builder=FCNet,
        value_network_model_config={
            "torso_fc_configs": [
                {"out_features": 256, "bias": True},
                {"out_features": 256, "bias": True},
            ],
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": 5e-3},
        buffer_config={"capacity": int(1e4)},
        discount=0.99,
        trace_decay=0.95,
        num_updates_per_iter=400,
        batch_size=32,
        max_trajectory_length=float("inf"),
        weight_clip=20.0,
        temperature=0.05,
        normalize_advantage=False,
        entropy_coeff=0.01,
        max_grad_l2_norm=0.5,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=True,
    ),
)
