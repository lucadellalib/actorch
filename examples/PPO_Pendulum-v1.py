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

"""Train Proximal Policy Optimization (PPO) on Pendulum-v1."""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# pip install gymnasium[classic_control]
# actorch run PPO_Pendulum-v1.py

import gymnasium as gym
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=PPO,
    stop={"timesteps_total": int(1e5)},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=PPO.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: gym.make("Pendulum-v1", **kwargs),
            config,
            num_workers=2,
        ),
        train_num_timesteps_per_iter=2048,
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
        policy_network_optimizer_config={"lr": 5e-5},
        value_network_model_builder=FCNet,
        value_network_model_config={
            "torso_fc_configs": [
                {"out_features": 256, "bias": True},
                {"out_features": 256, "bias": True},
            ],
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": 3e-3},
        discount=0.99,
        trace_decay=0.95,
        num_epochs=20,
        minibatch_size=16,
        ratio_clip=0.2,
        normalize_advantage=True,
        entropy_coeff=0.01,
        max_grad_l2_norm=0.5,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=True,
    ),
)
