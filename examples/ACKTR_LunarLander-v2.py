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

"""Train Actor-Critic Kronecker-Factored Trust Region on LunarLander-v2."""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# actorch run ACKTR_LunarLander-v2.py

import gymnasium as gym
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=ACKTR,
    stop={"timesteps_total": int(4e5)},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=ACKTR.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: gym.make("LunarLander-v2", **kwargs),
            config,
            num_workers=1,
        ),
        train_num_timesteps_per_iter=256,
        eval_freq=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes_per_iter=10,
        policy_network_model_builder=FCNet,
        policy_network_model_config={
            "torso_fc_configs": [
                {"out_features": 64, "bias": True},
                {"out_features": 64, "bias": True},
            ],
        },
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 1e-3},
        policy_network_preconditioner_config={
            "lr": 0.1,
            "factor_decay": 0.95,
            "damping": 0.001,
            "kl_clip": 0.001,
            "factor_update_freq": 50,
            "kfac_update_freq": 100,
        },
        value_network_model_builder=FCNet,
        value_network_model_config={
            "torso_fc_configs": [
                {"out_features": 64, "bias": True},
                {"out_features": 64, "bias": True},
            ],
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": 5e-3},
        discount=0.99,
        num_return_steps=50,
        normalize_advantage=False,
        entropy_coeff=0.01,
        max_grad_l2_norm=2.0,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=True,
    ),
)