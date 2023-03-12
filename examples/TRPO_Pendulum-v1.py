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

"""Train Trust Region Policy Optimization (TRPO) on Pendulum-v1."""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# pip install gymnasium[classic_control]
# actorch run TRPO_Pendulum-v1.py

import gymnasium as gym
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=TRPO,
    stop={"timesteps_total": int(2e5)},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=TRPO.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: gym.make("Pendulum-v1", **kwargs),
            config,
            num_workers=2,
        ),
        train_num_timesteps_per_iter=512,
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
        policy_network_optimizer_config={
            "max_constraint": 0.01,
            "num_cg_iters": 10,
            "max_backtracks": 15,
            "backtrack_ratio": 0.8,
            "hvp_reg_coeff": 1e-5,
            "accept_violation": False,
            "epsilon": 1e-8,
        },
        value_network_model_builder=FCNet,
        value_network_model_config={
            "torso_fc_configs": [
                {"out_features": 256, "bias": True},
                {"out_features": 256, "bias": True},
            ],
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": 3e-3},
        discount=0.9,
        trace_decay=0.95,
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
