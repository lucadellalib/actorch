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

"""Train Advantage-Weighted Regression (AWR) equipped with an affine flow policy on Pendulum-v1."""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# pip install gymnasium[classic_control]
# actorch run AWR-AffineFlow_Pendulum-v1.py

import gymnasium as gym
import torch
from torch import nn
from torch.optim import Adam

from actorch import *


class AffineFlow(NormalizingFlow):
    bijective = True  # override
    is_constant_jacobian = True  # override
    domain = torch.distributions.constraints.real  # override
    codomain = torch.distributions.constraints.real  # override

    # override
    def __init__(self, cache_size=0):
        super().__init__(cache_size)
        # Pendulum-v1 action shape: (1,)
        weight = nn.Parameter(torch.ones(1))
        self.register_parameter("weight", weight)

    # override
    def _call(self, x):
        return x * self.weight

    # override
    def _inverse(self, y):
        return y / (self.weight + 1e-6)

    # override
    def log_abs_det_jacobian(self, x, y):
        return self.weight.abs().log()

    # override
    def forward_shape(self, shape):
        return shape

    # override
    def inverse_shape(self, shape):
        return shape


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
        policy_network_normalizing_flows={
            "action": AffineFlow(),
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
