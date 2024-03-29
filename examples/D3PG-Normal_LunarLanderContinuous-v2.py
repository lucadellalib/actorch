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

"""Train Distributional Deep Deterministic Policy Gradient (D3PG) on LunarLanderContinuous-v2
assuming a normal value distribution.

"""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# pip install gymnasium[box2d]
# actorch run D3PG-Normal_LunarLanderContinuous-v2.py

import gymnasium as gym
from torch import nn
from torch.distributions import Normal
from torch.optim import Adam

from actorch import *


# Define custom model
class LayerNormFCNet(FCNet):
    # override
    def _setup_torso(self, in_shape):
        super()._setup_torso(in_shape)
        torso = nn.Sequential()
        for module in self.torso:
            torso.append(module)
            if isinstance(module, nn.Linear):
                torso.append(
                    nn.LayerNorm(module.out_features, elementwise_affine=False)
                )
        self.torso = torso


experiment_params = ExperimentParams(
    run_or_experiment=D3PG,
    stop={"timesteps_total": int(4e5)},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=D3PG.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: gym.make("LunarLanderContinuous-v2", **kwargs),
            config,
            num_workers=1,
        ),
        train_agent_builder=OUNoiseAgent,
        train_agent_config={
            "clip_action": True,
            "device": "cpu",
            "num_random_timesteps": 1000,
            "mean": 0.0,
            "volatility": 0.1,
            "reversion_speed": 0.15,
        },
        train_num_timesteps_per_iter=1024,
        eval_freq=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes_per_iter=10,
        policy_network_model_builder=LayerNormFCNet,
        policy_network_model_config={
            "torso_fc_configs": [
                {"out_features": 400, "bias": True},
                {"out_features": 300, "bias": True},
            ],
            "head_activation_builder": nn.Tanh,
        },
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 1e-3},
        value_network_model_builder=LayerNormFCNet,
        value_network_model_config={
            "torso_fc_configs": [
                {"out_features": 400, "bias": True},
                {"out_features": 300, "bias": True},
            ],
        },
        value_network_distribution_builder=Normal,
        value_network_distribution_parametrization={
            "loc": (
                {"loc": ()},
                lambda x: x["loc"],
            ),
            "scale": (
                {"log_scale": ()},
                lambda x: x["log_scale"].clamp(-20.0, 2.0).exp(),
            ),
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": 1e-3},
        buffer_config={
            "capacity": int(1e5),
            "prioritization": 1.0,
            "bias_correction": 0.4,
            "epsilon": 1e-5,
        },
        discount=0.995,
        num_return_steps=5,
        num_updates_per_iter=LambdaSchedule(
            lambda iter: (
                200 if iter >= 10 else 0
            ),  # Fill buffer with some trajectories before training
        ),
        batch_size=128,
        max_trajectory_length=10,
        sync_freq=1,
        polyak_weight=0.001,
        max_grad_l2_norm=5.0,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=True,
    ),
)
