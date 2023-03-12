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

"""Train Soft Actor-Critic (SAC) on BipedalWalker-v3."""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# pip install gymnasium[box2d]
# actorch run SAC_BipedalWalker-v3.py

import gymnasium as gym
import numpy as np
import torch.nn
from torch.distributions import Normal, TanhTransform
from torch.optim import Adam

from actorch import *


class Wrapper(gym.Wrapper):
    def __init__(self, env, action_noise=0.3, action_repeat=3, reward_scale=5):
        super().__init__(env)
        self.action_noise = action_noise
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale

    def step(self, action):
        action += self.action_noise * (
            1 - 2 * np.random.random(self.action_space.shape)
        )
        cumreward = 0.0
        for _ in range(self.action_repeat):
            observation, reward, terminated, truncated, info = super().step(action)
            cumreward += reward
            if terminated:
                return observation, 0.0, terminated, truncated, info
        return observation, self.reward_scale * cumreward, terminated, truncated, info


experiment_params = ExperimentParams(
    run_or_experiment=SAC,
    stop={"timesteps_total": int(5e5)},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=SAC.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: Wrapper(
                gym.wrappers.TimeLimit(
                    gym.make("BipedalWalker-v3", **kwargs),
                    max_episode_steps=200,
                ),
            ),
            config,
            num_workers=1,
        ),
        train_num_episodes_per_iter=1,
        eval_freq=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes_per_iter=10,
        policy_network_model_builder=FCNet,
        policy_network_model_config={
            "torso_fc_configs": [
                {"out_features": 400, "bias": True},
                {"out_features": 300, "bias": True},
            ],
            "head_activation_builder": torch.nn.Tanh,
        },
        policy_network_distribution_builders={
            "action": lambda loc, scale: TransformedDistribution(
                Normal(loc, scale),
                TanhTransform(cache_size=1),  # Use tanh normal to enforce action bounds
            ),
        },
        policy_network_distribution_parametrizations={
            "action": {
                "loc": (
                    {"loc": (4,)},
                    lambda x: x["loc"],
                ),
                "scale": (
                    {"log_scale": (4,)},
                    lambda x: x["log_scale"].clamp(-20.0, 2.0).exp(),
                ),
            },
        },
        policy_network_sample_fn=lambda d: d.sample(),  # `mode` does not exist in closed-form for tanh normal
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 1e-3},
        value_network_model_builder=FCNet,
        value_network_model_config={
            "torso_fc_configs": [
                {"out_features": 400, "bias": True},
                {"out_features": 300, "bias": True},
            ],
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": 1e-3},
        buffer_config={"capacity": int(3e5)},
        discount=0.98,
        num_return_steps=5,
        num_updates_per_iter=10,
        batch_size=1024,
        max_trajectory_length=20,
        sync_freq=1,
        polyak_weight=0.001,
        temperature=0.2,
        max_grad_l2_norm=1.0,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=True,
    ),
)
