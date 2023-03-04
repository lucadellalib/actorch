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

"""Train Advantage Actor-Critic on LunarLander-v2. Tune the value network learning
rate using the asynchronous version of HyperBand (https://arxiv.org/abs/1810.05934).

"""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# actorch run A2C-AsyncHyperBand_LunarLander-v2.py

import gymnasium as gym
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=A2C,
    resources_per_trial={"cpu": 1, "gpu": 0},
    num_samples=10,
    scheduler=AsyncHyperBandScheduler(
        time_attr="timesteps_total",
        metric="cumreward_100",
        mode="max",
        max_t=int(4e5),
    ),
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=A2C.Config(
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
                {"out_features": 256, "bias": True},
                {"out_features": 256, "bias": True},
            ],
        },
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 1e-3},
        value_network_model_builder=FCNet,
        value_network_model_config={
            "torso_fc_configs": [
                {"out_features": 256, "bias": True},
                {"out_features": 256, "bias": True},
            ],
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": tune.uniform(1e-4, 1e-3)},
        discount=0.995,
        num_return_steps=20,
        normalize_advantage=False,
        entropy_coeff=0.01,
        max_grad_l2_norm=1.0,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=True,
    ),
)
