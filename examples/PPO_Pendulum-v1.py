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

"""Train Proximal Policy Optimization on Pendulum-v1."""

# Navigate to `<path-to-repository>`, open a terminal and run:
# cd examples
# actorch run PPO_Pendulum-v1.py

import gym
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=PPO,
    stop={"timesteps_total": int(3e5)},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=PPO.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **config: gym.make("Pendulum-v1", **config),
            config,
            num_workers=1,
        ),
        train_num_episodes_per_iter=1,
        eval_freq=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes_per_iter=10,
        policy_network_model_builder=FCNet,
        policy_network_model_config={
            "torso_fc_configs": [{"out_features": 256, "bias": True}],
            "head_activation_builder": nn.Tanh,
            "independent_heads": ["action/log_scale"],
        },
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 5e-4},
        policy_network_optimizer_lr_scheduler_builder=StepLR,
        policy_network_optimizer_lr_scheduler_config={"step_size": 13000, "gamma": 0.7},
        value_network_model_builder=FCNet,
        value_network_model_config={
            "torso_fc_configs": [{"out_features": 256, "bias": True}],
        },
        value_network_optimizer_builder=Adam,
        value_network_optimizer_config={"lr": 3e-3, "weight_decay": 0.001},
        discount=0.99,
        trace_decay=0.95,
        num_epochs=10,
        minibatch_size=128,
        ratio_clip=0.2,
        normalize_advantage=True,
        entropy_coeff=0.001,
        max_grad_l2_norm=0.5,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=False,
    ),
)
