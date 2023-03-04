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

"""Train distributed data parallel REINFORCE on CartPole-v1."""

# Navigate to `<path-to-repository>/examples`, open a terminal and run:
# actorch run DistributedDataParallelREINFORCE_CartPole-v1.py

import gymnasium as gym
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=DistributedDataParallelREINFORCE,
    stop={"training_iteration": 50},
    # resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=DistributedDataParallelREINFORCE.Config(
        # Distributed execution configuration
        num_workers=2,
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
        # Algorithm configuration
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: gym.make("CartPole-v1", **kwargs),
            config,
            num_workers=2,
        ),
        train_num_episodes_per_iter=5,
        eval_freq=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes_per_iter=10,
        policy_network_model_builder=FCNet,
        policy_network_model_config={
            "torso_fc_configs": [{"out_features": 64, "bias": True}],
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
        suppress_warnings=True,
    ),
)
