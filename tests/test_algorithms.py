#!/usr/bin/env python3

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

"""Test algorithms."""

import atexit
import copy
import os
import shutil
import signal

import gymnasium as gym
import pytest
import ray
from gymnasium import spaces
from ray import tune

from actorch import ExperimentParams, Flat, algorithms, models


_TMP_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")


class RepeatObservationEnv(gym.Env):
    """Environment in which the agent has to repeat the observation.

    In the continuous case, the reward is set equal to the negative
    L1 distance between the action and the observation.
    In the discrete case, the reward is set equal to 1 if the action
    exactly matches the observation, 0 otherwise.

    """

    # override
    def __init__(self, space=None, episode_length=50, **kwargs):
        """Initialize the object.

        Parameters
        ----------
        space:
            The observation/action space.
            Default to ``gym.spaces.Discrete(2)``.
        episode_length:
            The episode length.

        """
        if space is None:
            space = spaces.Discrete(2)
        self.observation_space = self.space = space
        self.action_space = copy.deepcopy(self.observation_space)
        self.episode_length = episode_length
        self._observation = self._num_steps = None

    # override
    def reset(self, *args, **kwargs):
        self._num_timesteps = 0
        return self._next_observation(), {}

    # override
    def step(self, action, *args, **kwargs):
        self._num_timesteps += 1
        reward = self._reward(self.action_space, self._observation, action)
        terminal = truncated = self._num_timesteps >= self.episode_length
        return self._next_observation(), reward, terminal, truncated, {}

    def _next_observation(self):
        self._observation = self.observation_space.sample()
        return self._observation

    def _reward(self, space, observation, action):
        if isinstance(space, spaces.Tuple):
            return sum(
                self._reward(s, o, a) for s, o, a in zip(space, observation, action)
            )
        if isinstance(space, spaces.Dict):
            return sum(
                self._reward(s, o, a)
                for s, o, a in zip(
                    space.values(), observation.values(), action.values()
                )
            )
        if isinstance(space, spaces.Box):
            return -abs(action - observation).sum()
        if isinstance(space, spaces.Discrete):
            return 1.0 if action == observation else 0.0
        if isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete)):
            return 1.0 if (action == observation).all() else 0.0
        raise NotImplementedError(
            f"Unsupported space type: `{type(space).__module__}.{type(space).__name__}`"
        )


def _cleanup():
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    ray.shutdown()


atexit.register(_cleanup)
signal.signal(signal.SIGTERM, _cleanup)
signal.signal(signal.SIGINT, _cleanup)


@pytest.mark.parametrize(
    "algorithm_cls",
    [
        algorithms.A2C,
        algorithms.ACKTR,
        algorithms.AWR,
        algorithms.PPO,
        algorithms.REINFORCE,
    ],
)
@pytest.mark.parametrize(
    "space",
    [
        spaces.Box(low=0.0, high=5.0, shape=()),
        spaces.Box(low=0.0, high=5.0, shape=(3,)),
        spaces.Box(low=0.0, high=5.0, shape=(3, 4)),
        spaces.Box(low=0.0, high=5.0, shape=(3, 4, 5)),
        spaces.Discrete(1),
        spaces.Discrete(2),
        spaces.Discrete(4),
        spaces.MultiBinary(1),
        spaces.MultiBinary(5),
        spaces.MultiBinary([2]),
        spaces.MultiBinary([2, 3]),
        spaces.MultiBinary([2, 3, 4]),
        spaces.MultiDiscrete(1),
        spaces.MultiDiscrete([5]),
        spaces.MultiDiscrete([2, 3, 4]),
        spaces.Tuple(
            [
                spaces.Box(low=0.0, high=5.0, shape=(3, 4)),
                spaces.Discrete(2),
                spaces.MultiBinary(5),
                spaces.Dict(
                    {
                        "a": spaces.MultiDiscrete([2, 3, 10]),
                        "b": spaces.Discrete(8),
                        "c": spaces.Box(low=0.0, high=5.0, shape=(3,)),
                    }
                ),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "model_cls",
    [models.FCNet, models.ConvNet, models.LSTMNet],
)
def test_algorithm_mixed(algorithm_cls, model_cls, space):
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        experiments_params = ExperimentParams(
            run_or_experiment=algorithm_cls,
            stop={"training_iteration": 1},
            local_dir=_TMP_DIR,
            config=algorithm_cls.Config(
                train_env_builder=lambda **config: RepeatObservationEnv(
                    space,
                    **config,
                ),
                policy_network_model_builder=model_cls,
            ),
        )
        if model_cls == models.ConvNet and Flat(space).sample().ndim < 2:
            return
        if (
            algorithm_cls in [algorithms.ACKTR, algorithms.DistributedDataParallelACKTR]
            and model_cls == models.LSTMNet
        ):
            return
        if algorithm_cls not in [
            algorithms.REINFORCE,
            algorithms.DistributedDataParallelREINFORCE,
        ]:
            experiments_params["config"]["value_network_model_builder"] = model_cls
        tune.run(**experiments_params)
    finally:
        _cleanup()


@pytest.mark.parametrize(
    "algorithm_cls",
    [
        algorithms.DDPG,
        algorithms.TD3,
    ],
)
@pytest.mark.parametrize(
    "space",
    [
        spaces.Box(low=0.0, high=5.0, shape=()),
        spaces.Box(low=0.0, high=5.0, shape=(3,)),
        spaces.Box(low=0.0, high=5.0, shape=(3, 4)),
        spaces.Box(low=0.0, high=5.0, shape=(3, 4, 5)),
        spaces.Tuple(
            [
                spaces.Box(low=0.0, high=5.0, shape=(3, 4)),
                spaces.Box(low=1.0, high=2.0, shape=(2,)),
                spaces.Dict(
                    {
                        "a": spaces.Box(low=0.0, high=5.0, shape=(5, 6)),
                        "b": spaces.Box(low=0.0, high=5.0, shape=(3,)),
                    }
                ),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "model_cls",
    [models.FCNet, models.ConvNet, models.LSTMNet],
)
def test_algorithm_continuous(algorithm_cls, model_cls, space):
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        experiments_params = ExperimentParams(
            run_or_experiment=algorithm_cls,
            stop={"training_iteration": 1},
            local_dir=_TMP_DIR,
            config=algorithm_cls.Config(
                train_env_builder=lambda **config: RepeatObservationEnv(
                    space,
                    **config,
                ),
                policy_network_model_builder=model_cls,
            ),
        )
        if model_cls == models.ConvNet and Flat(space).sample().ndim < 2:
            return
        tune.run(**experiments_params)
    finally:
        _cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
