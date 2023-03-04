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

"""Serial batched environment."""

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env
from gymnasium.utils import seeding
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.utils import Nested, batch, flatten, unflatten


__all__ = [
    "SerialBatchedEnv",
]


class SerialBatchedEnv(BatchedEnv):
    """Batched environment based on loops."""

    # override
    def __init__(
        self,
        base_env_builder: "Callable[..., Env]",
        base_env_config: "Optional[Dict[str, Any]]" = None,
        num_workers: "int" = 1,
    ) -> "None":
        super().__init__(base_env_builder, base_env_config, num_workers)
        dummy_flat_observation = flatten(
            self.single_observation_space,
            self.single_observation_space.sample(),
            copy=False,
        )
        self._envs = []
        for _ in range(num_workers):
            env = self.base_env_builder(**self.base_env_config)
            self._envs.append(env)
        self._observation_buffer = [
            unflatten(
                self.single_observation_space,
                np.zeros_like(dummy_flat_observation),
                copy=False,
            )
            for _ in range(num_workers)
        ]
        self._reward = np.zeros(num_workers)
        self._terminal = np.zeros(num_workers, dtype=bool)
        self._truncated = np.zeros(num_workers, dtype=bool)
        self._info = np.array([{} for _ in range(num_workers)])

    # override
    def _reset(self, idx: "Sequence[int]") -> "Tuple[Nested[ndarray], ndarray]":
        for i in idx:
            self._observation_buffer[i], self._info[i] = self._envs[i].reset()
        observation = batch(self.observation_space, self._observation_buffer)  # Copy
        return observation, deepcopy(self._info)

    # override
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        for i in idx:
            (
                self._observation_buffer[i],
                self._reward[i],
                self._terminal[i],
                self._truncated[i],
                self._info[i],
            ) = self._envs[i].step(action[i])
        observation = batch(self.observation_space, self._observation_buffer)  # Copy
        return (
            observation,
            np.array(self._reward),  # Copy
            np.array(self._terminal),  # Copy
            np.array(self._truncated),  # Copy
            deepcopy(self._info),
        )

    # override
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        for i, env in enumerate(self._envs):
            # Bypass deprecation warning
            env.np_random, _ = seeding.np_random(seed[i])

    # override
    def _close(self) -> "None":
        for env in self._envs:
            env.close()
