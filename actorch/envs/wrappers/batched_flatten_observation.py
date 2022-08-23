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

"""Batched flatten observation environment wrapper."""

from typing import Optional, Sequence, Tuple, Union

from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.spaces import Flat
from actorch.envs.utils import Nested
from actorch.envs.wrappers.batched_wrapper import BatchedWrapper


__all__ = [
    "BatchedFlattenObservation",
]


class BatchedFlattenObservation(BatchedWrapper):
    """Batched environment wrapper that flattens the observation."""

    # override
    def __init__(self, env: "BatchedEnv") -> "None":
        super().__init__(env)
        self._single_observation_space = Flat(env.single_observation_space)
        self._observation_space = Flat(env.observation_space, is_batched=True)
        self._is_observation_flat = hasattr(env.unwrapped, "unflatten_observation")
        if self._is_observation_flat:
            # Disable unflatten_observation to improve performance
            env.unwrapped.unflatten_observation = False

    # override
    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[ndarray, ndarray]":
        observation, info = self.env.reset(mask)
        return self._observation(observation), info

    # override
    def step(
        self,
        action: "Nested[ndarray]",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]":
        observation, reward, terminal, truncated, info = self.env.step(action, mask)
        return self._observation(observation), reward, terminal, truncated, info

    def _observation(self, observation: "Nested[ndarray]") -> "ndarray":
        if self._is_observation_flat:
            return observation
        return self._observation_space.flatten(observation)
