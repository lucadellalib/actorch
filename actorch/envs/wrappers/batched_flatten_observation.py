# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Batched flatten observation environment wrapper."""

from typing import Optional, Sequence, Tuple, Union

from gym import ObservationWrapper
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.spaces import Flat
from actorch.envs.utils import Nested
from actorch.envs.wrappers.batched_wrapper import BatchedWrapper


__all__ = [
    "BatchedFlattenObservation",
]


class BatchedFlattenObservation(BatchedWrapper, ObservationWrapper):
    """Batched environment wrapper that flattens the observation."""

    def __init__(self, env: "BatchedEnv") -> "None":
        super().__init__(env)
        self.single_observation_space = Flat(env.single_observation_space)
        self.observation_space = Flat(env.observation_space, is_batched=True)
        self._is_observation_flat = hasattr(env.unwrapped, "unflatten_observation")
        if self._is_observation_flat:
            # Disable unflatten_observation to improve performance
            env.unwrapped.unflatten_observation = False

    # override
    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "ndarray":
        observation = self.env.reset(mask)
        return self.observation(observation)

    # override
    def step(
        self,
        action: "Nested[ndarray]",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[ndarray, ndarray, ndarray, ndarray]":
        observation, reward, done, info = self.env.step(action, mask)
        return self.observation(observation), reward, done, info

    # override
    def observation(self, observation: "Nested[ndarray]") -> "ndarray":
        if self._is_observation_flat:
            return observation
        return self.observation_space.flatten(observation)
