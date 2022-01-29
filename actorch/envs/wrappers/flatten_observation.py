# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Flatten observation (possibly batched) environment wrapper."""

from typing import Any, Tuple, Union

from gym import Env, ObservationWrapper
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.spaces import Flat
from actorch.envs.utils import Nested
from actorch.envs.wrappers.wrapper import Wrapper


__all__ = [
    "FlattenObservation",
]


class FlattenObservation(Wrapper, ObservationWrapper):
    """(Possibly batched) environment wrapper that flattens the observation."""

    def __init__(self, env: "Union[Env, BatchedEnv]") -> "None":
        super().__init__(env)
        is_batched = isinstance(env.unwrapped, BatchedEnv)
        if is_batched:
            self.single_observation_space = Flat(env.single_observation_space)
        self.observation_space = Flat(env.observation_space, is_batched)
        self._is_observation_flat = hasattr(env.unwrapped, "unflatten_observation")
        if self._is_observation_flat:
            # Disable unflatten_observation to improve performance
            env.unwrapped.unflatten_observation = False

    # override
    def reset(self, *args: "Any", **kwargs: "Any") -> "ndarray":
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)

    # override
    def step(self, *args: "Any", **kwargs: "Any") -> "Tuple[ndarray, Any, Any, Any]":
        observation, reward, done, info = self.env.step(*args, **kwargs)
        return self.observation(observation), reward, done, info

    # override
    def observation(self, observation: "Nested") -> "ndarray":
        if self._is_observation_flat:
            return observation
        return self.observation_space.flatten(observation)
