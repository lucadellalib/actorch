# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Batched environment wrapper."""

from typing import Any, Optional, Sequence, Tuple, Union

from gym import Wrapper, spaces
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.utils import Nested


__all__ = [
    "BatchedWrapper",
]


class BatchedWrapper(Wrapper, BatchedEnv):
    """Batched environment wrapper."""

    def __init__(self, env: "BatchedEnv") -> "None":
        super().__init__(env)
        self._single_observation_space = None
        self._single_action_space = None

    @property
    def single_observation_space(self) -> "spaces.Space":
        if self._single_observation_space is None:
            return self.env.single_observation_space
        return self._single_observation_space

    @single_observation_space.setter
    def single_observation_space(self, value: "spaces.Space") -> "None":
        self._single_observation_space = value

    @property
    def single_action_space(self) -> "spaces.Space":
        if self._single_action_space is None:
            return self.env.single_action_space
        return self._single_action_space

    @single_action_space.setter
    def single_action_space(self, value: "spaces.Space") -> "None":
        self._single_action_space = value

    # override
    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Nested[ndarray]":
        return self.env.reset(mask)

    # override
    def step(
        self,
        action: "Nested[ndarray]",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray]":
        return self.env.step(action, mask)

    # override
    def seed(
        self,
        seed: "Union[Optional[int], Sequence[Optional[int]], ndarray]" = None,
    ) -> "None":
        self.env.seed(seed)
        self.single_observation_space.seed(seed[0])
        self.observation_space.seed(seed[0])
        self.single_action_space.seed(seed[0])
        self.action_space.seed(seed[0])

    # override
    def render(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
        **kwargs: "Any",
    ) -> "None":
        return self.env.render(mask, **kwargs)

    # override
    def close(self) -> "None":
        return self.env.close()

    # override
    def _reset(self, idx: "Sequence[int]") -> "Nested[ndarray]":
        raise NotImplementedError

    # override
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray]":
        raise NotImplementedError

    # override
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        raise NotImplementedError

    # override
    def _render(self, idx: "Sequence[int]", **kwargs: "Any") -> "None":
        raise NotImplementedError

    # override
    def _close(self) -> "None":
        raise NotImplementedError

    def __len__(self) -> "int":
        return self.env.__len__()

    def __del__(self) -> "None":
        return self.env.__del__()

    # Implicitly forward all other attributes to self.env
    def __getattr__(self, name: "str") -> "Any":
        return getattr(self.env, name)
