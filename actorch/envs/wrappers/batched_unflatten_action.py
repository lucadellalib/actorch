# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Batched unflatten action environment wrapper."""

from typing import Optional, Sequence, Tuple, Union

from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.spaces import Flat
from actorch.envs.utils import Nested
from actorch.envs.wrappers.batched_wrapper import BatchedWrapper


__all__ = [
    "BatchedUnflattenAction",
]


class BatchedUnflattenAction(BatchedWrapper):
    """Batched environment wrapper that unflattens the action."""

    # override
    def __init__(self, env: "BatchedEnv") -> "None":
        super().__init__(env)
        self._single_action_space = Flat(env.single_action_space)
        self._action_space = Flat(env.action_space, is_batched=True)

    # override
    def step(
        self,
        action: "ndarray",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        return self.env.step(self._action(action), mask)

    def _action(self, action: "ndarray") -> "Nested[ndarray]":
        return self._action_space.unflatten(action)
