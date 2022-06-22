# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Batched unflatten action environment wrapper."""

from typing import Optional, Sequence, Tuple, Union

from gym import ActionWrapper
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.spaces import Flat
from actorch.envs.utils import Nested
from actorch.envs.wrappers.batched_wrapper import BatchedWrapper


__all__ = [
    "BatchedUnflattenAction",
]


class BatchedUnflattenAction(BatchedWrapper, ActionWrapper):
    """Batched environment wrapper that unflattens the action."""

    def __init__(self, env: "BatchedEnv") -> "None":
        super().__init__(env)
        self.single_action_space = Flat(env.single_action_space)
        self.action_space = Flat(env.action_space, is_batched=True)

    # override
    def step(
        self,
        action: "ndarray",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray]":
        return self.env.step(self.action(action), mask)

    # override
    def action(self, action: "ndarray") -> "Nested[ndarray]":
        return self.action_space.unflatten(action)

    # override
    def reverse_action(self, action: "Nested[ndarray]") -> "ndarray":
        return self.action_space.flatten(action)
