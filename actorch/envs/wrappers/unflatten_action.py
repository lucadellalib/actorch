# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Unflatten action (possibly batched) environment wrapper."""

from typing import Any, Tuple, Union

from gym import ActionWrapper, Env
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.spaces import Flat
from actorch.envs.utils import Nested
from actorch.envs.wrappers.wrapper import Wrapper


__all__ = [
    "UnflattenAction",
]


class UnflattenAction(Wrapper, ActionWrapper):
    """(Possibly batched) environment wrapper that unflattens the action."""

    def __init__(self, env: "Union[Env, BatchedEnv]") -> "None":
        super().__init__(env)
        is_batched = isinstance(env.unwrapped, BatchedEnv)
        if is_batched:
            self.single_action_space = Flat(env.single_action_space)
        self.action_space = Flat(env.action_space, is_batched)

    # override
    def step(
        self, action: "ndarray", *args: "Any", **kwargs: "Any"
    ) -> "Tuple[Any, Any, Any, Any]":
        return self.env.step(self.action(action), *args, **kwargs)

    # override
    def action(self, action: "ndarray") -> "Nested":
        return self.action_space.unflatten(action)

    # override
    def reverse_action(self, action: "Nested") -> "ndarray":
        return self.action_space.flatten(action)
