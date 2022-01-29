# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""(Possibly batched) environment wrapper."""

from typing import Any, Tuple, Union

from gym import Env
from gym import Wrapper as GymWrapper
from gym import spaces

from actorch.envs.batched_env import BatchedEnv


__all__ = [
    "Wrapper",
]


class Wrapper(GymWrapper):
    """(Possibly batched) environment wrapper."""

    def __init__(self, env: "Union[Env, BatchedEnv]") -> "None":
        super().__init__(env)
        self._single_observation_space = None
        self._single_action_space = None

    @property
    def single_observation_space(self) -> "spaces.Space":
        if self._single_observation_space is None:
            return self.env.single_observation_space
        return self._single_observation_space

    @single_observation_space.setter
    def single_observation_space(self, space: "spaces.Space") -> "None":
        self._single_observation_space = space

    @property
    def single_action_space(self) -> "spaces.Space":
        if self._single_action_space is None:
            return self.env.single_action_space
        return self._single_action_space

    @single_action_space.setter
    def single_action_space(self, space: "spaces.Space") -> "None":
        self._single_action_space = space

    # override
    def reset(self, *args: "Any", **kwargs: "Any") -> "Any":
        return self.env.reset(*args, **kwargs)

    # override
    def step(self, *args: "Any", **kwargs: "Any") -> "Tuple[Any, Any, Any, Any]":
        return self.env.step(*args, **kwargs)

    # override
    def seed(self, *args: "Any", **kwargs: "Any") -> "None":
        return self.env.seed(*args, **kwargs)

    # override
    def render(self, *args: "Any", **kwargs: "Any") -> "None":
        return self.env.render(*args, **kwargs)

    # override
    def close(self, *args: "Any", **kwargs: "Any") -> "None":
        return self.env.close(*args, **kwargs)

    def __len__(self) -> "int":
        return self.env.__len__()

    # Implicitly forward all other attributes to self.env
    def __getattr__(self, name: "str") -> "Any":
        return getattr(self.env, name)
