# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Serial batched environment."""

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from gym import Env
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.utils import Nested, batch, flatten, unflatten


__all__ = [
    "SerialBatchedEnv",
]


class SerialBatchedEnv(BatchedEnv):
    """Batched environment based on loops."""

    def __init__(
        self,
        env_builder: "Callable[..., Env]",
        env_config: "Optional[Dict[str, Any]]" = None,
        num_workers: "int" = 1,
    ) -> "None":
        super().__init__(env_builder, env_config, num_workers)
        dummy_flat_observation = flatten(
            self.single_observation_space,
            self.single_observation_space.sample(),
            copy=False,
        )
        self._envs = [self.env_builder() for _ in range(num_workers)]
        self._observation_buffer = [
            unflatten(
                self.single_observation_space,
                np.zeros_like(dummy_flat_observation),
                copy=False,
            )
            for _ in range(num_workers)
        ]
        self._reward = np.zeros(num_workers)
        self._done = np.zeros(num_workers, dtype=bool)
        self._info = np.array([{} for _ in range(num_workers)])

    # override
    def _reset(self, idx: "Sequence[int]") -> "Nested[ndarray]":
        for i in idx:
            self._observation_buffer[i] = self._envs[i].reset()
        observation = batch(self.observation_space, self._observation_buffer)  # Copy
        return observation

    # override
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray]":
        for i in idx:
            (
                self._observation_buffer[i],
                self._reward[i],
                self._done[i],
                self._info[i],
            ) = self._envs[i].step(action[i])
        observation = batch(self.observation_space, self._observation_buffer)  # Copy
        return (
            observation,
            np.array(self._reward),  # Copy
            np.array(self._done),  # Copy
            deepcopy(self._info),
        )

    # override
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        for i, env in enumerate(self._envs):
            env.seed(seed[i])

    # override
    def _render(self, idx: "Sequence[int]", **kwargs: "Any") -> "None":
        for i in idx:
            self._envs[i].render(**kwargs)

    # override
    def _close(self) -> "None":
        for env in self._envs:
            env.close()
