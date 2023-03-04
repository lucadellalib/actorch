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

"""Batched environment."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from gymnasium import Env, Space
from numpy import ndarray

from actorch.envs.utils import Nested, batch_space, unbatch


__all__ = [
    "BatchedEnv",
]


class BatchedEnv(ABC):
    """Environment that runs multiple copies of a base
    environment in a synchronous fashion.

    """

    def __init__(
        self,
        base_env_builder: "Callable[..., Env]",
        base_env_config: "Optional[Dict[str, Any]]" = None,
        num_workers: "int" = 1,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        base_env_builder:
            The base environment builder, i.e. a callable that
            receives keyword arguments from a base environment
            configuration and returns a base environment.
        base_env_config:
            The base environment configuration.
            Default to ``{}``.
        num_workers:
            The number of copies of the base environment.

        Raises
        ------
        ValueError
            If `num_workers` is not in the integer interval [1, inf).

        """
        if num_workers < 1 or not float(num_workers).is_integer():
            raise ValueError(
                f"`num_workers` ({num_workers}) must be in the integer interval [1, inf)"
            )
        num_workers = int(num_workers)
        self.base_env_builder = base_env_builder
        self.base_env_config = base_env_config or {}
        self.num_workers = num_workers
        with self.base_env_builder(
            **self.base_env_config,
        ) as base_env:
            self._single_observation_space = base_env.observation_space
            self._single_action_space = base_env.action_space
            self._repr = (
                f"<{type(self).__name__}"
                f"(base_env: {base_env}, "
                f"num_workers: {num_workers})>"
            )
        self._observation_space = batch_space(
            self.single_observation_space, num_workers
        )
        self._action_space = batch_space(self.single_action_space, num_workers)
        self._is_closed = False

    @property
    def single_observation_space(self) -> "Space":
        """Return the single observation space.

        Returns
        -------
            The single observation space.

        """
        return self._single_observation_space

    @property
    def single_action_space(self) -> "Space":
        """Return the single action space.

        Returns
        -------
            The single action space.

        """
        return self._single_action_space

    @property
    def observation_space(self) -> "Space":
        """Return the observation space.

        Returns
        -------
            The observation space.

        """
        return self._observation_space

    @property
    def action_space(self) -> "Space":
        """Return the action space.

        Returns
        -------
            The action space.

        """
        return self._action_space

    @property
    def unwrapped(self) -> "BatchedEnv":
        """Return the unwrapped environment.

        Returns
        -------
            The unwrapped environment.

        """
        return self

    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray]":
        """Reset a batch of workers.

        In the following, let `N` denote the number of workers
        and `O` the shape of a single observation leaf value.

        Parameters
        ----------
        mask:
            The boolean array indicating which workers are
            to reset (True) and which are not (False), shape: ``[N]``.
            If a scalar or a singleton, it is broadcast accordingly.
            Default to ``np.ones(N, dtype=bool)``.

        Returns
        -------
            - The batched initial observation, shape of a leaf value: ``[N, *O]``;
            - the batched auxiliary diagnostic information, shape: ``[N]``.

        Raises
        ------
        RuntimeError
            If the environment has already been closed.
        ValueError
            If length of `mask` is not equal to the number of workers.

        Warnings
        --------
        If ``mask[i]`` is False, the corresponding observation
        and auxiliary diagnostic information preserve their
        previous values.

        """
        if self._is_closed:
            raise RuntimeError("Trying to reset a closed environment")
        if mask is None:
            mask = np.ones(self.num_workers, dtype=bool)
        else:
            mask = np.array(mask, copy=False, ndmin=1)
            if len(mask) == 1:
                mask = np.broadcast_to(mask, (self.num_workers,))
            if len(mask) != self.num_workers:
                raise ValueError(
                    f"Length of `mask` ({len(mask)}) must be equal "
                    f"to the number of workers ({self.num_workers})"
                )
        idx = np.where(mask)[0].tolist()
        return self._reset(idx)

    def step(
        self,
        action: "Nested[ndarray]",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        """Step a batch of workers.

        In the following, let `N` denote the number of workers,
        `O` the shape of a single observation leaf value and
        `A` the shape of a single action leaf value.

        Parameters
        ----------
        action:
            The batched action, shape of a leaf value: ``[N, *A]``.
        mask:
            The boolean array indicating which workers are
            to step (True) and which are not (False), shape: ``[N]``.
            If a scalar or a singleton, it is broadcast accordingly.
            Default to ``np.ones(N, dtype=bool)``.

        Returns
        -------
            - The batched next observation, shape of a leaf value: ``[N, *O]``;
            - the batched reward, shape: ``[N]``;
            - the batched terminal flag, shape: ``[N]``;
            - the batched truncated flag, shape: ``[N]``;
            - the batched auxiliary diagnostic information, shape: ``[N]``.

        Raises
        ------
        RuntimeError
            If the environment has already been closed.
        ValueError
            If length of `action` or `mask` is not equal
            to the number of workers.

        Warnings
        --------
        If ``mask[i]`` is False, the corresponding observation,
        reward, terminal flag, truncated flag and auxiliary
        diagnostic information preserve their previous values.

        """
        if self._is_closed:
            raise RuntimeError("Trying to step a closed environment")
        if mask is None:
            mask = np.ones(self.num_workers, dtype=bool)
        else:
            mask = np.array(mask, copy=False, ndmin=1)
            if len(mask) == 1:
                mask = np.broadcast_to(mask, (self.num_workers,))
            if len(mask) != self.num_workers:
                raise ValueError(
                    f"Length of `mask` ({len(mask)}) must be equal "
                    f"to the number of workers ({self.num_workers})"
                )
        action = unbatch(self.action_space, action)
        if len(action) != self.num_workers:
            raise ValueError(
                f"Length of `action` ({len(action)}) must be equal "
                f"to the number of workers ({self.num_workers})"
            )
        idx = np.where(mask)[0].tolist()
        return self._step(action, idx)

    def seed(
        self,
        seed: "Union[Optional[int], Sequence[Optional[int]], ndarray]" = None,
    ) -> "None":
        """Seed all workers.

        In the following, let `N` denote the number of workers.

        Parameters
        ----------
        seed:
            The batched seed, shape: ``[N]``.
            If a scalar or a scalar singleton, it
            is expanded to ``[seed + i for i in range(N)]`` or
            ``[seed[0] + i for i in range(N)]``, respectively.
            Default to ``[None] * N``.

        Raises
        ------
        RuntimeError
            If the environment has already been closed.
        ValueError
            If length of `seed` is not equal to the number of workers.

        Warnings
        --------
        Observation and action spaces are seeded along with the
        workers (using ``seed[0]``).

        """
        if self._is_closed:
            raise RuntimeError("Trying to seed a closed environment")
        if seed is None:
            seed = [None] * self.num_workers
        elif np.isscalar(seed):
            seed = [seed + i for i in range(self.num_workers)]
        elif len(seed) == 1:
            seed = (
                [None] * self.num_workers
                if seed[0] is None
                else [seed[0] + i for i in range(self.num_workers)]
            )
        elif len(seed) != self.num_workers:
            raise ValueError(
                f"Length of `seed` ({len(seed)}) must be equal "
                f"to the number of workers ({self.num_workers})"
            )
        elif isinstance(seed, np.ndarray):
            seed = seed.tolist()
        self.single_observation_space.seed(seed[0])
        self.single_action_space.seed(seed[0])
        self.observation_space.seed(seed[0])
        self.action_space.seed(seed[0])
        self._seed(seed)

    def close(self) -> "None":
        """Close all workers."""
        if self._is_closed:
            return
        self._close()
        self._is_closed = True

    def __len__(self) -> "int":
        return self.num_workers

    def __enter__(self) -> "BatchedEnv":
        return self

    def __exit__(self, *args: "Any", **kwargs: "Any") -> "None":
        if not self._is_closed:
            self.close()

    def __del__(self) -> "None":
        if hasattr(self, "_is_closed") and not self._is_closed:
            self.close()

    def __repr__(self) -> "str":
        return self._repr

    @abstractmethod
    def _reset(self, idx: "Sequence[int]") -> "Tuple[Nested[ndarray], ndarray]":
        """See documentation of `reset`."""
        raise NotImplementedError

    @abstractmethod
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        """See documentation of `step`."""
        raise NotImplementedError

    @abstractmethod
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        """See documentation of `seed`."""
        raise NotImplementedError

    @abstractmethod
    def _close(self) -> "None":
        """See documentation of `close`."""
        raise NotImplementedError
