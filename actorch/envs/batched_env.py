# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Batched environment."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from gym import Env
from numpy import ndarray

from actorch.envs.utils import Nested, batch_space, unbatch


__all__ = [
    "BatchedEnv",
]


class BatchedEnv(ABC, Env):
    """Environment that runs multiple copies of a base environment."""

    def __init__(
        self, env_builder: "Callable[[], Env]", num_workers: "int" = 1
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        env_builder:
            The base environment builder.
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
        self.env_builder = env_builder
        self.num_workers = num_workers
        dummy_env = env_builder()
        self.single_observation_space = dummy_env.observation_space
        self.single_action_space = dummy_env.action_space
        self.metadata = dummy_env.metadata
        self.reward_range = dummy_env.reward_range
        self.spec = dummy_env.spec
        self._repr = (
            f"<{self.__class__.__name__}"
            f"(env: {dummy_env}, "
            f"num_workers: {num_workers})>"
        )
        dummy_env.close()
        del dummy_env
        self.observation_space = batch_space(self.single_observation_space, num_workers)
        self.action_space = batch_space(self.single_action_space, num_workers)
        self._is_closed = False

    # override
    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Nested[ndarray]":
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
            The batched initial observation,
            shape of a leaf value: ``[N, *O]``.

        Raises
        ------
        RuntimeError
            If the environment has already been closed.
        ValueError
            If length of `mask` is not equal to the number of workers.

        Warnings
        --------
        If ``mask[i]`` is ``False``, the corresponding
        observation preserves its previous value.

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

    # override
    def step(
        self,
        action: "Nested[ndarray]",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray]":
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
            - the batched end-of-episode flag, shape: ``[N]``;
            - the batched auxiliary diagnostic information, shape: ``[N]``.

        Raises
        ------
        RuntimeError
            If the environment has already been closed.
        ValueError
            If length of `mask` or `action` is not equal
            to the number of workers.

        Warnings
        --------
        If ``mask[i]`` is ``False``, the corresponding
        observation, reward, end-of-episode flag and
        auxiliary diagnostic information preserve
        their previous value.

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

    # override
    def seed(
        self,
        seed: "Union[Optional[int], Sequence[Optional[int]], ndarray]" = None,
    ) -> "None":
        """Seed all workers.

        In the following, let `N` denote the number of workers.

        Parameters
        ----------
        seed:
            The batched seed, shape ``[N]``.
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
        self.observation_space.seed(seed[0])
        self.single_action_space.seed(seed[0])
        self.action_space.seed(seed[0])
        self._seed(seed)

    # override
    def render(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
        **kwargs: "Any",
    ) -> "None":
        """Render a batch of workers.

        In the following, let `N` denote the number of workers.

        Parameters
        ----------
        mask:
            The boolean array indicating which workers are
            to render (True) and which are not (False), shape: ``[N]``.
            If a scalar or a singleton, it is broadcast accordingly.
            Default to ``np.ones(N, dtype=bool)``.

        Raises
        ------
        RuntimeError
            If the environment has already been closed.
        ValueError
            If length of `mask` is not equal to the number of workers.

        """
        if self._is_closed:
            raise RuntimeError("Trying to render a closed environment")
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
        self._render(idx, **kwargs)

    # override
    def close(self) -> "None":
        """Close all workers."""
        if self._is_closed:
            return
        self._close()
        self._is_closed = True

    def __len__(self) -> "int":
        return self.num_workers

    def __del__(self) -> "None":
        if not self._is_closed:
            self.close()

    def __repr__(self) -> "str":
        return self._repr

    __str__ = __repr__

    @abstractmethod
    def _reset(self, idx: "Sequence[int]") -> "Nested[ndarray]":
        """See documentation of method `reset`."""
        raise NotImplementedError

    @abstractmethod
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray]":
        """See documentation of method `step`."""
        raise NotImplementedError

    @abstractmethod
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        """See documentation of method `seed`."""
        raise NotImplementedError

    @abstractmethod
    def _render(self, idx: "Sequence[int]", **kwargs: "Any") -> "None":
        """See documentation of method `render`."""
        raise NotImplementedError

    @abstractmethod
    def _close(self) -> "None":
        """See documentation of method `close`."""
        raise NotImplementedError
