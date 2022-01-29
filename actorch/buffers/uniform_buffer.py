# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Uniform experience replay buffer."""

from sys import getsizeof
from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray

from actorch.buffers.buffer import Buffer
from actorch.buffers.utils import compute_trajectory_priorities
from actorch.registry import register
from actorch.utils import normalize_byte_size


__all__ = [
    "UniformBuffer",
]


@register
class UniformBuffer(Buffer):
    """Replay buffer that samples (possibly independent) batched experience
    trajectories uniformly at random.

    """

    _INITIAL_CAPACITY = 1024
    _CAPACITY_MULTIPLIER = 2

    # override
    @property
    def num_experiences(self) -> "int":
        return self._num_experiences

    # override
    @property
    def num_full_trajectories(self) -> "int":
        return len(self._full_trajectory_start_idx) if self._num_experiences > 0 else 0

    @property
    def byte_size(self) -> "int":
        """Return the total size in bytes of the stored experiences.

        Returns
        -------
            The total size in bytes of the stored experiences.

        """
        return self._byte_size

    # override
    def reset(self) -> "None":
        self._experiences = {
            k: np.zeros((self._INITIAL_CAPACITY, *v["shape"]), dtype=v["dtype"])
            for k, v in self.spec.items()
        }
        self._terminals = np.zeros(self._INITIAL_CAPACITY, dtype=bool)
        self._num_experiences = self._byte_size = 0
        self._idx = None
        self._overflow = False
        self._cache = {}

    # override
    def add(self, experience: "Dict[str, ndarray]", terminal: "ndarray") -> "None":
        self._cache = {}
        batch_size = len(terminal)
        if self._idx is None:
            self._idx = np.arange(-batch_size, 0, dtype=np.int64)
        if self._batch_size != batch_size:
            raise ValueError(
                f"Batch size ({batch_size}) must be constant ({self._batch_size}) "
                f"throughout calls of method `add`"
            )
        self._idx += batch_size
        if not self._overflow:
            self._num_experiences += batch_size
            self._overflow = self._num_experiences > self.capacity
        if self._overflow:
            self._num_experiences = self.capacity
            self._idx %= self.capacity
        if self._num_experiences > len(self._terminals):
            new_size = max(
                round(len(self._terminals) * self._CAPACITY_MULTIPLIER),
                self._num_experiences,
            )
            new_size = min(new_size, self.capacity)
            self._resize(new_size)
        for k, v in experience.items():
            self._experiences[k][self._idx, ...] = v
            if not self._overflow:
                self._byte_size += getsizeof(v)
        self._terminals[self._idx] = terminal

    # override
    def sample(
        self,
        batch_size: "int",
        num_timesteps: "Union[int, float]" = 1,
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray, ndarray]":
        if batch_size < 1 or not float(batch_size).is_integer():
            raise ValueError(
                f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
            )
        if num_timesteps != float("inf"):
            if num_timesteps < 1 or not float(num_timesteps).is_integer():
                raise ValueError(
                    f"`num_timesteps` ({num_timesteps}) must be in the integer interval [1, inf]"
                )
            num_timesteps = int(num_timesteps)
        if self._num_experiences < 1:
            raise RuntimeError(
                "At least 1 experience must be added before calling method `sample`"
            )
        trajectory_start_idx, is_weight = self._compute_trajectory_start_idx(batch_size)
        idx = self._randomize_trajectory_start_idx(trajectory_start_idx, num_timesteps)
        experiences, idxes, mask = self._sample_with_idx(idx, num_timesteps)
        return experiences, idxes, is_weight, mask

    @property
    def _full_trajectory_start_idx(self) -> "ndarray":
        try:
            full_trajectory_start_idx = self._cache["full_trajectory_start_idx"]
        except KeyError:
            terminals = np.array(self._terminals)  # Copy
            terminals[self._idx] = True
            full_trajectory_start_idx = (
                np.where(terminals)[0] + self._batch_size
            ) % self._num_experiences
            self._cache["full_trajectory_start_idx"] = full_trajectory_start_idx
        return full_trajectory_start_idx

    @property
    def _trajectory_lengths(self) -> "ndarray":
        try:
            trajectory_lengths = self._cache["trajectory_lengths"]
        except KeyError:
            priorities = np.ones(self._num_experiences, dtype=np.float32)
            trajectory_lengths = compute_trajectory_priorities(
                self._idx,
                self._terminals[: self._num_experiences],
                priorities,
                self._num_experiences,
                self._batch_size,
            ).astype(np.int64)
            self._cache["trajectory_lengths"] = trajectory_lengths
        return trajectory_lengths

    @property
    def _batch_size(self) -> "int":
        return len(self._idx)

    def _resize(self, new_size: "int") -> "None":
        for k, v in self.spec.items():
            self._experiences[k].resize((new_size, *v["shape"]), refcheck=False)
        self._terminals.resize((new_size,), refcheck=False)

    def _compute_trajectory_start_idx(
        self,
        batch_size: "int",
    ) -> "Tuple[ndarray, ndarray]":
        # Select `batch_size` (possibly independent) batched
        # experience trajectories uniformly at random
        trajectory_start_idx = np.random.choice(
            self._full_trajectory_start_idx,
            batch_size,
            replace=self.num_full_trajectories < batch_size,
        )
        is_weight = np.ones(batch_size, dtype=np.float32)
        return trajectory_start_idx, is_weight

    def _randomize_trajectory_start_idx(
        self,
        trajectory_start_idx: "ndarray",
        num_timesteps: "Union[int, float]",
    ) -> "ndarray":
        # For each trajectory, select a random start index
        trajectory_length = self._trajectory_lengths[trajectory_start_idx]
        offset = self._batch_size * np.random.randint(
            0, (trajectory_length - num_timesteps).clip(1, None)
        )
        idx = trajectory_start_idx + offset
        if self._overflow:
            idx %= self.capacity
        return idx

    def _sample_with_idx(
        self,
        idx: "ndarray",
        num_timesteps: "Union[int, float]",
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray]":
        trajectory_length = self._trajectory_lengths[idx]
        max_trajectory_length = min(num_timesteps, trajectory_length.max())
        max_trajectory_stop_idx = idx + self._batch_size * (max_trajectory_length - 1)
        idxes = np.linspace(
            idx,
            max_trajectory_stop_idx,
            max_trajectory_length,
            dtype=np.int64,
            axis=1,
        )
        mask = (
            idxes
            < (idx + self._batch_size * trajectory_length.clip(None, num_timesteps))[
                ..., None
            ]
        )
        idxes[~mask] = -1
        if self._overflow:
            idxes %= self.capacity
        experiences = {k: v[idxes] for k, v in self._experiences.items()}
        return experiences, idxes, mask

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(capacity: {self.capacity}, "
            f"num_experiences: {self.num_experiences}, "
            f"num_full_trajectories: {self.num_full_trajectories}, "
            f"size: {normalize_byte_size(self.byte_size)})"
        )
