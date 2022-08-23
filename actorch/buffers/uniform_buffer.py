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

"""Uniform experience replay buffer."""

from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray

from actorch.buffers.buffer import Buffer
from actorch.buffers.utils import compute_trajectory_priorities


__all__ = [
    "UniformBuffer",
]


class UniformBuffer(Buffer):
    """Replay buffer that samples (possibly independent) batched
    experience trajectories uniformly at random.

    """

    _STATE_VARS = Buffer._STATE_VARS + [
        "_experiences",
        "_terminals",
        "_num_experiences",
        "_idx",
        "_overflow",
        "_cache",
    ]  # override
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

    # override
    def _reset(self) -> "None":
        super()._reset()
        initial_capacity = min(self.capacity, self._INITIAL_CAPACITY)
        self._experiences = {
            k: np.zeros((initial_capacity, *v["shape"]), dtype=v["dtype"])
            for k, v in self.spec.items()
        }
        self._terminals = np.zeros(initial_capacity, dtype=bool)
        self._num_experiences = 0
        self._idx = None
        self._overflow = False
        self._cache: "Dict[str, ndarray]" = {}

    # override
    def _add(
        self,
        experience: "Dict[str, ndarray]",
        terminal: "ndarray",
    ) -> "None":
        # Invalidate cache
        self._cache.clear()
        batch_size = len(terminal)
        if self._idx is None:
            if batch_size > self.capacity:
                raise ValueError(
                    f"Batch size ({batch_size}) must be in the integer interval "
                    f"[1, `capacity` initialization argument ({self.capacity})]"
                )
            self._idx = np.arange(-batch_size, 0, dtype=np.int64)
        if batch_size != self._batch_size:
            raise ValueError(
                f"Batch size ({batch_size}) must remain equal to its initial "
                f"value ({self._batch_size}) throughout calls of `add`"
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
        self._terminals[self._idx] = terminal

    # override
    def _sample(
        self,
        batch_size: "int",
        max_trajectory_length: "Union[int, float]" = 1,
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray, ndarray]":
        trajectory_start_idx, is_weight = self._compute_trajectory_start_idx(batch_size)
        idx = self._randomize_trajectory_start_idx(
            trajectory_start_idx, max_trajectory_length
        )
        experiences, idxes, mask = self._sample_with_idx(idx, max_trajectory_length)
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
                self._batch_size,
                self._num_experiences,
                self._idx,
                self._terminals[: self._num_experiences],
                priorities[: self._num_experiences],
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
        # Select batch_size (possibly independent) batched
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
        max_trajectory_length: "Union[int, float]",
    ) -> "ndarray":
        # For each trajectory, select a random start index
        trajectory_length = self._trajectory_lengths[trajectory_start_idx]
        offset = self._batch_size * np.random.randint(
            0, (trajectory_length - max_trajectory_length).clip(min=1)
        )
        idx = trajectory_start_idx + offset
        if self._overflow:
            idx %= self.capacity
        return idx

    def _sample_with_idx(
        self,
        idx: "ndarray",
        max_trajectory_length: "Union[int, float]",
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray]":
        trajectory_length = self._trajectory_lengths[idx]
        max_trajectory_length = min(max_trajectory_length, trajectory_length.max())
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
            < (
                idx
                + self._batch_size * trajectory_length.clip(max=max_trajectory_length)
            )[..., None]
        )
        idxes[~mask] = -1
        if self._overflow:
            idxes %= self.capacity
        experiences = {k: v[idxes] for k, v in self._experiences.items()}  # Copy
        return experiences, idxes, mask
