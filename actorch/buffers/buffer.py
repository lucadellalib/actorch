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

"""Experience replay buffer."""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from actorch.schedules import Schedule
from actorch.utils import CheckpointableMixin


__all__ = [
    "Buffer",
]


class Buffer(ABC, CheckpointableMixin):
    """Replay buffer that stores and samples batched experience trajectories."""

    _STATE_VARS = ["capacity", "spec", "_sampled_idx"]  # override

    def __init__(
        self,
        capacity: "Union[int, float]",
        spec: "Dict[str, Dict[str, Any]]",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        capacity:
            The maximum number of experiences to store. When the
            buffer overflows, old experiences are removed (FIFO).
            Set to ``float("inf")`` for unlimited capacity.
        spec:
            The specification, i.e. a dict that maps names of the values
            to store to dicts with the following key-value pairs:
            - "shape": the event shape of the value to store. Default to ``()``;
            - "dtype": the dtype of the value to store. Default to ``np.float32``.

        Raises
        ------
        ValueError
            If `capacity` is not in the integer interval [1, inf].

        """
        if capacity != float("inf"):
            if capacity < 1 or not float(capacity).is_integer():
                raise ValueError(
                    f"`capacity` ({capacity}) must be in the integer interval [1, inf]"
                )
            capacity = int(capacity)
        self.capacity = capacity
        self.spec = {
            k: {
                "shape": v.get("shape", ()),
                "dtype": v.get("dtype", np.float32),
            }
            for k, v in spec.items()
        }
        self.reset()
        self._schedules: "Dict[str, Schedule]" = {}

    @property
    def schedules(self) -> "Dict[str, Schedule]":
        """Return the buffer public schedules.

        Returns
        -------
            The buffer public schedules, i.e. a dict that maps
            names of the schedules to the schedules themselves.

        """
        return self._schedules

    def reset(self) -> "None":
        """Reset the buffer state."""
        return self._reset()

    def add(
        self,
        experience: "Dict[str, Union[Number, Sequence[Number], ndarray]]",
        terminal: "Union[bool, Sequence[bool], ndarray]",
    ) -> "None":
        """Add a batched experience to the buffer.

        In the following, let `B` denote the batch size.

        Parameters
        ----------
        experience:
            The batched experience, i.e. a dict whose key-value pairs
            are consistent with the given `spec` initialization argument,
            shape of ``experience[name]``: ``[B, *spec[name]["shape"]]``.
            If a scalar, it is converted to a 1D array.
        terminal:
            The boolean array indicating which trajectories terminate
            at the current timestep (True) and which do not (False),
            shape: ``[B]``.
            If a scalar, it is converted to a 1D array.

        Raises
        ------
        ValueError
            If lengths of `experience` are not identical or not equal
            to the length of `terminal`.

        """
        terminal = np.array(terminal, copy=False, ndmin=1)
        batch_size = len(terminal)
        for key, value in experience.items():
            experience[key] = value = np.array(value, copy=False, ndmin=1)
            if len(value) != batch_size:
                lengths = {k: len(v) for k, v in experience.items()}
                raise ValueError(
                    f"Lengths of `experience` ({lengths}) must be identical "
                    f"and equal to the length of `terminal` ({batch_size})"
                )
        return self._add(experience, terminal)

    def sample(
        self,
        batch_size: "int",
        max_trajectory_length: "Union[int, float]" = 1,
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray]":
        """Sample a batched experience trajectory from the buffer.

        Parameters
        ----------
        batch_size:
            The batch size.
        max_trajectory_length:
            The maximum number of experiences to sample from each trajectory.
            Set to ``float("inf")`` to sample full trajectories.

        Returns
        -------
            - The batched experiences, i.e. a dict whose key-value pairs are
              consistent with the given `spec` initialization argument, shape of
              ``experiences[name]``: ``[batch_size, max_trajectory_length, *spec[name]["shape"]]``;
            - the batched importance sampling weight, shape: ``[batch_size]``;
            - the mask, i.e. the boolean array indicating which trajectory
              elements are valid (True) and which are not (False),
              shape: ``[batch_size, max_trajectory_length]``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.
        RuntimeError
            If the buffer is empty.

        Warnings
        --------
        If none of the sampled trajectories is long enough, the actual
        trajectory length will be less than `max_trajectory_length`.

        """
        if batch_size < 1 or not float(batch_size).is_integer():
            raise ValueError(
                f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
            )
        batch_size = int(batch_size)
        if max_trajectory_length != float("inf"):
            if (
                max_trajectory_length < 1
                or not float(max_trajectory_length).is_integer()
            ):
                raise ValueError(
                    f"`max_trajectory_length` ({max_trajectory_length}) "
                    f"must be in the integer interval [1, inf]"
                )
            max_trajectory_length = int(max_trajectory_length)
        if self.num_experiences < 1:
            raise RuntimeError(
                "At least 1 experience must be added before calling `sample`"
            )
        experiences, idxes, is_weight, mask = self._sample(
            batch_size,
            max_trajectory_length,
        )
        self._sampled_idx = idxes[mask]
        return experiences, is_weight, mask

    def update_priority(
        self,
        priority: "Union[float, Sequence[float], ndarray]",
    ) -> "None":
        """Update the batched priority of the last sampled
        masked batched experiences.

        In the following, let `B` denote the effective batch
        size (i.e. the batch size after masking).

        Parameters
        ----------
        priority:
            The new batched priority, shape: ``[B]``.
            If a scalar, it is converted to a 1D array.

        Raises
        ------
        ValueError
            If an invalid argument value is given.
        RuntimeError
            If `sample` has not been called at least once
            before calling `update_priority`.

        Warnings
        --------
        The batched index of the last sampled masked batched
        experiences might contain duplicates; in such case the
        corresponding batched priority is updated according
        to NumPy indexing semantics.

        """
        priority = np.array(priority, dtype=np.float32, copy=False, ndmin=1)
        if (priority < 0.0).any():
            raise ValueError(
                f"`priority` ({priority}) must be in the interval [0, inf)"
            )
        if self._sampled_idx is None:
            raise RuntimeError(
                "`sample` must be called at least once before calling `update_priority`"
            )
        if len(priority) != len(self._sampled_idx):
            raise ValueError(
                f"Length of `priority` ({len(priority)}) must be equal to "
                f"the length of the last sampled masked batched experiences "
                f"({len(self._sampled_idx)})"
            )
        return self._update_priority(self._sampled_idx, priority)

    def __len__(self) -> "int":
        return self.num_experiences

    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(capacity: {self.capacity}, "
            f"spec: {self.spec}, "
            f"num_experiences: {self.num_experiences}, "
            f"num_full_trajectories: {self.num_full_trajectories})"
        )

    def _reset(self) -> "None":
        """See documentation of `reset`."""
        self._sampled_idx = None

    def _update_priority(
        self,
        idx: "ndarray",
        priority: "ndarray",
    ) -> "None":
        """See documentation of `update_priority`."""
        pass

    @property
    @abstractmethod
    def num_experiences(self) -> "int":
        """Return the number of stored experiences.

        Returns
        -------
            The number of stored experiences.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_full_trajectories(self) -> "int":
        """Return the number of stored full trajectories.

        Returns
        -------
            The number of stored full trajectories.

        """
        raise NotImplementedError

    @abstractmethod
    def _add(
        self,
        experience: "Dict[str, ndarray]",
        terminal: "ndarray",
    ) -> "None":
        """See documentation of `add`."""
        raise NotImplementedError

    @abstractmethod
    def _sample(
        self,
        batch_size: "int",
        max_trajectory_length: "Union[int, float]" = 1,
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray, ndarray]":
        """See documentation of `sample`."""
        raise NotImplementedError
