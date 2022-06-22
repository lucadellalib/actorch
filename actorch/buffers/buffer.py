# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Experience replay buffer."""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from actorch.schedules import Schedule


__all__ = [
    "Buffer",
]


class Buffer(ABC):
    """Replay buffer that stores and samples batched experience trajectories."""

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
        self._schedules = {}

    @property
    def schedules(self) -> "Dict[str, Schedule]":
        """Return the buffer public schedules.

        Returns
        -------
            The buffer public schedules, i.e. a dict that maps
            names of the schedules to the schedules themselves.

        """
        return self._schedules

    def state_dict(self) -> "Dict[str, Any]":
        """Return the buffer state dict.

        Returns
        -------
            The buffer state dict.

        """
        return {
            k: v.state_dict() if hasattr(v, "state_dict") else v
            for k, v in self.__dict__.items()
        }

    def load_state_dict(self, state_dict: "Dict[str, Any]") -> "None":
        """Load a state dict into the buffer.

        Parameters
        ----------
        state_dict:
            The state dict.

        Raises
        ------
        KeyError
            If `state_dict` contains unknown keys.

        """
        for k, v in state_dict.items():
            if k not in self.__dict__:
                raise KeyError(f"{k}")
            if hasattr(self.__dict__[k], "load_state_dict"):
                self.__dict__[k].load_state_dict(v)
            else:
                self.__dict__[k] = v

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
            at the current timestep (True), and which do not (False),
            shape: ``[B]``. If a scalar, it is converted to a 1D array.

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
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray, ndarray]":
        """Sample batched experience trajectories from the buffer.

        Parameters
        ----------
        batch_size:
            The batch size.
        max_trajectory_length:
            The maximum number of experiences to sample from each trajectory.
            Set to ``float("inf")`` to sample full trajectories.

        Returns
        -------
            - The sampled batched experiences, i.e. a dict whose key-value pairs
              are consistent with the given `spec` initialization argument, shape of
              ``experiences[name]``: ``[batch_size, max_trajectory_length, *spec[name]["shape"]]``;
            - the sampled batched indices, shape ``[batch_size, max_trajectory_length]``;
            - the batched importance sampling weight, shape ``[batch_size]``;
            - the mask, i.e. the boolean array indicating which trajectory
              elements are valid (True) and which are not (False),
              shape ``[batch_size, max_trajectory_length]``.

        Raises
        ------
        ValueError:
            If an invalid argument value is given.
        RuntimeError:
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
        return self._sample(batch_size, max_trajectory_length)

    def update_priority(
        self,
        idx: "Union[int, Sequence[int], ndarray]",
        priority: "Union[float, Sequence[float], ndarray]",
    ) -> "None":
        """Update the `idx`-th batched priority.

        In the following, let `B` denote the batch size.

        Parameters
        ----------
        idx:
            The batched index, shape: ``[B]``.
            If a scalar, it is converted to a 1D array.
        priority:
            The new batched priority, shape: ``[B]``.
            If a scalar, it is converted to a 1D array.

        Raises
        ------
        IndexError
            If `idx` is out of range or not integer.
        ValueError
            If `priority` is not in the interval [0, inf).

        """
        idx = np.array(idx, copy=False, ndmin=1)
        if (
            (idx >= self.num_experiences)
            | (idx < -self.num_experiences)
            | (idx % 1 != 0)
        ).any():
            raise IndexError(
                f"`idx` ({idx}) must be in the integer interval "
                f"[-{self.num_experiences}, {self.num_experiences})"
            )
        priority = np.array(priority, dtype=np.float32, copy=False, ndmin=1)
        if (priority < 0.0).any():
            raise ValueError(
                f"`priority` ({priority}) must be in the interval [0, inf)"
            )
        idx = idx.astype(np.int64)
        return self._update_priority(idx, priority)

    def __len__(self) -> "int":
        return self.num_experiences

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(capacity: {self.capacity}, "
            f"num_experiences: {self.num_experiences}, "
            f"num_full_trajectories: {self.num_full_trajectories})"
        )

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
    def reset(self) -> "None":
        """Reset the buffer state."""
        raise NotImplementedError

    @abstractmethod
    def _add(
        self,
        experience: "Dict[str, ndarray]",
        terminal: "ndarray",
    ) -> "None":
        """See documentation of `add'."""
        raise NotImplementedError

    @abstractmethod
    def _sample(
        self,
        batch_size: "int",
        max_trajectory_length: "Union[int, float]" = 1,
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray, ndarray]":
        """See documentation of `sample'."""
        raise NotImplementedError

    def _update_priority(
        self,
        idx: "ndarray",
        priority: "ndarray",
    ) -> "None":
        """See documentation of `update_priority'."""
        pass
