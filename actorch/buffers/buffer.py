# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Experience replay buffer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

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
            - "shape": the shape of the value to store. Default to ``()``;
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

    @property
    def schedules(self) -> "List[Schedule]":
        """Return the buffer public schedules.

        Returns
        -------
            The buffer public schedules.

        """
        return []

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
    def add(
        self,
        experience: "Dict[str, ndarray]",
        terminal: "ndarray",
    ) -> "None":
        """Add a batched experience to the buffer.

        In the following, let `B` denote the batch size.

        Parameters
        ----------
        experience:
            The batched experience, i.e. a dict whose key-value pairs
            are consistent with the given `spec` initialization argument,
            shape of ``experience[name]``: ``[B, *spec[name]["shape"]]``.
        terminal:
            The boolean array indicating which trajectories terminate
            at the current timestep (True), and which do not (False),
            shape: ``[B]``.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        batch_size: "int",
        trajectory_length: "Union[int, float]" = 1,
    ) -> "Tuple[Dict[str, ndarray], ndarray, ndarray, ndarray]":
        """Sample batched experience trajectories from the buffer.

        Parameters
        ----------
        batch_size:
            The batch size.
        trajectory_length:
            The number of experiences to sample from each trajectory.
            Set to ``float("inf")`` to sample full trajectories.

        Returns
        -------
            - The sampled batched experiences, i.e. a dict whose key-value pairs
              are consistent with the given `spec` initialization argument, shape of
              ``experiences[name]``: ``[batch_size, trajectory_length, *spec[name]["shape"]]``;
            - the sampled batched indices, shape ``[batch_size, trajectory_length]``;
            - the batched importance sampling weight, shape ``[batch_size]``;
            - the mask, i.e. the boolean array indicating which trajectory
              elements are valid (True) and which are not (False),
              shape ``[batch_size, trajectory_length]``.

        """
        raise NotImplementedError

    def update_priority(self, idx: "ndarray", priority: "ndarray") -> "None":
        """Update the `idx`-th batched priority.

        In the following, let `B` denote the batch size.

        Parameters
        ----------
        idx:
            The batched index, shape: ``[B]``.
        priority:
            The new batched priority, shape: ``[B]``.

        """
        pass

    def __len__(self) -> "int":
        return self.num_experiences

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(capacity: {self.capacity}, "
            f"num_experiences: {self.num_experiences}, "
            f"num_full_trajectories: {self.num_full_trajectories})"
        )
