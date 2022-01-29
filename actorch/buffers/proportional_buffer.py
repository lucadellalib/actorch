# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Proportional experience replay buffer."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from actorch.buffers.uniform_buffer import UniformBuffer
from actorch.buffers.utils import compute_trajectory_priorities
from actorch.registry import register
from actorch.schedules import ConstantSchedule, Schedule
from actorch.utils import normalize_byte_size


__all__ = [
    "ProportionalBuffer",
]


@register
class ProportionalBuffer(UniformBuffer):
    """Prioritized replay buffer that samples (possibly independent) batched
    trajectories with probabilities proportional to their mean priorities.

    References
    ----------
    .. [1] T. Schaul, J. Quan, I. Antonoglou, and D. Silver.
           "Prioritized Experience Replay". In: ICLR. 2016.
           URL: https://arxiv.org/abs/1511.05952

    """

    def __init__(
        self,
        capacity: "Union[int, float]",
        spec: "Dict[str, Dict[str, Any]]",
        prioritization: "Optional[Schedule]" = None,
        bias_correction: "Optional[Schedule]" = None,
        epsilon: "float" = 1e-5,
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
        prioritization:
            The prioritization schedule (`alpha` in the literature;
            0: no prioritization, 1: full prioritization).
            Default to ``ConstantSchedule(1.0)``.
        bias_correction:
            The bias correction schedule (`beta` in the literature;
            0: no correction, 1: full correction).
            Default to ``ConstantSchedule(0.4)``.
        epsilon:
            The term added to the priorities to ensure a
            non-zero probability of sampling each trajectory.

        Raises
        ------
        ValueError
            If an invalid argument value is provided.

        """
        if epsilon <= 0.0:
            raise ValueError(f"`epsilon` ({epsilon}) must be in the interval (0, inf)")
        self.prioritization = prioritization or ConstantSchedule(1.0)
        self.bias_correction = bias_correction or ConstantSchedule(0.4)
        self.epsilon = epsilon
        super().__init__(capacity, spec)

    # override
    @property
    def schedules(self) -> "List[Schedule]":
        return [self.prioritization, self.bias_correction]

    # override
    def reset(self) -> "None":
        super().reset()
        self._priorities = np.zeros_like(self._terminals, dtype=np.float32)
        self._max_priority = 1.0

    # override
    def add(self, experience: "Dict[str, ndarray]", terminal: "ndarray") -> "None":
        super().add(experience, terminal)
        prioritization = self.prioritization()
        if prioritization < 0.0 or prioritization > 1.0:
            raise ValueError(
                f"`prioritization` ({prioritization}) must be in the interval [0, 1]"
            )
        self._priorities[self._idx] = self._max_priority ** prioritization

    # override
    def update_priority(self, idx: "ndarray", priority: "ndarray") -> "None":
        if ((idx >= self._num_experiences) | (idx < -self._num_experiences)).any():
            raise IndexError(
                f"`idx` ({idx}) must be in the integer interval "
                f"[-{self._num_experiences}, {self._num_experiences})"
            )
        if (priority < 0.0).any():
            raise ValueError(
                f"`priority` ({priority}) must be in the interval [0, inf)"
            )
        prioritization = self.prioritization()
        if prioritization < 0.0 or prioritization > 1.0:
            raise ValueError(
                f"`prioritization` ({prioritization}) must be in the interval [0, 1]"
            )
        # Invalidate priority cache
        self._cache.pop("trajectory_priorities", None)
        priority += self.epsilon
        self._priorities[idx] = priority ** prioritization
        self._max_priority = max(self._max_priority, priority.max())

    @property
    def _trajectory_priorities(self) -> "ndarray":
        try:
            trajectory_priorities = self._cache["trajectory_priorities"]
        except KeyError:
            trajectory_priorities = compute_trajectory_priorities(
                self._idx,
                self._terminals[: self._num_experiences],
                self._priorities[: self._num_experiences],
                self._num_experiences,
                self._batch_size,
            )
            trajectory_priorities /= self._trajectory_lengths
            self._cache["trajectory_priorities"] = trajectory_priorities
        return trajectory_priorities

    @property
    def _full_trajectory_priority(self) -> "ndarray":
        return self._trajectory_priorities[self._full_trajectory_start_idx]

    # override
    def _resize(self, new_size: "int") -> "None":
        super()._resize(new_size)
        self._priorities.resize((new_size,), refcheck=False)

    # override
    def _compute_trajectory_start_idx(
        self,
        batch_size: "int",
    ) -> "Tuple[ndarray, ndarray]":
        # Select `batch_size` (possibly independent) trajectories based on their priorities
        bias_correction = self.bias_correction()
        if bias_correction < 0.0 or bias_correction > 1.0:
            raise ValueError(
                f"`bias_correction` ({bias_correction}) must be in the interval [0, 1]"
            )
        trajectory_priority = self._full_trajectory_priority
        cdf = trajectory_priority.cumsum()
        threshold = np.random.random(batch_size) * cdf[-1]
        trajectory_idx = (np.searchsorted(cdf, threshold, side="right") - 1).clip(
            0, None
        )
        trajectory_start_idx = self._full_trajectory_start_idx[trajectory_idx]
        min_prob = cdf.min() / cdf[-1]
        prob = cdf[trajectory_idx] / cdf[-1]
        max_is_weight = (min_prob * self.num_full_trajectories) ** (-bias_correction)
        is_weight = (prob * self.num_full_trajectories) ** (
            -bias_correction
        ) / max_is_weight
        return trajectory_start_idx, is_weight

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(capacity: {self.capacity}, "
            f"prioritization: {self.prioritization}, "
            f"bias_correction: {self.bias_correction}, "
            f"epsilon: {self.epsilon}, "
            f"num_experiences: {self.num_experiences}, "
            f"num_full_trajectories: {self.num_full_trajectories}, "
            f"size: {normalize_byte_size(self.byte_size)})"
        )
