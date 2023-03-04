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

"""Rank-based experience replay buffer."""

from typing import Any, Dict, Union

import numpy as np
from numpy import ndarray

from actorch.buffers.proportional_buffer import ProportionalBuffer
from actorch.schedules import Schedule


__all__ = [
    "RankBasedBuffer",
]


class RankBasedBuffer(ProportionalBuffer):
    """Prioritized replay buffer that samples (possibly independent) batched
    experience trajectories with probabilities proportional to their ranks.

    References
    ----------
    .. [1] T. Schaul, J. Quan, I. Antonoglou, and D. Silver.
           "Prioritized Experience Replay".
           In: ICLR. 2016.
           URL: https://arxiv.org/abs/1511.05952

    """

    _STATE_VARS = ProportionalBuffer._STATE_VARS  # override
    _STATE_VARS.remove("epsilon")

    # override
    def __init__(
        self,
        capacity: "Union[int, float]",
        spec: "Dict[str, Dict[str, Any]]",
        prioritization: "Union[float, Schedule]" = 1.0,
        bias_correction: "Union[float, Schedule]" = 0.4,
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
        prioritization:
            The schedule for the prioritization coefficient (`alpha` in the
            literature; 0: no prioritization, 1: full prioritization).
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
        bias_correction:
            The schedule for the bias correction coefficient (`beta` in the
            literature; 0: no correction, 1: full correction).
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.

        """
        super().__init__(capacity, spec, prioritization, bias_correction)
        del self.epsilon

    # override
    def _add(
        self,
        experience: "Dict[str, ndarray]",
        terminal: "ndarray",
    ) -> "None":
        super()._add(experience, terminal)
        self._priorities[self._idx] = self._max_priority

    # override
    def _update_priority(
        self,
        idx: "ndarray",
        priority: "ndarray",
    ) -> "None":
        # Invalidate priority cache
        self._cache.pop("trajectory_priorities", None)
        self._cache.pop("full_trajectory_priority", None)
        self._priorities[idx] = priority
        self._max_priority = max(self._max_priority, priority.max())

    # override
    @property
    def _full_trajectory_priority(self) -> "ndarray":
        try:
            full_trajectory_priority = self._cache["full_trajectory_priority"]
        except KeyError:
            prioritization = self.prioritization()
            if prioritization < 0.0 or prioritization > 1.0:
                raise ValueError(
                    f"`prioritization` ({prioritization}) must be in the interval [0, 1]"
                )
            tmp = super()._full_trajectory_priority.argsort()[::-1]
            full_trajectory_rank = np.empty_like(tmp)
            full_trajectory_rank[tmp] = np.arange(self.num_full_trajectories) + 1
            full_trajectory_priority = full_trajectory_rank ** (-prioritization)
            self._cache["full_trajectory_priority"] = full_trajectory_priority
        return full_trajectory_priority

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(capacity: {self.capacity}, "
            f"spec: {self.spec}, "
            f"prioritization: {self.prioritization}, "
            f"bias_correction: {self.bias_correction}, "
            f"num_experiences: {self.num_experiences}, "
            f"num_full_trajectories: {self.num_full_trajectories})"
        )
