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

"""Linear schedule."""

from typing import Sequence, Union

import numpy as np
from numpy import ndarray

from actorch.schedules.schedule import Schedule


__all__ = [
    "LinearSchedule",
]


class LinearSchedule(Schedule):
    """Linear schedule."""

    _STATE_VARS = Schedule._STATE_VARS + [
        "initial_value",
        "final_value",
        "num_timesteps",
        "_delta",
        "_is_increasing",
        "_value",
    ]  # override

    # override
    def __init__(
        self,
        initial_value: "Union[int, float, Sequence[Union[int, float]], ndarray]",
        final_value: "Union[int, float, Sequence[Union[int, float]], ndarray]",
        num_timesteps: "Union[int, Sequence[int], ndarray]",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        initial_value:
            The (possibly batched) initial value.
        final_value:
            The (possibly batched) final value.
        num_timesteps:
            The (possibly batched) number of timesteps
            after which the value is set to `final_value`.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        try:
            initial_value, final_value, num_timesteps = [
                np.array(v)
                for v in np.broadcast_arrays(
                    initial_value,
                    final_value,
                    num_timesteps,
                )  # Copy
            ]
        except Exception:
            raise ValueError(
                f"`initial_value` ({initial_value}), `final_value` ({final_value}) and "
                f"`num_timesteps` ({num_timesteps}) must be broadcastable to a single shape"
            )
        if ((num_timesteps < 1) | (num_timesteps % 1 != 0)).any():
            raise ValueError(
                f"`num_timesteps` ({num_timesteps}) must be in the integer interval [1, inf)"
            )
        batch_size = (initial_value.shape or [None])[0]
        self.initial_value = np.array(initial_value, copy=False, ndmin=1)
        self.final_value = np.array(final_value, copy=False, ndmin=1)
        self.num_timesteps = np.array(num_timesteps, copy=False, ndmin=1)
        self._delta = (self.final_value - self.initial_value) / self.num_timesteps
        self._is_increasing = self._delta >= 0
        self._value = np.zeros(batch_size or 1)
        super().__init__(batch_size)

    # override
    def _reset(self, mask: "ndarray") -> "None":
        self._value[mask] = self.initial_value[mask]

    # override
    def _step(self, mask: "ndarray") -> "None":
        self._value += np.where(mask, self._delta, 0.0)
        min_value = self._value.clip(max=self.final_value)
        max_value = self._value.clip(min=self.final_value)
        self._value = np.where(self._is_increasing, min_value, max_value)

    # override
    def __call__(self) -> "Union[int, float, ndarray]":
        value = np.array(self._value)  # Copy
        return value if self.batch_size else value.item()

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(initial_value: {self.initial_value}, "
            f"final_value: {self.final_value}, "
            f"num_timesteps: {self.num_timesteps})"
        )
