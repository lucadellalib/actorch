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

"""Lambda schedule."""

from typing import Callable, Optional, Union

import numpy as np
from numpy import ndarray

from actorch.schedules.schedule import Schedule


__all__ = [
    "LambdaSchedule",
]


class LambdaSchedule(Schedule):
    """Schedule defined by a given function."""

    _STATE_VARS = Schedule._STATE_VARS + [
        "_num_elapsed_timesteps",
        "_value",
    ]  # override

    # override
    def __init__(
        self,
        step_fn: "Callable[[Union[int, ndarray]], Union[int, float, ndarray]]",
        batch_size: "Optional[int]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        step_fn:
            The function that defines the time-dependent scheduling schema.
            It receives as an argument the (possibly batched) number of elapsed
            timesteps and returns the corresponding (possibly batched) schedule value.
        batch_size:
            The batch size.

        """
        self.step_fn = step_fn
        self._num_elapsed_timesteps = np.zeros(batch_size or 1, dtype=np.int64)
        super().__init__(batch_size)

    # override
    def _reset(self, mask: "ndarray") -> "None":
        self._num_elapsed_timesteps[mask] = -1
        self.step(mask)

    # override
    def _step(self, mask: "ndarray") -> "None":
        self._num_elapsed_timesteps[mask] += 1
        self._value = self.step_fn(self._num_elapsed_timesteps)

    # override
    def __call__(self) -> "Union[int, float, ndarray]":
        value = np.array(self._value)  # Copy
        return value if self.batch_size else value.item()

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(step_fn: {self.step_fn}, "
            f"batch_size: {self.batch_size})"
        )
