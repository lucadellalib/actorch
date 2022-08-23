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

"""Constant schedule."""

from typing import Sequence, Union

import numpy as np
from numpy import ndarray

from actorch.schedules.schedule import Schedule


__all__ = [
    "ConstantSchedule",
]


class ConstantSchedule(Schedule):
    """Constant schedule."""

    _STATE_VARS = Schedule._STATE_VARS + ["value"]  # override

    # override
    def __init__(
        self,
        value: "Union[int, float, Sequence[Union[int, float]], ndarray]",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        value:
            The (possibly batched) constant value to return.

        """
        self.value = value
        batch_size = (np.asarray(value).shape or [None])[0]
        super().__init__(batch_size)

    # override
    def __call__(self) -> "Union[int, float, ndarray]":
        value = np.array(self.value)  # Copy
        return value if self.batch_size else value.item()

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(value: {self.value})"
