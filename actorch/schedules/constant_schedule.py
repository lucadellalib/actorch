# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
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
