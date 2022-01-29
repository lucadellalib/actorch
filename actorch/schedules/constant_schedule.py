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

    def __call__(self) -> "Union[int, float, ndarray]":
        value = np.array(self.value)  # Copy
        return value if self.batch_size else value.item()

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(value: {self.value})"
