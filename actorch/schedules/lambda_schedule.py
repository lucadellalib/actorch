# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
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
            This function receives as an argument the (possibly batched) number of
            elapsed timesteps, and returns the corresponding (possibly batched)
            schedule value.
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

    def __call__(self) -> "Union[int, float, ndarray]":
        value = np.array(self._value)  # Copy
        return value if self.batch_size else value.item()

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(step_fn: {self.step_fn}, "
            f"batch_size: {self.batch_size})"
        )
