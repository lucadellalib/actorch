# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Exponential schedule."""

from typing import Any, Dict, Sequence, Union

import numpy as np
from numpy import ndarray

from actorch.schedules.lambda_schedule import LambdaSchedule


__all__ = [
    "ExponentialSchedule",
]


class ExponentialSchedule(LambdaSchedule):
    """Exponential schedule."""

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
            If an invalid argument value is provided.

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
        if (initial_value * final_value <= 0.0).any():
            raise ValueError(
                f"`initial_value` ({initial_value}) and `final_value` ({final_value}) "
                f"must be in the interval (-inf, 0) or (0, inf)"
            )
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_timesteps = num_timesteps
        rate = (final_value / initial_value) ** (1 / num_timesteps)
        batch_size = (initial_value.shape or [None])[0]

        def step_fn(
            num_elapsed_timesteps: Union[int, ndarray],
        ) -> "Union[int, float, ndarray]":
            value = np.where(
                num_elapsed_timesteps < num_timesteps,
                initial_value * (rate ** num_elapsed_timesteps),
                final_value,
            )
            return value if batch_size else value.item()

        super().__init__(step_fn, batch_size)

    # override
    def state_dict(self) -> "Dict[str, Any]":
        state_dict = super().state_dict()
        del state_dict["step_fn"]
        return state_dict

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(initial_value: {self.initial_value}, "
            f"final_value: {self.final_value}, "
            f"num_timesteps: {self.num_timesteps})"
        )
