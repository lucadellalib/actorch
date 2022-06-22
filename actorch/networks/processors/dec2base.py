# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Decimal-to-base processor."""

from typing import Tuple

import torch
from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "Dec2Base",
]


class Dec2Base(Processor):
    """Convert a tensor from base 10 to a given (possibly mixed) base."""

    def __init__(self, base: "Tuple[int, ...]") -> "None":
        """Initialize the object.

        Parameters
        ----------
        base:
            The (possibly mixed) base.

        Raises
        ------
        ValueError
            If elements of `base` are not in the integer interval [1, inf).

        """
        self.base = []
        for x in base:
            if x < 1 or not float(x).is_integer():
                raise ValueError(
                    f"Elements of `base` ({base}) must be "
                    f"in the integer interval [1, inf)"
                )
            self.base.append(x)
        self.base = torch.as_tensor(self.base)
        self._in_shape = torch.Size([1])
        self._out_shape = torch.Size([len(self.base)])
        self._max_value = self.base.prod(dim=-1)
        weights = self.base.flip(dims=(-1,)).roll(1, dims=-1)
        weights[..., 0] = 1
        weights = weights.cumprod(dim=-1).flip(dims=(-1,)).movedim(-1, 0)
        self._weights = weights

    @property
    def in_shape(self) -> "Size":
        return self._in_shape

    @property
    def out_shape(self) -> "Size":
        return self._out_shape

    def __call__(self, input: "Tensor") -> "Tensor":
        if ((input < 0) | (input >= self._max_value) | (input != input.int())).any():
            raise ValueError(
                f"`input` ({input}) must be in the integer "
                f"interval [0, {self._max_value})"
            )
        digits = torch.zeros(*input.shape, len(self._weights.shape) + 1)
        digits[..., -1] = input
        for i, weight in enumerate(self._weights):
            quotient = digits[..., -(i + 1)].div(weight, rounding_mode="trunc")
            remainder = digits[..., -(i + 1)] - (weight * quotient)
            digits[..., -(i + 1)], digits[..., -(i + 2)] = quotient, remainder
        return digits[..., 1:].flip(dims=(-1,)).int()

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}" f"(base: {self.base})"
