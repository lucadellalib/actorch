# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Base-to-decimal processor."""

from typing import Tuple

import torch
from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "Base2Dec",
]


class Base2Dec(Processor):
    """Convert a tensor from a given (possibly mixed) base to base 10."""

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
        self._in_shape = torch.Size([len(self.base)])
        self._out_shape = torch.Size([1])
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
        if (
            (input < 0) | (input >= self.base.expand_as(input)) | (input != input.int())
        ).any():
            raise ValueError(
                f"`input` ({input}) must be in the integer interval "
                f"[{torch.zeros_like(self.base)}, {self.base})"
            )
        return (self._weights * input).sum(dim=-1).int()

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}" f"(base: {self.base})"
