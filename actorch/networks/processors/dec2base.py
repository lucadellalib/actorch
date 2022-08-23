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

"""Decimal-to-base processor."""

from typing import Tuple

import torch
from torch import Size, Tensor

from actorch.networks.processors import base2dec as b2d  # Avoid circular import
from actorch.networks.processors.processor import Processor


__all__ = [
    "Dec2Base",
]


class Dec2Base(Processor):
    """Convert a tensor from base 10 to a given (possibly mixed) base."""

    # override
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
        self.base = tuple(self.base)
        self._tensor_base = torch.as_tensor(self.base)
        self._in_shape = Size([])
        self._out_shape = Size([len(self._tensor_base)])
        self._max_value = self._tensor_base.prod(dim=-1)
        weights = self._tensor_base.flip(dims=(-1,)).roll(1, dims=-1)
        weights[..., 0] = 1
        weights = weights.cumprod(dim=-1).flip(dims=(-1,)).movedim(-1, 0)
        self._weights = weights
        super().__init__()

    # override
    @property
    def in_shape(self) -> "Size":
        return self._in_shape

    # override
    @property
    def out_shape(self) -> "Size":
        return self._out_shape

    # override
    @property
    def inv(self) -> "b2d.Base2Dec":
        return b2d.Base2Dec(self.base)

    # override
    def _forward(self, input: "Tensor") -> "Tensor":
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

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(base: {self.base})"
