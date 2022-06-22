# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""One-hot decode processor."""

import torch
from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "OneHotDecode",
]


class OneHotDecode(Processor):
    """One-hot decode a tensor."""

    def __init__(self, num_classes: "int") -> "None":
        """Initialize the object.

        Parameters
        ----------
        num_classes:
            The number of classes.

        Raises
        ------
        ValueError
            If `num_classes` is not in the integer interval [1, inf).

        """
        if num_classes < 1 or not float(num_classes).is_integer():
            raise ValueError(
                f"`num_classes` ({num_classes}) must be in the integer interval [1, inf)"
            )
        self.num_classes = int(num_classes)
        self._in_shape = torch.Size([self.num_classes])
        self._out_shape = torch.Size([1])

    @property
    def in_shape(self) -> "Size":
        return self._in_shape

    @property
    def out_shape(self) -> "Size":
        return self._out_shape

    def __call__(self, input: "Tensor") -> "Tensor":
        return input.argmax(dim=-1, keepdim=True)

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(num_classes: {self.num_classes})"
        )
