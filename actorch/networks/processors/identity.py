# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Identity processor."""

from typing import Tuple

import torch
from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "Identity",
]


class Identity(Processor):
    """Return a tensor unaltered."""

    def __init__(self, shape: "Tuple[int, ...]") -> "None":
        """Initialize the object.

        Parameters
        ----------
        shape:
            The input/output event shape.

        """
        self.shape = torch.Size(shape)

    @property
    def in_shape(self) -> "Size":
        return self.shape

    @property
    def out_shape(self) -> "Size":
        return self.shape

    def __call__(self, input: "Tensor") -> "Tensor":
        return input

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}" f"(shape: {self.shape})"
