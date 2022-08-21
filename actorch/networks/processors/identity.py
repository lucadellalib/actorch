# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Identity processor."""

from typing import Tuple

from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "Identity",
]


class Identity(Processor):
    """Return a tensor unaltered."""

    # override
    def __init__(self, shape: "Tuple[int, ...]") -> "None":
        """Initialize the object.

        Parameters
        ----------
        shape:
            The input/output event shape.

        """
        self.shape = Size(shape)
        super().__init__()

    # override
    @property
    def in_shape(self) -> "Size":
        return self.shape

    # override
    @property
    def out_shape(self) -> "Size":
        return self.shape

    # override
    @property
    def inv(self) -> "Identity":
        return Identity(self.shape)

    # override
    def _forward(self, input: "Tensor") -> "Tensor":
        return input

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(shape: {self.shape})"
