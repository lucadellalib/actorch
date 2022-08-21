# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""One-hot decode processor."""

from torch import Size, Tensor

from actorch.networks.processors import one_hot_encode as ohe  # Avoid circular import
from actorch.networks.processors.processor import Processor


__all__ = [
    "OneHotDecode",
]


class OneHotDecode(Processor):
    """One-hot decode a tensor."""

    # override
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
        self._in_shape = Size([self.num_classes])
        self._out_shape = Size([])
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
    def inv(self) -> "ohe.OneHotEncode":
        return ohe.OneHotEncode(self.num_classes)

    # override
    def _forward(self, input: "Tensor") -> "Tensor":
        return input.argmax(dim=-1)

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(num_classes: {self.num_classes})"
