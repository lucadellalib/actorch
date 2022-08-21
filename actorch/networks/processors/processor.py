# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Processor."""

from abc import abstractmethod

from torch import Size, Tensor


__all__ = [
    "Processor",
]


class Processor:
    """Injective tensor-to-tensor transform."""

    def __init__(self) -> "None":
        """Initialize the object."""
        self._in_shape_str = " ".join(str(x) for x in self.in_shape)

    def __call__(self, input: "Tensor") -> "Tensor":
        """Transform a tensor.

        Parameters
        ----------
        input:
            The tensor.

        Returns
        -------
            The transformed tensor.

        Raises
        ------
        RuntimeError
            If event shape of `input` is not equal
            to the processor input event shape.

        """
        input_shape_str = " ".join(str(x) for x in input.shape)
        if not input_shape_str.endswith(self._in_shape_str):
            event_shape = input.shape[max(input.ndim - len(self.in_shape), 0) :]
            raise RuntimeError(
                f"Event shape of `input` ({event_shape}) must be equal "
                f"to the processor input event shape ({self.in_shape})"
            )
        return self._forward(input)

    def __repr__(self) -> "str":
        return f"{type(self).__name__}()"

    @property
    def inv(self) -> "Processor":
        """Return the inverse processor.

        Returns
        -------
            The inverse processor.

        Raises
        ------
        NotImplementedError
            If no inverse processor exists.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def in_shape(self) -> "Size":
        """Return the input event shape.

        Returns
        -------
            The input event shape.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def out_shape(self) -> "Size":
        """Return the output event shape.

        Returns
        -------
            The output event shape.

        """
        raise NotImplementedError

    @abstractmethod
    def _forward(self, input: "Tensor") -> "Tensor":
        """See documentation of `__call__`."""
        raise NotImplementedError
