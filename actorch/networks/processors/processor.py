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
    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}()"

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
    def __call__(self, input: "Tensor") -> "Tensor":
        """Transform a tensor.

        Parameters
        ----------
        input:
            The tensor.

        Returns
        -------
            The transformed tensor.

        """
        raise NotImplementedError
