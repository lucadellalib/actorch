# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Parameter module."""

from typing import Any, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter


__all__ = [
    "ParameterModule",
]


class ParameterModule(Module):
    """Wrap a `torch.nn.Parameter` in a ``torch.nn.Module``."""

    parameter: Parameter
    """The underlying parameter."""

    def __init__(
        self,
        in_shape: "Tuple[int, ...]",
        out_shape: "Tuple[int, ...]",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        in_shape:
            The input event shape.
        out_shape:
            The output event shape
            (i.e. the underlying parameter shape).

        """
        super().__init__()
        self.in_shape = torch.Size(in_shape)
        self.out_shape = torch.Size(out_shape)
        self.parameter = Parameter(torch.zeros(out_shape))

    # override
    def forward(
        self,
        input: "Tensor",
        *args: "Any",
        **kwargs: "Any",
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            The tensor.

        Returns
        -------
            The underlying parameter, expanded
            to match the batch shape of `input`.

        """
        batch_shape = input.shape[: input.ndim - len(self.in_shape)]
        expanded_parameter = self.parameter.expand(
            batch_shape + (-1,) * self.parameter.ndim
        )
        return expanded_parameter

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(in_shape: {self.in_shape}, "
            f"out_shape: {self.out_shape})"
        )
