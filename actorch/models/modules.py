# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Neural network modules."""

from typing import Any, Tuple

import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


__all__ = [
    "ParameterModule",
]


class ParameterModule(nn.Module):
    """Wrap a `torch.nn.Parameter` so that it can be included in a
    `torch.nn.Sequential`, `torch.nn.ModuleList` or `torch.nn.ModuleDict`.

    """

    def __init__(self, in_shape: "Tuple[int, ...]", parameter: "Parameter") -> "None":
        """Initialize the object.

        Parameters
        ----------
        in_shape:
            The input event shape.
        parameter:
            The parameter to wrap.

        """
        super().__init__()
        self.in_shape = in_shape
        self.parameter = parameter

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
            The input.

        Returns
        -------
            The underlying parameter, expanded
            to match the input batch shape.

        """
        batch_shape = input.shape[: input.ndim - len(self.in_shape)]
        expanded_parameter = self.parameter.expand(
            batch_shape + (-1,) * self.parameter.ndim
        )
        return expanded_parameter
