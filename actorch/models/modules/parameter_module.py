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

"""Parameter module."""

from typing import Any, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter


__all__ = [
    "ParameterModule",
]


class ParameterModule(Module):
    """Wrap a `torch.nn.Parameter` in a `torch.nn.Module`."""

    parameter: "Parameter"
    """The underlying parameter."""

    # override
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

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(in_shape: {self.in_shape}, "
            f"out_shape: {self.out_shape})"
        )
