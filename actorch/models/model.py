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

"""Model."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, device, nn


__all__ = [
    "Model",
]


class Model(ABC, nn.Module):
    """Generic multimodal input multimodal output model.

      input 1   ...  input n
         |              |
    .____|______________|____.
    |         model          |
    |________________________|
         |              |
         |              |
      output 1  ...  output m

    """

    # override
    def __init__(
        self,
        in_shapes: "Dict[str, Tuple[int, ...]]",
        out_shapes: "Dict[str, Tuple[int, ...]]",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        in_shapes:
            The input event shapes, i.e. a dict that maps names
            of the inputs to their corresponding event shapes.
        out_shapes:
            The output event shapes, i.e. a dict that maps names
            of the outputs to their corresponding event shapes.

        """
        super().__init__()
        self.in_shapes = {k: torch.Size(v) for k, v in in_shapes.items()}
        self.out_shapes = {k: torch.Size(v) for k, v in out_shapes.items()}
        self._setup()

    # override
    def forward(
        self,
        inputs: "Dict[str, Tensor]",
        states: "Optional[Dict[str, Tensor]]" = None,
        mask: "Optional[Tensor]" = None,
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor]]":
        """Forward pass.

        In the following, let `B = {B_1, ..., B_k}` denote the batch shape.

        Parameters
        ----------
        inputs:
            The inputs, i.e. a dict whose key-value pairs are consistent
            with the given `in_shapes` initialization argument, shape of
            ``inputs[name]``: ``[*B, *in_shapes[name]]``.
        states:
            The states, i.e. a dict with arbitrary key-value pairs.
            Useful, for example, to store hidden states of recurrent models.
            Default to ``{}``.
        mask:
            The boolean tensor indicating which batch elements are
            valid (True) and which are not (False), shape: ``[*B]``.
            Default to ``torch.ones(B, dtype=torch.bool)``.

        Returns
        -------
            - The outputs, i.e. a dict whose key-value pairs are consistent
              with the given `out_shapes` initialization argument, shape of
              ``outputs[name]``: ``[*B, *out_shapes[name]]``;
            - the possibly updated states.

        Warnings
        --------
        Batch dimensions of `states` must be moved to the front.

        """
        if states is None:
            states = {}
        if mask is None:
            first_key = list(inputs)[0]
            first_input = inputs[first_key]
            batch_shape = first_input.shape[
                : first_input.ndim - len(self.in_shapes[first_key])
            ]
            mask = torch.ones(batch_shape, dtype=torch.bool, device=first_input.device)
        outputs, states = self._forward(inputs, states, mask)
        return outputs, states

    def _setup(self) -> "None":
        """Setup the model."""
        pass

    @abstractmethod
    def _forward(
        self,
        inputs: "Dict[str, Tensor]",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor]]":
        """See documentation of `forward'."""
        raise NotImplementedError

    @abstractmethod
    def get_example_inputs(
        self,
        batch_shape: "Tuple[int, ...]" = (1,),
        device: "Optional[Union[device, str]]" = "cpu",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]":
        """Return a sequence of example inputs for `forward`.

        Useful, for example, to trace the model through `torch.jit.trace`.

        Parameters
        ----------
        batch_shape:
            The batch shape.
        device:
            The device.

        Returns
        -------
            The example inputs.

        """
        raise NotImplementedError
