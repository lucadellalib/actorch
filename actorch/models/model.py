# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Model."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor, device


__all__ = [
    "Model",
]


class Model(ABC, nn.Module):
    """Generic many-to-many model that maps multiple
    inputs to multiple outputs.

      Input 1  ....  Input n
         |              |
     ____|______________|____
    |         Torso          |
    |________________________|
         |              |
         |              |
      Output 1 .... Output m

    """

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
        self.in_shapes = in_shapes
        self.out_shapes = out_shapes
        self._setup()

    # override
    def forward(
        self,
        inputs: "Dict[str, Tensor]",
        states: "Optional[Dict[str, Tensor]]" = None,
        mask: "Optional[Tensor]" = None,
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor]]":
        """Forward pass.

        In the following, let `B = [B_1, ..., B_k]` denote the batch shape.

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

        """
        first_key = list(inputs.keys())[0]
        first_input = inputs[first_key]
        batch_shape = first_input.shape[
            : first_input.ndim - len(self.in_shapes[first_key])
        ]
        if states is None:
            states = {}
        if mask is None:
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
        """See documentation of method `forward'."""
        raise NotImplementedError

    @abstractmethod
    def get_example_inputs(
        self,
        batch_shape: "Tuple[int, ...]" = (1,),
        device: "Optional[Union[device, str]]" = "cpu",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]":
        """Return a sequence of example inputs for method `forward`.

        Useful, for example, to trace the model via `torch.jit.trace`.

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
