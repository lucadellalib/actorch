# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Broadcast concatenation module."""

import math
from functools import reduce
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn


__all__ = [
    "BroadcastCat",
]


class BroadcastCat(nn.Module):
    """Concatenate a sequence of tensors along a dimension.

    Tensor shapes are expanded to the right, and broadcasting
    is applied along all dimensions except for the `dim`-th.

    """

    def __init__(
        self,
        in_shapes: "Sequence[Tuple[int, ...]]",
        dim: "int" = 0,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        in_shapes:
            The input event shapes.
        dim:
            The dimension.

        Raises
        ------
        IndexError
            If `dim` is out of range or not integer.

        """
        super().__init__()
        max_event_ndims = max([len(shape) for shape in in_shapes])
        if (
            dim < -max_event_ndims
            or dim >= max_event_ndims
            or not float(dim).is_integer()
        ):
            raise IndexError(
                f"`dim` ({dim}) must be in the integer interval [-{max_event_ndims}, {max_event_ndims})"
            )
        self.in_shapes = [torch.Size(s) for s in in_shapes]
        self.dim = (int(dim) + max_event_ndims) % max_event_ndims
        expanded_in_shapes = [
            shape + (1,) * (max_event_ndims - len(shape)) for shape in self.in_shapes
        ]
        # Broadcast along all dimensions except for the dim-th
        target_in_shape = [
            reduce(lambda x, y: x * y // math.gcd(x, y), sizes) if i != self.dim else -1
            for i, sizes in enumerate(zip(*expanded_in_shapes))
        ]
        self._repeat_shapes = [
            tuple(
                target_size // expanded_size if target_size != -1 else 1
                for target_size, expanded_size in zip(
                    target_in_shape, expanded_in_shape
                )
            )
            for expanded_in_shape in expanded_in_shapes
        ]

    # override
    def forward(
        self,
        inputs: "Sequence[Tensor]",
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        inputs:
            The tensors.

        Returns
        -------
            The broadcast concatenation of `inputs`.

        """
        expanded_inputs = []
        batch_ndims = None
        for i, input in enumerate(inputs):
            event_ndims = len(self.in_shapes[i])
            if batch_ndims is None:
                batch_ndims = input.ndim - event_ndims
            target_event_shape = self._repeat_shapes[i]
            expanded_input = input[
                (...,) + (None,) * (len(target_event_shape) - event_ndims)
            ].repeat((1,) * batch_ndims + target_event_shape)
            expanded_inputs.append(expanded_input)
        return torch.cat(expanded_inputs, dim=batch_ndims + self.dim)

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(in_shapes: {self.in_shapes}, "
            f"dim: {self.dim})"
        )
