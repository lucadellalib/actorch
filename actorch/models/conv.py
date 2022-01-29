# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Convolutional model."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from actorch.models.fc import FC
from actorch.models.hydra import Hydra
from actorch.registry import register


__all__ = [
    "Conv",
]


@register
class Conv(FC):
    """Convolutional model."""

    def __init__(
        self,
        in_shapes: "Dict[str, Tuple[int, ...]]",
        out_shapes: "Dict[str, Tuple[int, ...]]",
        torso_conv_configs: "Optional[List[Dict[str, Any]]]" = None,
        torso_activation_builder: "Callable[..., nn.Module]" = nn.ReLU,
        head_fc_bias: "bool" = True,
        head_activation_builder: "Callable[..., nn.Module]" = nn.Identity,
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
        torso_conv_configs:
            The configurations of the torso convolutional layers.
            Default to ``[
                {"out_channels": 8, "kernel_size": 4, "bias": True},
                {"out_channels": 4, "kernel_size": 2, "bias": True},
            ]``.
        torso_activation_builder:
            The torso activation builder
            (the same for all torso convolutional layers).
        head_fc_bias:
            True to learn an additive bias in the head
            fully connected layer, False otherwise
            (the same for all heads).
        head_activation_builder:
            The head activation builder
            (the same for all heads).

        """
        self.torso_conv_configs = torso_conv_configs or [
            {"out_channels": 8, "kernel_size": 4, "bias": True},
            {"out_channels": 4, "kernel_size": 2, "bias": True},
        ]
        self.torso_activation_builder = torso_activation_builder
        self.head_fc_bias = head_fc_bias
        self.head_activation_builder = head_activation_builder
        Hydra.__init__(self, in_shapes, out_shapes)

    # override
    def _setup_torso(self, in_shape: "Tuple[int, ...]") -> "None":
        n = len(in_shape)
        if n == 2:
            conv_builder = nn.Conv1d
        elif n == 3:
            conv_builder = nn.Conv2d
        elif n == 4:
            conv_builder = nn.Conv3d
        else:
            raise NotImplementedError(
                f"Convolution operator is not implemented for {n - 1}D input"
            )
        in_channels = [in_shape[0]] + [
            config["out_channels"] for config in self.torso_conv_configs
        ]
        torso = []
        for i in range(len(self.torso_conv_configs)):
            torso += [
                conv_builder(in_channels[i], **self.torso_conv_configs[i]),
                self.torso_activation_builder(),
            ]
        self.torso = nn.Sequential(*torso)

    # override
    def _forward_torso(
        self,
        input: "Tensor",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Tensor, Dict[str, Tensor]]":
        input = input[mask]
        masked_output = self.torso(input)
        out_shape = masked_output.shape[1:]
        output = torch.zeros(mask.shape + out_shape, device=masked_output.device)
        output[mask] = masked_output
        return output, states
