# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Convolutional neural network model."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Size, Tensor, nn

from actorch.models.fc_net import FCNet


__all__ = [
    "ConvNet",
]


class ConvNet(FCNet):
    """Convolutional neural network model."""

    def __init__(
        self,
        in_shapes: "Dict[str, Tuple[int, ...]]",
        out_shapes: "Dict[str, Tuple[int, ...]]",
        torso_conv_configs: "Optional[Sequence[Dict[str, Any]]]" = None,
        torso_activation_builder: "Callable[..., nn.Module]" = nn.ReLU,
        torso_activation_config: "Optional[Dict[str, Any]]" = None,
        head_fc_bias: "bool" = True,
        head_activation_builder: "Callable[..., nn.Module]" = nn.Identity,
        head_activation_config: "Optional[Dict[str, Any]]" = None,
        independent_heads: "Optional[Sequence[str]]" = None,
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
            Argument `in_channels` is set internally.
            Default to ``[
                {"out_channels": 8, "kernel_size": 4, "bias": True},
                {"out_channels": 4, "kernel_size": 2, "bias": True},
            ]``.
        torso_activation_builder:
            The torso activation builder (the same for all
            torso convolutional layers), i.e. a callable that
            receives keyword arguments from a configuration
            and returns an activation.
        torso_activation_config:
            The torso activation configuration
            (the same for all torso fully connected layers).
            Default to ``{}``.
        head_fc_bias:
            True to learn an additive bias in the head
            fully connected layer, False otherwise
            (the same for all input-dependent heads).
        head_activation_builder:
            The head activation builder (the same for all
            input-dependent heads), i.e. a callable that
            receives keyword arguments from a configuration
            and returns an activation.
        head_activation_config:
            The head activation configuration
            (the same for all input-dependent heads).
            Default to ``{}``.
        independent_heads:
            The names of the input-independent heads.
            These heads return a constant output tensor,
            optimized during training.
            Default to ``[]``.

        """
        self.torso_conv_configs = torso_conv_configs or [
            {"out_channels": 8, "kernel_size": 4, "bias": True},
            {"out_channels": 4, "kernel_size": 2, "bias": True},
        ]
        self.torso_activation_builder = torso_activation_builder
        self.torso_activation_config = torso_activation_config or {}
        self.head_fc_bias = head_fc_bias
        self.head_activation_builder = head_activation_builder
        self.head_activation_config = head_activation_config or {}
        self.independent_heads = independent_heads or []
        super(FCNet, self).__init__(in_shapes, out_shapes)

    # override
    def _setup_torso(self, in_shape: "Size") -> "None":
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
                self.torso_activation_builder(**self.torso_activation_config),
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
