# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Fully connected neural network model."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Size, Tensor, device, nn

from actorch.models.hydra import Hydra
from actorch.models.modules import ParameterModule


__all__ = [
    "FCNet",
]


class FCNet(Hydra):
    """Fully connected neural network model."""

    def __init__(
        self,
        in_shapes: "Dict[str, Tuple[int, ...]]",
        out_shapes: "Dict[str, Tuple[int, ...]]",
        torso_fc_configs: "Optional[Sequence[Dict[str, Any]]]" = None,
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
        torso_fc_configs:
            The configurations of the torso fully connected layers.
            Argument `in_features` is set internally.
            Default to ``[
                {"out_features": 64, "bias": True},
                {"out_features": 64, "bias": True},
            ]``.
        torso_activation_builder:
            The torso activation builder (the same for all
            torso fully connected layers), i.e. a callable that
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
        self.torso_fc_configs = torso_fc_configs or [
            {"out_features": 64, "bias": True},
            {"out_features": 64, "bias": True},
        ]
        self.torso_activation_builder = torso_activation_builder
        self.torso_activation_config = torso_activation_config or {}
        self.head_fc_bias = head_fc_bias
        self.head_activation_builder = head_activation_builder
        self.head_activation_config = head_activation_config or {}
        self.independent_heads = independent_heads or []
        super().__init__(in_shapes, out_shapes)

    # override
    def _setup_torso(self, in_shape: "Size") -> "None":
        in_features = [in_shape.numel()] + [
            config["out_features"] for config in self.torso_fc_configs
        ]
        torso = []
        for i in range(len(self.torso_fc_configs)):
            torso += [
                nn.Linear(in_features[i], **self.torso_fc_configs[i]),
                self.torso_activation_builder(**self.torso_activation_config),
            ]
        self.torso = nn.Sequential(*torso)

    # override
    def _setup_heads(self, in_shape: "Size") -> "None":
        heads = {}
        for name, out_shape in self.out_shapes.items():
            if name in self.independent_heads:
                heads[name] = ParameterModule(in_shape, out_shape)
            else:
                head = [
                    nn.Linear(
                        in_shape.numel(),
                        out_shape.numel(),
                        self.head_fc_bias,
                    ),
                    self.head_activation_builder(**self.head_activation_config),
                ]
                heads[name] = nn.Sequential(*head)
        self.heads = nn.ModuleDict(heads)

    # override
    def _forward_torso(
        self,
        input: "Tensor",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Tensor, Dict[str, Tensor]]":
        input = input[mask].flatten(start_dim=1)
        masked_output = self.torso(input)
        out_shape = masked_output.shape[1:]
        output = torch.zeros(mask.shape + out_shape, device=masked_output.device)
        output[mask] = masked_output
        return output, states

    # override
    def _forward_heads(
        self,
        input: "Tensor",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor]]":
        outputs = {}
        for name, out_shape in self.out_shapes.items():
            masked_input = input[mask].flatten(start_dim=1)
            masked_output = self.heads[name](masked_input)
            masked_output = masked_output.reshape(-1, *out_shape)
            output = torch.zeros(mask.shape + out_shape, device=masked_output.device)
            output[mask] = masked_output
            outputs[name] = output
        return outputs, states

    # override
    def get_example_inputs(
        self,
        batch_shape: "Tuple[int, ...]" = (1,),
        device: "Optional[Union[device, str]]" = "cpu",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]":
        inputs = {
            name: torch.rand(batch_shape + in_shape, device=device)
            for name, in_shape in self.in_shapes.items()
        }
        states = {}
        mask = torch.ones(batch_shape, dtype=torch.bool, device=device)
        return inputs, states, mask
