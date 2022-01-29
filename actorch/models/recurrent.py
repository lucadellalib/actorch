# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Recurrent model."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor, device

from actorch.models.fc import FC
from actorch.models.hydra import Hydra
from actorch.registry import register


__all__ = [
    "Recurrent",
]


@register
class Recurrent(FC):
    """Recurrent model."""

    def __init__(
        self,
        in_shapes: "Dict[str, Tuple[int, ...]]",
        out_shapes: "Dict[str, Tuple[int, ...]]",
        torso_lstm_config: "Optional[Dict[str, Any]]" = None,
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
        torso_lstm_config:
            The torso LSTM configuration.
            Default to ``{
                "hidden_size": 64,
                "num_layers": 2,
                "bias": True,
                "batch_first": False,
                "dropout": 0.,
                "bidirectional": False,
                "proj_size": 0,
            }``.
        head_fc_bias:
            True to learn an additive bias in the head
            fully connected layer, False otherwise
            (the same for all heads).
        head_activation_builder:
            The head activation builder
            (the same for all heads).

        """
        self.torso_lstm_config = torso_lstm_config or {
            "hidden_size": 64,
            "num_layers": 2,
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": False,
            "proj_size": 0,
        }
        self.head_fc_bias = head_fc_bias
        self.head_activation_builder = head_activation_builder
        Hydra.__init__(self, in_shapes, out_shapes)

    # override
    def _setup_torso(self, in_shape: "Tuple[int, ...]") -> "None":
        self.torso = nn.LSTM(torch.Size(in_shape).numel(), **self.torso_lstm_config)

    # override
    def _forward_torso(
        self,
        input: "Tensor",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Tensor, Dict[str, Tensor]]":
        batch_shape = mask.shape
        batch_first = self.torso_lstm_config["batch_first"]
        input = input.flatten(start_dim=mask.ndim)
        if mask.ndim < 2:
            mask = mask[..., None] if batch_first else mask[None]
        mask = (
            mask.reshape(-1, mask.shape[-1])
            if batch_first
            else mask.reshape(mask.shape[0], -1)
        )
        input = input.reshape(*mask.shape, -1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            input,
            mask.sum(dim=-1 if batch_first else 0).to("cpu"),
            batch_first,
            enforce_sorted=False,
        )
        # Copy to avoid side effects
        states = {k: x for k, x in states.items()}
        hidden_state = states.get("hidden_state")
        cell_state = states.get("cell_state")
        lstm_state = None
        if hidden_state is not None and cell_state is not None:
            lstm_state = tuple(
                x.reshape(x.shape[0], -1, x.shape[-1])
                for x in [hidden_state, cell_state]
            )
        packed_output, lstm_state = self.torso(
            packed_input,
            lstm_state,
        )
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first)
        output = output.reshape(*batch_shape, -1)
        states["hidden_state"], states["cell_state"] = tuple(
            x.reshape(
                -1, *(batch_shape[:-1] if batch_first else batch_shape[1:]), x.shape[-1]
            )
            for x in lstm_state
        )
        return output, states

    # override
    def get_example_inputs(
        self,
        batch_shape: "Tuple[int, ...]" = (1,),
        device: "Optional[Union[device, str]]" = "cpu",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]":
        example_inputs = super().get_example_inputs(batch_shape, device)
        hidden_size = self.torso_lstm_config["hidden_size"]
        num_layers = self.torso_lstm_config["num_layers"]
        num_directions = 2 if self.torso_lstm_config["bidirectional"] else 1
        proj_size = self.torso_lstm_config["proj_size"]
        example_inputs[1]["hidden_state"] = torch.rand(
            (
                num_directions * num_layers,
                *batch_shape,
                proj_size if proj_size > 0 else hidden_size,
            ),
            device=device,
        )
        example_inputs[1]["cell_state"] = torch.rand(
            (num_directions * num_layers, *batch_shape, hidden_size),
            device=device,
        )
        return example_inputs
