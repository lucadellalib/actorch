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

"""Long short-term memory neural network model."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Size, Tensor, device, nn

from actorch.models.fc_net import FCNet


__all__ = [
    "LSTMNet",
]


class LSTMNet(FCNet):
    """Long short-term memory neural network model."""

    # override
    def __init__(
        self,
        in_shapes: "Dict[str, Tuple[int, ...]]",
        out_shapes: "Dict[str, Tuple[int, ...]]",
        torso_lstm_config: "Optional[Dict[str, Any]]" = None,
        head_fc_bias: "bool" = True,
        head_activation_builder: "Optional[Callable[..., nn.Module]]" = None,
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
        torso_lstm_config:
            The torso LSTM configuration.
            Argument `input_size` is set internally.
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
            (the same for all input-dependent heads).
        head_activation_builder:
            The head activation builder (the same for all
            input-dependent heads), i.e. a callable that
            receives keyword arguments from a configuration
            and returns an activation.
            Default to ``torch.nn.Identity``.
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
        self.torso_lstm_config = torso_lstm_config or {
            "hidden_size": 64,
            "num_layers": 2,
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": False,
            "proj_size": 0,
        }
        super().__init__(
            in_shapes,
            out_shapes,
            head_fc_bias=head_fc_bias,
            head_activation_builder=head_activation_builder,
            head_activation_config=head_activation_config,
            independent_heads=independent_heads,
        )
        del self.torso_fc_configs
        del self.torso_activation_builder
        del self.torso_activation_config

    # override
    def _setup_torso(self, in_shape: "Size") -> "None":
        self.torso = nn.LSTM(in_shape.numel(), **self.torso_lstm_config)

    # override
    def _forward_torso(
        self,
        input: "Tensor",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Tensor, Dict[str, Tensor]]":
        batch_shape = mask.shape
        batch_first = self.torso_lstm_config["batch_first"]
        if mask.ndim < 2:
            # Add batch and/or time dimension
            mask = mask[(None,) * (2 - mask.ndim)]
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
                x.movedim(-2, 0).reshape(x.shape[-2], -1, x.shape[-1])
                for x in [hidden_state, cell_state]
            )
        packed_output, lstm_state = self.torso(
            packed_input,
            lstm_state,
        )
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first)
        output = output.reshape(*batch_shape, -1)
        states["hidden_state"], states["cell_state"] = tuple(
            x.movedim(0, -2).reshape(
                mask.shape[0 if batch_first else 1],
                x.shape[0],
                x.shape[-1],
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
        batch_first = self.torso_lstm_config["batch_first"]
        num_directions = 2 if self.torso_lstm_config["bidirectional"] else 1
        proj_size = self.torso_lstm_config["proj_size"]
        missing_batch_ndims = 2 - len(batch_shape)
        if missing_batch_ndims > 0:
            batch_shape = (
                batch_shape + (1,) * missing_batch_ndims
                if batch_first
                else (1,) * missing_batch_ndims + batch_shape
            )
        state_batch_shape = batch_shape[:-1] if batch_first else batch_shape[1:]
        example_inputs[1]["hidden_state"] = torch.rand(
            (
                *state_batch_shape,
                num_directions * num_layers,
                proj_size if proj_size > 0 else hidden_size,
            ),
            device=device,
        )
        example_inputs[1]["cell_state"] = torch.rand(
            (*state_batch_shape, num_directions * num_layers, hidden_size),
            device=device,
        )
        return example_inputs
