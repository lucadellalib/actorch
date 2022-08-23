#!/usr/bin/env python3

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

"""Test models."""

import pytest
import torch
from torch import nn

from actorch.models import ConvNet, FCNet, LSTMNet


_TEST_CASES = {}

_TEST_CASES["FCNet"] = [
    {
        "in_shapes": {
            "obs": (2,),
        },
        "out_shapes": {
            "loc": (5,),
            "scale": (5,),
        },
        "independent_heads": ["scale"],
    },
    {
        "in_shapes": {
            "obs": (2, 3, 4),
            "action": (3, 6),
            "reward": (1,),
        },
        "out_shapes": {
            "loc": (3, 3),
            "scale": (3, 3),
        },
        "torso_fc_configs": [
            {"out_features": 64, "bias": True},
            {"out_features": 64, "bias": True},
            {"out_features": 32, "bias": False},
        ],
        "torso_activation_builder": nn.Identity,
    },
]

_TEST_CASES["ConvNet"] = [
    {
        "in_shapes": {
            "obs": (2, 10),
        },
        "out_shapes": {
            "loc": (5,),
            "scale": (5,),
        },
        "independent_heads": ["scale"],
    },
    {
        "in_shapes": {
            "obs": (2, 30, 40),
        },
        "out_shapes": {
            "loc": (5,),
            "scale": (5,),
        },
    },
    {
        "in_shapes": {
            "obs": (2, 30, 40, 50),
        },
        "out_shapes": {
            "loc": (5,),
            "scale": (5,),
        },
    },
    {
        "in_shapes": {
            "obs": (2, 30, 40),
            "action": (30, 30),
            "reward": (1,),
        },
        "out_shapes": {
            "loc": (30, 30),
            "scale": (30, 30),
        },
        "torso_conv_configs": [
            {"out_channels": 8, "kernel_size": 4, "bias": True},
            {"out_channels": 8, "kernel_size": 4, "bias": True},
            {"out_channels": 8, "kernel_size": 4, "bias": False},
        ],
        "torso_activation_builder": nn.LeakyReLU,
        "torso_activation_config": {"negative_slope": 1e-2},
    },
]

_TEST_CASES["LSTMNet"] = [
    {
        "in_shapes": {
            "obs": (2,),
        },
        "out_shapes": {
            "loc": (5,),
            "scale": (5,),
        },
        "independent_heads": ["scale"],
    },
    {
        "in_shapes": {
            "obs": (2,),
        },
        "out_shapes": {
            "loc": (5,),
            "scale": (5,),
        },
        "torso_lstm_config": {
            "hidden_size": 128,
            "num_layers": 3,
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": True,
            "proj_size": 16,
        },
    },
    {
        "in_shapes": {
            "obs": (2, 3, 4),
            "action": (3, 3),
            "reward": (1,),
        },
        "out_shapes": {
            "loc": (3, 3),
            "scale": (3, 3),
        },
        "torso_lstm_config": {
            "hidden_size": 64,
            "num_layers": 2,
            "bias": True,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": True,
            "proj_size": 0,
        },
    },
]


@pytest.mark.parametrize("model_config", _TEST_CASES["FCNet"])
@pytest.mark.parametrize("batch_shape", [(1,), (4,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
def test_fc_net(model_config, batch_shape):
    torch.manual_seed(0)
    return _test_model(FCNet, model_config, batch_shape)


@pytest.mark.parametrize("model_config", _TEST_CASES["ConvNet"])
@pytest.mark.parametrize("batch_shape", [(1,), (4,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
def test_conv_net(model_config, batch_shape):
    torch.manual_seed(0)
    return _test_model(ConvNet, model_config, batch_shape)


@pytest.mark.parametrize("model_config", _TEST_CASES["LSTMNet"])
@pytest.mark.parametrize("batch_shape", [(1,), (4,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
def test_lstm_net(model_config, batch_shape):
    torch.manual_seed(0)
    batch_first = (
        model_config["torso_lstm_config"].get("batch_first", False)
        if "torso_lstm_config" in model_config
        else False
    )
    if len(batch_shape) < 2:
        batch_shape = (batch_shape + (1,)) if batch_first else ((1,) + batch_shape)
    B, T = (
        (batch_shape[:-1], batch_shape[-1])
        if batch_first
        else (batch_shape[1:], batch_shape[0])
    )
    mask = torch.arange(T)
    seq_lengths = torch.randint(T, B) + 1
    mask = (
        mask.expand(*B, -1)
        if batch_first
        else mask[(...,) + (None,) * len(B)].expand(-1, *B)
    )
    seq_lengths = (
        seq_lengths[..., None].expand(*B, T)
        if batch_first
        else seq_lengths.expand(T, *B)
    )
    mask = mask < seq_lengths
    return _test_model(LSTMNet, model_config, batch_shape, mask=mask)


def _test_model(model_builder, model_config, batch_shape, states=None, mask=None):
    model = model_builder(**model_config)
    inputs = model.get_example_inputs(batch_shape)[0]
    outputs, states = model(inputs, states, mask)
    _test_shapes(model_config, batch_shape, outputs)
    _test_backward(outputs)
    _test_tracing(model)


def _test_shapes(model_config, batch_shape, outputs):
    for name, output in outputs.items():
        shape = model_config["out_shapes"][name]
        output_batch_shape = output.shape[: output.ndim - len(shape)]
        output_shape = output.shape[len(output_batch_shape) :]
        assert (
            batch_shape == output_batch_shape
        ), f"batch_shape: {batch_shape}, \noutput_batch_shape: {output_batch_shape}"
        assert shape == output_shape, f"shape: {shape}, \noutput_shape: {output_shape}"


def _test_backward(outputs):
    return {k: x.sum().backward(retain_graph=True) for k, x in outputs.items()}


def _test_tracing(model):
    example_inputs = model.get_example_inputs()
    if not example_inputs[1]:
        # If states is empty, add a dummy value to avoid
        # RuntimeError: Dictionary inputs must have entries
        example_inputs[1]["_"] = torch.empty(1)
    traced_model = torch.jit.trace(model, example_inputs, strict=False)
    outputs = model(*example_inputs)
    traced_model_outputs = traced_model(*example_inputs)
    assert all(
        (output == traced_model_output).all()
        for output, traced_model_output in zip(
            outputs[0].values(), traced_model_outputs[0].values()
        )
    ), f"outputs: {outputs}, \ntraced_model_outputs: {traced_model_outputs}"


if __name__ == "__main__":
    pytest.main([__file__])
