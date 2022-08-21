#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Test value estimation."""

import pytest
import torch
import torch.nn.functional as F

from actorch.algorithms.value_estimation import n_step_return, vtrace
from actorch.distributions import Finite, MaskedDistribution


@pytest.mark.parametrize("num_return_steps", [1, 2, 4, 6, 10])
@pytest.mark.parametrize("bootstrap", [True, False])
def test_n_step_return(num_return_steps, bootstrap):
    torch.manual_seed(0)
    rewards = torch.rand(3, 6)
    terminals = torch.zeros(3, 6)
    terminals[0, 2] = 1
    mask = torch.zeros(3, 6)
    mask[0, 0] = 1
    mask[0, 1] = 1
    mask[0, 2] = 1
    mask[1, 0] = 1
    mask[2, :] = 1
    if bootstrap:
        mask = F.pad(mask, [1, 0], value=True)
    distributional_state_values = MaskedDistribution(
        Finite(torch.rand(3, 6 + bootstrap, 5), validate_args=False),
        mask,
        validate_args=False,
    )

    targets, advantages = n_step_return(
        distributional_state_values.mean,
        rewards,
        terminals,
        mask=mask[:, 1:] if bootstrap else mask,
        num_return_steps=num_return_steps,
    )

    distributional_targets, advantages_mean = n_step_return(
        distributional_state_values,
        rewards,
        terminals,
        mask=mask[:, 1:] if bootstrap else mask,
        num_return_steps=num_return_steps,
    )
    assert distributional_targets.mean.isclose(
        targets, atol=1e-6
    ).all(), f"distributional_targets.mean: {distributional_targets.mean}, \ntargets: {targets}"
    assert advantages_mean.isclose(
        advantages, atol=1e-6
    ).all(), f"advantages_mean: {advantages_mean}, \nadvantages: {advantages}"
    assert isinstance(
        distributional_targets, MaskedDistribution
    ), distributional_targets
    assert isinstance(
        distributional_targets.base_dist, Finite
    ), distributional_targets.base_dist


@pytest.mark.parametrize("num_return_steps", [1, 2, 4, 6, 10])
@pytest.mark.parametrize("bootstrap", [True, False])
def test_vtrace(num_return_steps, bootstrap):
    torch.manual_seed(0)
    rewards = torch.rand(3, 6)
    terminals = torch.zeros(3, 6)
    terminals[0, 2] = 1
    log_is_weights = torch.rand(3, 6)
    mask = torch.zeros(3, 6)
    mask[0, 0] = 1
    mask[0, 1] = 1
    mask[0, 2] = 1
    mask[1, 0] = 1
    mask[2, :] = 1
    if bootstrap:
        mask = F.pad(mask, [1, 0], value=True)
    distributional_state_values = MaskedDistribution(
        Finite(torch.rand(3, 6 + bootstrap, 5), validate_args=False),
        mask,
        validate_args=False,
    )

    targets, advantages = vtrace(
        distributional_state_values.mean,
        rewards,
        terminals,
        log_is_weights,
        mask=mask[:, 1:] if bootstrap else mask,
        num_return_steps=num_return_steps,
    )

    distributional_targets, advantages_mean = vtrace(
        distributional_state_values,
        rewards,
        terminals,
        log_is_weights,
        mask=mask[:, 1:] if bootstrap else mask,
        num_return_steps=num_return_steps,
    )
    assert distributional_targets.mean.isclose(
        targets, atol=1e-6
    ).all(), f"distributional_targets.mean: {distributional_targets.mean}, \ntargets: {targets}"
    assert advantages_mean.isclose(
        advantages, atol=1e-6
    ).all(), f"advantages_mean: {advantages_mean}, \nadvantages: {advantages}"
    assert isinstance(
        distributional_targets, MaskedDistribution
    ), distributional_targets
    assert isinstance(
        distributional_targets.base_dist, Finite
    ), distributional_targets.base_dist


if __name__ == "__main__":
    pytest.main([__file__])
