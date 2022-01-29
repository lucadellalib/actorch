# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Generalized estimator."""

import warnings
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import AffineTransform, Distribution

from actorch.distributions import CatDistribution, SumTransform, TransformedDistribution


__all__ = [
    "generalized_estimator",
]


def generalized_estimator(
    state_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    dones: "Tensor",
    mask: "Tensor",
    trace_weights: "Tensor",
    delta_weights: "Tensor",
    advantage_weights: "Tensor",
    action_values: "Optional[Union[Tensor, Distribution]]" = None,
    next_state_values: "Optional[Union[Tensor, Distribution]]" = None,
    discount: "float" = 0.99,
    num_return_steps: "int" = 1,
    trace_decay: "float" = 1.0,
    standardize_advantage: "bool" = False,
    epsilon: "float" = 1e-6,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (distributional) generalized estimator targets and the corresponding
    advantages of a trajectory.

    In the following, let `B` denote the batch size and `T` the trajectory maximum length.

    Parameters
    ----------
    state_values:
        The (distributional) state values (`v_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
    rewards:
        The rewards (`r_t` in the literature), shape: ``[B, T]``.
    dones:
        The end-of-episode flags, shape: ``[B, T]``.
    mask:
        The boolean tensor indicating which elements (or batch elements
        if distributional) are valid (True) and which are not (False),
        shape: ``[B, T]``.
    trace_weights:
        The trace weights (`c_t` or `traces` in the literature), shape: ``[B, T]``.
    delta_weights:
        The delta weights (`rho_t` in the literature), shape: ``[B, T]``.
    advantage_weights:
        The advantage weights, shape: ``[B, T]``.
    action_values:
        The (distributional) action values (`q_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
    next_state_values:
        The (distributional) next state values (`v_t+1` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
        Default to `state_values` from the second timestep on, with a bootstrap value set to 0.
    discount:
        The discount factor (`gamma` in the literature).
    num_return_steps:
        The number of return steps (`n` in the literature).
    trace_decay:
        The trace-decay parameter (`lambda` in the literature).
    standardize_advantage:
        True to standardize the advantages, False otherwise.
    epsilon:
        The term added to the denominators to improve numerical stability.

    Returns
    -------
        - The (distributional) generalized estimator targets,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, ``[B, T]``.

    Raises
    ------
    ValueError
        If an invalid argument value is provided.

    """
    if discount < 0.0 or discount > 1.0:
        raise ValueError(f"`discount` ({discount}) must be in the interval [0, 1]")
    if num_return_steps < 1 or not float(num_return_steps).is_integer():
        raise ValueError(
            f"`num_return_steps` ({num_return_steps}) must be in the integer interval [1, inf)"
        )
    if trace_decay < 0.0 or trace_decay > 1.0:
        raise ValueError(
            f"`trace_decay` ({trace_decay}) must be in the interval [0, 1]"
        )
    if epsilon < 0.0:
        raise ValueError(f"`epsilon` ({epsilon}) must be in the interval [0, inf)")
    is_distributional = isinstance(state_values, Distribution)
    targets = (
        _compute_distributional_targets if is_distributional else _compute_targets
    )(
        state_values,
        rewards,
        dones,
        mask,
        trace_weights,
        delta_weights,
        action_values,
        next_state_values,
        discount,
        num_return_steps,
    )
    advantages = _compute_advantages(
        targets.mean if is_distributional else targets,
        state_values.mean if is_distributional else state_values,
        rewards,
        mask,
        advantage_weights,
        action_values.mean
        if (is_distributional and action_values is not None)
        else action_values,
        next_state_values.mean
        if (is_distributional and next_state_values is not None)
        else next_state_values,
        discount,
        trace_decay,
        standardize_advantage,
        epsilon,
    )
    return targets, advantages


def _compute_targets(
    state_values: "Tensor",
    rewards: "Tensor",
    dones: "Tensor",
    mask: "Tensor",
    trace_weights: "Tensor",
    delta_weights: "Tensor",
    action_values: "Optional[Tensor]" = None,
    next_state_values: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    num_return_steps: "int" = 1,
) -> "Tensor":
    # x[~mask] = 0.0 is equivalent to x *= mask
    # x[~mask] = 1.0 is equivalent to x += ~mask
    state_values, rewards, dones = state_values.clone(), rewards.clone(), dones.clone()
    trace_weights, delta_weights = trace_weights.clone(), delta_weights.clone()
    dones, mask = dones.bool(), mask.bool()
    state_values *= mask
    rewards *= mask
    if action_values is not None:
        action_values = action_values.clone()
        action_values *= mask
    dones += ~mask
    trace_weights += ~mask
    delta_weights += ~mask
    B, T = state_values.shape
    num_return_steps = min(num_return_steps, T)
    length = mask.sum(dim=1)
    if next_state_values is None:
        next_state_values = F.pad(state_values[:, 1:], [0, 1])
        next_state_values[torch.arange(B), length - 1] = 0.0
    else:
        next_state_values = next_state_values.clone()
    next_state_values *= mask
    next_state_values *= ~dones
    state_or_action_values = (
        action_values if action_values is not None else state_values
    )
    deltas = delta_weights * (
        rewards + discount * next_state_values - state_or_action_values
    )
    deltas *= mask
    window = discount * trace_weights
    window += ~mask
    # If the window is constant, use convolution, which is more memory and time-efficient
    is_constant_window = (window == window[:, :1])[mask].all()
    deltas = F.pad(deltas, [0, num_return_steps - 1])
    window = F.pad(window, [0, num_return_steps - 1], value=1.0)
    if not is_constant_window:
        deltas = F.unfold(deltas[:, None, :, None], (T, 1))
        window = F.unfold(window[:, None, :, None], (T, 1))
    window = window.roll(1, dims=-1)
    window[..., 0] = 1
    window = window.cumprod(dim=-1)
    errors = (
        F.conv1d(deltas[None], window[:, None, :num_return_steps], groups=B)[0]
        if is_constant_window
        else (deltas * window).sum(dim=-1)
    )
    targets = state_or_action_values + errors
    targets *= mask
    return targets


def _compute_distributional_targets(
    state_values: "Distribution",
    rewards: "Tensor",
    dones: "Tensor",
    mask: "Tensor",
    trace_weights: "Tensor",
    delta_weights: "Tensor",
    action_values: "Optional[Distribution]" = None,
    next_state_values: "Optional[Distribution]" = None,
    discount: "float" = 0.99,
    num_return_steps: "int" = 1,
) -> "Distribution":
    warnings.warn(
        "Distributional value estimation is experimental and may not work as expected",
    )
    # x[~mask] = 0.0 is equivalent to x *= mask
    # x[~mask] = 1.0 is equivalent to x += ~mask
    rewards, dones = rewards.clone(), dones.clone()
    trace_weights, delta_weights = trace_weights.clone(), delta_weights.clone()
    dones, mask = dones.bool(), mask.bool()
    rewards *= mask
    dones += ~mask
    trace_weights += ~mask
    delta_weights += ~mask
    B, T = state_values.batch_shape
    num_return_steps = min(num_return_steps, T)
    state_or_action_values = (
        action_values if action_values is not None else state_values
    )
    coeffs = torch.stack(
        [
            delta_weights * rewards,  # Offsets
            -delta_weights,  # `state_or_action_values` coefficients
            discount * delta_weights,  # `next_state_values` coefficients
        ]
    )
    window = discount * trace_weights
    window += ~mask
    coeffs = F.pad(coeffs, [0, num_return_steps - 1])
    window = F.pad(window, [0, num_return_steps - 1], value=1.0)
    # Pad mask using edge values
    mask = F.pad(mask, [0, num_return_steps - 1])
    mask[:, T - 1 :] = mask[:, T - 1 : T]
    coeffs = F.unfold(coeffs[..., None], (T, 1)).reshape(3, B, T, -1)
    window = F.unfold(window[:, None, :, None], (T, 1))
    mask = F.unfold(mask[:, None, :, None].float(), (T, 1)).bool()
    window = window.roll(1, dims=-1)
    window[..., 0] = 1
    window = window.cumprod(dim=-1)
    coeffs *= window
    coeffs[1, :, :, 0] += 1
    coeffs[2, ...] *= ~dones[..., None]
    if next_state_values is None:
        # To fix
        next_state_values = state_values
        coeffs[2, ...] = coeffs[2, ...].roll(1, dims=-1)
        coeffs[2, ..., 0] = 0.0
    coeffs *= mask
    with _patch_expand():
        state_or_action_values = state_or_action_values.expand((B, T, num_return_steps))
        next_state_values = next_state_values.expand((B, T, num_return_steps))
    state_or_action_values = TransformedDistribution(
        state_or_action_values,
        [
            AffineTransform(coeffs[0], coeffs[1]),
            SumTransform((num_return_steps,)),
        ],
        validate_args=False,
    ).reduced_dist
    next_state_values = TransformedDistribution(
        next_state_values,
        [
            AffineTransform(torch.zeros_like(coeffs[2]), coeffs[2]),
            SumTransform((num_return_steps,)),
        ],
        validate_args=False,
    ).reduced_dist
    targets = TransformedDistribution(
        CatDistribution(
            [state_or_action_values, next_state_values],
            dim=-1,
        ),
        SumTransform((2,)),
        validate_args=False,
    ).reduced_dist
    return targets


def _compute_advantages(
    targets: "Tensor",
    state_values: "Tensor",
    rewards: "Tensor",
    mask: "Tensor",
    advantage_weights: "Tensor",
    action_values: "Optional[Tensor]" = None,
    next_state_values: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    trace_decay: "float" = 1.0,
    standardize_advantage: "bool" = False,
    epsilon: "float" = 1e-6,
) -> "Tensor":
    # x[~mask] = 0.0 is equivalent to x *= mask
    # x[~mask] = 1.0 is equivalent to x += ~mask
    targets, state_values, rewards = (
        targets.clone(),
        state_values.clone(),
        rewards.clone(),
    )
    advantage_weights = advantage_weights.clone()
    mask = mask.bool()
    targets *= mask
    state_values *= mask
    rewards *= mask
    advantage_weights += ~mask
    B, T = state_values.shape
    length = mask.sum(dim=1)
    if action_values is not None:
        # Replace action values with current action values estimate (Retrace-like)
        action_values = targets
    else:
        # Compute action values from current state values estimate (V-trace-like)
        next_targets = F.pad(
            trace_decay * targets[:, 1:] + (1 - trace_decay) * state_values[:, 1:],
            [0, 1],
        )
        if next_state_values is None:
            next_state_values = F.pad(state_values[:, 1:], [0, 1])
            next_state_values[torch.arange(B), length - 1] = 0.0
        else:
            next_state_values = next_state_values.clone()
        next_state_values *= mask
        next_targets[torch.arange(B), length - 1] = next_state_values[
            torch.arange(B), length - 1
        ]
        action_values = rewards + discount * next_targets
    advantages = advantage_weights * (action_values - state_values)
    advantages *= mask
    if standardize_advantage:
        advantages_mean = (advantages.sum(dim=1) / length)[:, None]
        advantages -= advantages_mean
        advantages *= mask
        advantages_stddev = (
            ((advantages ** 2).sum(dim=1) / length).sqrt().clamp(min=epsilon)[:, None]
        )
        advantages /= advantages_stddev
        advantages *= mask
    return advantages


def _unfoldNd(input: "Tensor", shape: "Tuple[int, ...]") -> "Tensor":
    B, T, N = shape[0:3]
    if input.ndim == 2:
        result = F.pad(input, [0, N - 1])
        return F.unfold(result[:, None, :, None], (T, 1))
    result = input.reshape(B, 1, T, 1, -1)
    result = F.pad(result, [0, 0, 0, 0, 0, N - 1]).movedim(-1, 0)
    result = torch.stack([F.unfold(x, (T, 1)) for x in result], dim=-1)
    result = result.reshape(B, T, N, *input.shape[2:])
    return result


@contextmanager
def _patch_expand() -> "Iterator[None]":
    expand_backup = torch.Tensor.expand
    try:
        torch.Tensor.expand = _unfoldNd
        yield
    finally:
        torch.Tensor.expand = expand_backup
