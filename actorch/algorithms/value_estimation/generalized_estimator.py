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

"""Generalized estimator."""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import AffineTransform, Distribution

from actorch.algorithms.value_estimation.utils import distributional_gather
from actorch.distributions import CatDistribution, SumTransform, TransformedDistribution


__all__ = [
    "generalized_estimator",
]


def generalized_estimator(
    state_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    terminals: "Tensor",
    trace_weights: "Tensor",
    delta_weights: "Tensor",
    advantage_weights: "Tensor",
    action_values: "Optional[Union[Tensor, Distribution]]" = None,
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    num_return_steps: "int" = 1,
    trace_decay: "float" = 1.0,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (possibly distributional) generalized estimator targets
    and the corresponding advantages of a trajectory.

    In the following, let `B` denote the batch size and `T` the maximum
    trajectory length.

    Parameters
    ----------
    state_values:
        The (possibly distributional) state values (`v_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape):
        ``[B, T + 1]`` if a bootstrap value is given, ``[B, T]`` otherwise.
    rewards:
        The rewards (`r_t` in the literature), shape: ``[B, T]``.
    terminals:
        The terminal flags, shape: ``[B, T]``.
    trace_weights:
        The trace weights (`c_t` or `traces` in the literature), shape: ``[B, T]``.
    delta_weights:
        The delta weights (`rho_t` in the literature), shape: ``[B, T]``.
    advantage_weights:
        The advantage weights, shape: ``[B, T]``.
    action_values:
        The (possibly distributional) action values (`q_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
    mask:
        The boolean tensor indicating which elements (or batch elements
        if distributional) are valid (True) and which are not (False),
        shape: ``[B, T]``.
        Default to ``torch.ones_like(rewards, dtype=torch.bool)``.
    discount:
        The discount factor (`gamma` in the literature).
    num_return_steps:
        The number of return steps (`n` in the literature).
    trace_decay:
        The trace-decay parameter (`lambda` in the literature).

    Returns
    -------
        - The (possibly distributional) generalized estimator targets,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, ``[B, T]``.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    """
    if rewards.ndim != 2:
        raise ValueError(
            f"Number of dimensions of `rewards` ({rewards.ndim}) must be equal to 2"
        )
    B, T = rewards.shape
    if B < 1:
        raise ValueError(f"Batch size ({B}) must be in the integer interval [1, inf)")
    if T < 1:
        raise ValueError(
            f"Maximum trajectory length ({T}) must be in the integer interval [1, inf)"
        )
    if terminals.shape != rewards.shape:
        raise ValueError(
            f"Shape of `terminals` ({terminals.shape}) must be "
            f"equal to the shape of `rewards` ({rewards.shape})"
        )
    if trace_weights.shape != rewards.shape:
        raise ValueError(
            f"Shape of `trace_weights` ({trace_weights.shape}) must "
            f"be equal to the shape of `rewards` ({rewards.shape})"
        )
    if delta_weights.shape != rewards.shape:
        raise ValueError(
            f"Shape of `delta_weights` ({delta_weights.shape}) must "
            f"be equal to the shape of `rewards` ({rewards.shape})"
        )
    if advantage_weights.shape != rewards.shape:
        raise ValueError(
            f"Shape of `advantage_weights` ({advantage_weights.shape}) "
            f"must be equal to the shape of `rewards` ({rewards.shape})"
        )
    if mask is None:
        mask = torch.ones_like(rewards, dtype=torch.bool)
    elif mask.shape != rewards.shape:
        raise ValueError(
            f"Shape of `mask` ({mask.shape}) must be equal "
            f"to the shape of `rewards` ({rewards.shape})"
        )
    if discount <= 0.0 or discount > 1.0:
        raise ValueError(f"`discount` ({discount}) must be in the interval (0, 1]")
    if num_return_steps < 1 or not float(num_return_steps).is_integer():
        raise ValueError(
            f"`num_return_steps` ({num_return_steps}) "
            f"must be in the integer interval [1, inf)"
        )
    num_return_steps = int(num_return_steps)
    if trace_decay < 0.0 or trace_decay > 1.0:
        raise ValueError(
            f"`trace_decay` ({trace_decay}) must be in the interval [0, 1]"
        )
    compute_targets_args = [
        state_values,
        rewards,
        terminals,
        trace_weights,
        delta_weights,
        action_values,
        mask,
        discount,
        num_return_steps,
    ]
    if isinstance(state_values, Distribution):
        if state_values.batch_shape != (B, T) and state_values.batch_shape != (
            B,
            T + 1,
        ):
            raise ValueError(
                f"Batch shape of `state_values` ({state_values.batch_shape}) "
                f"must be equal to the shape of `rewards` ({torch.Size([B, T])}) "
                f"or to the shape of `rewards` with a size along the time "
                f"dimension increased by 1 ({torch.Size([B, T + 1])})"
            )
        if action_values is not None and action_values.batch_shape != rewards.shape:
            raise ValueError(
                f"Batch shape of `action_values` ({action_values.batch_shape}) "
                f"must be equal to the shape of `rewards` ({rewards.shape})"
            )
        targets, state_values, next_state_values = _compute_distributional_targets(
            *compute_targets_args
        )
        advantages = _compute_advantages(
            targets.mean,
            state_values.mean,
            rewards,
            advantage_weights,
            action_values.mean if action_values is not None else None,
            next_state_values.mean,
            mask,
            discount,
            trace_decay,
        )
        return targets, advantages
    if state_values.shape != (B, T) and state_values.shape != (B, T + 1):
        raise ValueError(
            f"Shape of `state_values` ({state_values.shape}) must be "
            f"equal to the shape of `rewards` ({torch.Size([B, T])}) "
            f"or to the shape of `rewards` with a size along the time "
            f"dimension increased by 1 ({torch.Size([B, T + 1])})"
        )
    if action_values is not None and action_values.shape != rewards.shape:
        raise ValueError(
            f"Shape of `action_values` ({action_values.shape}) must "
            f"be equal to the shape of `rewards` ({rewards.shape})"
        )
    targets, state_values, next_state_values = _compute_targets(*compute_targets_args)
    advantages = _compute_advantages(
        targets,
        state_values,
        rewards,
        advantage_weights,
        action_values,
        next_state_values,
        mask,
        discount,
        trace_decay,
    )
    return targets, advantages


def _compute_targets(
    state_values: "Tensor",
    rewards: "Tensor",
    terminals: "Tensor",
    trace_weights: "Tensor",
    delta_weights: "Tensor",
    action_values: "Optional[Tensor]",
    mask: "Tensor",
    discount: "float",
    num_return_steps: "int",
) -> "Tuple[Tensor, Tensor, Tensor]":
    # x[~mask] = 0.0 is equivalent to x *= mask
    # x[~mask] = 1.0 is equivalent to x *= mask, x += ~mask
    state_values, rewards, terminals = (
        state_values.clone(),
        rewards.clone(),
        terminals.clone(),
    )
    trace_weights, delta_weights = trace_weights.clone(), delta_weights.clone()
    terminals, mask = terminals.bool(), mask.bool()
    rewards *= mask
    if action_values is not None:
        action_values = action_values.clone()
        action_values *= mask
    terminals += ~mask
    trace_weights *= mask
    trace_weights += ~mask
    delta_weights *= mask
    delta_weights += ~mask
    B, T = rewards.shape
    num_return_steps = min(num_return_steps, T)
    if state_values.shape == (B, T + 1):
        next_state_values = state_values[:, 1:].clone()
        state_values = state_values[:, :-1]
        state_values *= mask
    else:
        state_values *= mask
        next_state_values = F.pad(state_values, [0, 1])[:, 1:]
    next_state_values *= mask
    next_state_values *= ~terminals
    state_or_action_values = (
        action_values if action_values is not None else state_values
    )
    deltas = delta_weights * (
        rewards + discount * next_state_values - state_or_action_values
    )
    deltas *= mask
    window = discount * trace_weights
    window *= mask
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
    return targets, state_values, next_state_values


def _compute_distributional_targets(
    state_values: "Distribution",
    rewards: "Tensor",
    terminals: "Tensor",
    trace_weights: "Tensor",
    delta_weights: "Tensor",
    action_values: "Optional[Distribution]",
    mask: "Tensor",
    discount: "float",
    num_return_steps: "int",
) -> "Tuple[Distribution, Distribution, Distribution]":
    # x[~mask] = 0.0 is equivalent to x *= mask
    # x[~mask] = 1.0 is equivalent to x *= mask, x += ~mask
    rewards, terminals = rewards.clone(), terminals.clone()
    trace_weights, delta_weights = trace_weights.clone(), delta_weights.clone()
    terminals, mask = terminals.bool(), mask.bool()
    rewards *= mask
    terminals += ~mask
    trace_weights *= mask
    trace_weights += ~mask
    delta_weights *= mask
    delta_weights += ~mask
    B, T = rewards.shape
    num_return_steps = min(num_return_steps, T)
    length = mask.sum(dim=1, keepdim=True)
    idx = torch.arange(1, T + 1).expand(B, T)
    if state_values.batch_shape == (B, T + 1):
        next_state_values = distributional_gather(
            state_values,
            1,
            idx.clamp(max=length),
            mask * (~terminals),
        )
        idx = torch.arange(0, T).expand(B, T)
        state_values = distributional_gather(
            state_values,
            1,
            idx.clamp(max=(length - 1).clamp(min=0)),
            mask,
        )
    else:
        next_state_values = distributional_gather(
            state_values,
            1,
            idx.clamp(max=(length - 1).clamp(min=0)),
            F.pad(mask, [0, 1])[:, 1:] * (~terminals),
        )
    state_or_action_values = (
        action_values if action_values is not None else state_values
    )
    window = discount * trace_weights
    window *= mask
    window += ~mask
    if ((trace_weights == 1.0) & (trace_weights == delta_weights))[mask].all():
        # N-step return-like computation:
        # targets depend only on rewards and next state values
        window = F.pad(window, [0, num_return_steps - 1], value=1.0)
        window = window.roll(1, dims=-1)
        window[..., 0] = 1.0
        window = window.cumprod(dim=-1)
        offsets = delta_weights * rewards
        next_state_value_coeffs = mask.type_as(rewards)
        offsets = F.pad(offsets[None], [0, num_return_steps - 1])
        next_state_value_coeffs = F.pad(
            next_state_value_coeffs[None], [0, num_return_steps - 1]
        )
        offsets = F.conv1d(offsets, window[:, None, :num_return_steps], groups=B)[0]
        next_state_value_coeffs = (
            discount
            ** F.conv1d(
                next_state_value_coeffs,
                torch.ones(B, 1, num_return_steps, device=offsets.device),
                groups=B,
            )[0]
        )
        next_state_value_coeffs *= ~terminals
        next_state_value_coeffs *= mask
        # Gather
        idx = torch.arange(num_return_steps - 1, T + num_return_steps - 1).expand(B, T)
        next_state_values = distributional_gather(
            next_state_values,
            1,
            idx.clamp(max=(length - 1).clamp(min=0)),
        )
        # Transform
        targets = TransformedDistribution(
            next_state_values,
            AffineTransform(offsets, next_state_value_coeffs),
            next_state_values._validate_args,
        ).reduced_dist
        return targets, state_values, next_state_values
    coeffs = torch.stack(
        [
            delta_weights * rewards,  # Offsets
            -delta_weights,  # state_or_action_values coefficients
            discount * delta_weights,  # next_state_values coefficients
        ]
    )
    coeffs = F.pad(coeffs, [0, num_return_steps - 1])
    window = F.pad(window, [0, num_return_steps - 1], value=1.0)
    # Pad mask using edge values
    mask = F.pad(mask, [0, num_return_steps - 1])
    mask[:, T - 1 :] = mask[:, T - 1 : T]
    coeffs = F.unfold(coeffs[..., None], (T, 1)).reshape(3, B, T, -1)
    window = F.unfold(window[:, None, :, None], (T, 1))
    mask = F.unfold(mask[:, None, :, None].type_as(rewards), (T, 1)).bool()
    window = window.roll(1, dims=-1)
    window[..., 0] = 1
    window = window.cumprod(dim=-1)
    coeffs *= window
    coeffs[1, :, :, 0] += 1
    coeffs[2, ...] *= ~terminals[..., None]
    coeffs *= mask
    state_or_action_values = _distributional_unfoldNd(
        state_or_action_values, (B, T, num_return_steps)
    )
    next_state_values = _distributional_unfoldNd(
        next_state_values, (B, T, num_return_steps)
    )
    # Transform
    validate_args = (
        state_or_action_values._validate_args or next_state_values._validate_args
    )
    targets = TransformedDistribution(
        CatDistribution(
            [state_or_action_values, next_state_values],
            dim=-1,
            validate_args=validate_args,
        ),
        [
            AffineTransform(
                torch.stack([coeffs[0], torch.zeros_like(coeffs[0])], dim=-1),
                torch.stack([coeffs[1], coeffs[2]], dim=-1),
            ),
            SumTransform((2,)),
            SumTransform((num_return_steps,)),
        ],
        validate_args=validate_args,
    ).reduced_dist
    return targets, state_values, next_state_values


def _compute_advantages(
    targets: "Tensor",
    state_values: "Tensor",
    rewards: "Tensor",
    advantage_weights: "Tensor",
    action_values: "Optional[Tensor]",
    next_state_values: "Tensor",
    mask: "Tensor",
    discount: "float",
    trace_decay: "float",
) -> "Tensor":
    # x[~mask] = 0.0 is equivalent to x *= mask
    # x[~mask] = 1.0 is equivalent to x *= mask, x += ~mask
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
    advantage_weights *= mask
    advantage_weights += ~mask
    B, T = rewards.shape
    if action_values is not None:
        # Replace action values with current action values estimate (Retrace-like)
        action_values = targets
    else:
        # Compute action values from current state values estimate (V-trace-like)
        length = mask.sum(dim=1)
        next_state_values *= mask
        next_targets = F.pad(
            trace_decay * targets + (1 - trace_decay) * state_values,
            [0, 1],
        )[:, 1:]
        next_targets[torch.arange(B), length - 1] = next_state_values[
            torch.arange(B), length - 1
        ]
        action_values = rewards + discount * next_targets
    advantages = advantage_weights * (action_values - state_values)
    advantages *= mask
    return advantages


def _distributional_unfoldNd(
    distribution: "Distribution",
    target_shape: "Tuple[int, ...]",
) -> "Distribution":
    expand_backup = torch.Tensor.expand

    def expand(input: "Tensor", *args: "Any", **kwargs: "Any") -> "Tensor":
        B, T, N = target_shape[0:3]
        is_bool_input = input.dtype == torch.bool
        if is_bool_input:
            input = input.float()
        if input.ndim == 2:
            result = F.pad(input, [0, N - 1])
            result = F.unfold(result[:, None, :, None], (T, 1))
        else:
            result = input.reshape(B, 1, T, 1, -1)
            result = F.pad(result, [0, 0, 0, 0, 0, N - 1]).movedim(-1, 0)
            result = torch.stack([F.unfold(x, (T, 1)) for x in result], dim=-1)
            result = result.reshape(B, T, N, *input.shape[2:])
        if is_bool_input:
            result = result.bool()
        return result

    torch.Tensor.expand = expand
    try:
        return distribution.expand(torch.Size(target_shape))
    finally:
        torch.Tensor.expand = expand_backup
