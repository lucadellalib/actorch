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

"""Retrace."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from actorch.algorithms.value_estimation.generalized_estimator import (
    generalized_estimator,
)


__all__ = [
    "retrace",
]


def retrace(
    state_values: "Union[Tensor, Distribution]",
    action_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    terminals: "Tensor",
    log_is_weights: "Tensor",
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    trace_decay: "float" = 1.0,
    max_is_weight_trace: "float" = 1.0,
    max_is_weight_advantage: "float" = 1.0,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (possibly distributional) Retrace targets, a.k.a. Retrace(lambda),
    and the corresponding advantages of a trajectory.

    In the following, let `B` denote the batch size and `T` the maximum
    trajectory length.

    Parameters
    ----------
    state_values:
        The (possibly distributional) state values (`v_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape):
        ``[B, T + 1]`` if a bootstrap value is given, ``[B, T]`` otherwise.
    action_values:
        The (possibly distributional) action values (`q_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
    rewards:
        The rewards (`r_t` in the literature), shape: ``[B, T]``.
    terminals:
        The terminal flags, shape: ``[B, T]``.
    log_is_weights:
        The log importance sampling weights, defined as
        ``target_policy.log_prob(action) - behavior_policy.log_prob(action)``,
        where `target_policy` (`pi` in the literature) is the policy being learned about,
        and `behavior_policy` (`mu` in the literature) the one used to generate the data,
        shape: ``[B, T]``.
    mask:
        The boolean tensor indicating which elements (or batch elements
        if distributional) are valid (True) and which are not (False),
        shape: ``[B, T]``.
        Default to ``torch.ones_like(rewards, dtype=torch.bool)``.
    discount:
        The discount factor (`gamma` in the literature).
    trace_decay:
        The trace-decay parameter (`lambda` in the literature).
    max_is_weight_trace:
        The maximum importance sampling weight for trace computation (`c_bar` in the literature).
    max_is_weight_advantage:
        The maximum importance sampling weight for advantage computation.

    Returns
    -------
        - The (possibly distributional) Retrace targets,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    References
    ----------
    .. [1] R. Munos, T. Stepleton, A. Harutyunyan, and M. G. Bellemare.
           "Safe and Efficient Off-Policy Reinforcement Learning".
           In: NeurIPS. 2016, pp. 1054-1062.
           URL: https://arxiv.org/abs/1606.02647

    """
    if max_is_weight_trace <= 0.0:
        raise ValueError(
            f"`max_is_weight_trace` ({max_is_weight_trace}) must be in the interval (0, inf)"
        )
    if max_is_weight_advantage <= 0.0:
        raise ValueError(
            f"`max_is_weight_advantage` ({max_is_weight_advantage}) must be in the interval (0, inf)"
        )
    is_weights = log_is_weights.exp()
    trace_weights = trace_decay * is_weights.clamp(max=max_is_weight_trace)
    delta_weights = torch.ones_like(is_weights)
    advantage_weights = is_weights.clamp(max=max_is_weight_advantage)
    return generalized_estimator(
        state_values=state_values,
        rewards=rewards,
        terminals=terminals,
        trace_weights=trace_weights,
        delta_weights=delta_weights,
        advantage_weights=advantage_weights,
        action_values=action_values,
        mask=mask,
        discount=discount,
        num_return_steps=rewards.shape[1],
        trace_decay=trace_decay,
    )
