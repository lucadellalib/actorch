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

"""Importance sampling."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from actorch.algorithms.value_estimation.generalized_estimator import (
    generalized_estimator,
)


__all__ = [
    "importance_sampling",
]


def importance_sampling(
    state_values: "Union[Tensor, Distribution]",
    action_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    terminals: "Tensor",
    log_is_weights: "Tensor",
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (possibly distributional) importance sampling targets,
    a.k.a. IS, and the corresponding advantages of a trajectory.

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

    Returns
    -------
        - The (possibly distributional) importance sampling targets,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    References
    ----------
    .. [1] D. Precup, R. S. Sutton, and S. Singh.
           "Eligibility traces for off-policy policy evaluation".
           In: ICML. 2000.
           URL: https://scholarworks.umass.edu/cs_faculty_pubs/80/
    .. [2] D. Precup, R. S. Sutton, and S. Dasgupta.
           "Off-policy temporal-difference learning with function approximation".
           In: ICML. 2001, pp. 417-424.
           URL: https://www.cs.mcgill.ca/~dprecup/publications/PSD-01.pdf

    """
    is_weights = log_is_weights.exp()
    trace_weights = is_weights
    delta_weights = torch.ones_like(is_weights)
    advantage_weights = torch.ones_like(is_weights)
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
        trace_decay=1.0,
    )
