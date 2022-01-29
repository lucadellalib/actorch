# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
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
    dones: "Tensor",
    mask: "Tensor",
    log_is_weights: "Tensor",
    next_state_values: "Optional[Union[Tensor, Distribution]]" = None,
    discount: "float" = 0.99,
    standardize_advantage: "bool" = False,
    epsilon: "float" = 1e-6,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (distributional) importance sampling targets, a.k.a. IS, and the corresponding
    advantages of a trajectory.

    In the following, let `B` denote the batch size and `T` the trajectory maximum length.

    Parameters
    ----------
    state_values:
        The (distributional) state values (`v_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
    action_values:
        The (distributional) action values (`q_t` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
    rewards:
        The rewards (`r_t` in the literature), shape: ``[B, T]``.
    dones:
        The end-of-episode flags, shape: ``[B, T]``.
    mask:
        The boolean tensor indicating which elements (or batch elements
        if distributional) are valid (True) and which are not (False),
        shape: ``[B, T]``.
    log_is_weights:
        The log importance sampling weights, defined as
        ``target_policy.log_prob(action) - behavior_policy.log_prob(action)``,
        where `target_policy` (`pi` in the literature) is the policy being learned about,
        and `behavior_policy` (`mu` in the literature) the one used to generate the data,
        shape: ``[B, T]``.
    next_state_values:
        The (distributional) next state values (`v_t+1` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
        Default to `state_values` from the second timestep on, with a bootstrap value set to 0.
    discount:
        The discount factor (`gamma` in the literature).
    standardize_advantage:
        True to standardize the advantages, False otherwise.
    epsilon:
        The term added to the denominators to improve numerical stability.

    Returns
    -------
        - The (distributional) importance sampling targets,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    References
    ----------
    .. [1] D. Precup, R. S. Sutton, and S. Singh. "Eligibility traces for off-policy
           policy evaluation". In: ICML. 2000.
           URL: https://scholarworks.umass.edu/cs_faculty_pubs/80/
    .. [2] D. Precup, R. S. Sutton, and S. Dasgupta. "Off-policy temporal-difference
           learning with function approximation". In: ICML. 2001, pp. 417-424.
           URL: https://www.cs.mcgill.ca/~dprecup/publications/PSD-01.pdf

    """
    is_weights = log_is_weights.exp()
    trace_weights = is_weights
    delta_weights = torch.ones_like(is_weights)
    advantage_weights = torch.ones_like(is_weights)
    return generalized_estimator(
        state_values=state_values,
        rewards=rewards,
        dones=dones,
        mask=mask,
        trace_weights=trace_weights,
        delta_weights=delta_weights,
        advantage_weights=advantage_weights,
        action_values=action_values,
        next_state_values=next_state_values,
        discount=discount,
        num_return_steps=mask.shape[1],
        trace_decay=1.0,
        standardize_advantage=standardize_advantage,
        epsilon=epsilon,
    )
