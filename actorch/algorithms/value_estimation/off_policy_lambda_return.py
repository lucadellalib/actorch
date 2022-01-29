# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Off-policy lambda return."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from actorch.algorithms.value_estimation.generalized_estimator import (
    generalized_estimator,
)


__all__ = [
    "off_policy_lambda_return",
]


def off_policy_lambda_return(
    state_values: "Union[Tensor, Distribution]",
    action_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    dones: "Tensor",
    mask: "Tensor",
    next_state_values: "Optional[Union[Tensor, Distribution]]" = None,
    discount: "float" = 0.99,
    trace_decay: "float" = 1.0,
    standardize_advantage: "bool" = False,
    epsilon: "float" = 1e-6,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (distributional) off-policy lambda returns, a.k.a. Harutyunyan's et al. Q(lambda),
    and the corresponding advantages of a trajectory.

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
    next_state_values:
        The (distributional) next state values (`v_t+1` in the literature),
        shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``.
        Default to `state_values` from the second timestep on, with a bootstrap value set to 0.
    discount:
        The discount factor (`gamma` in the literature).
    trace_decay:
        The trace-decay parameter (`lambda` in the literature).
    standardize_advantage:
        True to standardize the advantages, False otherwise.
    epsilon:
        The term added to the denominators to improve numerical stability.

    Returns
    -------
        - The (distributional) off-policy lambda returns,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape ``[B, T]``.

    References
    ----------
    .. [1] A. Harutyunyan, M. G. Bellemare, T. Stepleton, and R. Munos. "Q(lambda) with
           off-policy corrections". In: Algorithmic Learning Theory (2016), pp. 305-320.
           URL: https://arxiv.org/abs/1602.04951

    """
    trace_weights = trace_decay * torch.ones_like(mask)
    delta_weights = torch.ones_like(mask)
    advantage_weights = torch.ones_like(mask)
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
        trace_decay=trace_decay,
        standardize_advantage=standardize_advantage,
        epsilon=epsilon,
    )
