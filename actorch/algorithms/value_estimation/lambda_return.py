# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Lambda return."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from actorch.algorithms.value_estimation.vtrace import vtrace


__all__ = [
    "lambda_return",
]


def lambda_return(
    state_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    dones: "Tensor",
    mask: "Tensor",
    next_state_values: "Optional[Union[Tensor, Distribution]]" = None,
    discount: "float" = 0.99,
    trace_decay: "float" = 1.0,
    standardize_advantage: "bool" = False,
    epsilon: "float" = 1e-6,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (distributional) lambda returns, a.k.a. TD(lambda), and
    the corresponding advantages, a.k.a. GAE(lambda), of a trajectory.

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
        - The (distributional) lambda returns,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape ``[B, T]``.

    References
    ----------
    .. [1] R. S. Sutton and A. G. Barto. "Reinforcement Learning: An Introduction".
           MIT Press, 1998.
           URL: http://incompleteideas.net/sutton/book/ebook/node74.html
    .. [2] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. "High-Dimensional
           Continuous Control Using Generalized Advantage Estimation". In: ICLR. 2016.
           URL: https://arxiv.org/abs/1506.02438

    """
    return vtrace(
        state_values=state_values,
        rewards=rewards,
        dones=dones,
        mask=mask,
        log_is_weights=torch.zeros_like(mask),
        next_state_values=next_state_values,
        discount=discount,
        num_return_steps=mask.shape[1],
        trace_decay=trace_decay,
        max_is_weight_trace=1.0,
        max_is_weight_delta=1.0,
        max_is_weight_advantage=1.0,
        standardize_advantage=standardize_advantage,
        epsilon=epsilon,
    )
