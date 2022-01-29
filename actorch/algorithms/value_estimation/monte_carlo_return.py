# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Monte Carlo return."""

from typing import Tuple

import torch
from torch import Tensor

from actorch.algorithms.value_estimation.n_step_return import n_step_return


__all__ = [
    "monte_carlo_return",
]


def monte_carlo_return(
    rewards: "Tensor",
    mask: "Tensor",
    discount: "float" = 0.99,
    standardize_advantage: "bool" = False,
    epsilon: "float" = 1e-6,
) -> "Tuple[Tensor, Tensor]":
    """Compute the Monte Carlo returns and the corresponding advantages of a trajectory.

    In the following, let `B` denote the batch size and `T` the trajectory maximum length.

    Parameters
    ----------
    rewards:
        The rewards (`r_t` in the literature), shape: ``[B, T]``.
    mask:
        The boolean tensor indicating which elements
        are valid (True) and which are not (False),
        shape: ``[B, T]``.
    discount:
        The discount factor (`gamma` in the literature).
    standardize_advantage:
        True to standardize the advantages, False otherwise.
    epsilon:
        The term added to the denominators to improve numerical stability.

    Returns
    -------
        - The Monte Carlo returns, shape: ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    References
    ----------
    .. [1] R. S. Sutton and A. G. Barto. "Reinforcement Learning: An Introduction".
           MIT Press, 1998.
           URL: http://incompleteideas.net/sutton/book/ebook/node51.html

    """
    return n_step_return(
        state_values=torch.zeros_like(mask),
        rewards=rewards,
        dones=torch.zeros_like(mask),
        mask=mask,
        discount=discount,
        num_return_steps=mask.shape[1],
        standardize_advantage=standardize_advantage,
        epsilon=epsilon,
    )
