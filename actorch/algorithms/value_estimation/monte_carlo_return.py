# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Monte Carlo return."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from actorch.algorithms.value_estimation.n_step_return import n_step_return


__all__ = [
    "monte_carlo_return",
]


def monte_carlo_return(
    rewards: "Tensor",
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
) -> "Tuple[Tensor, Tensor]":
    """Compute the Monte Carlo returns and the corresponding advantages of
    a trajectory.

    In the following, let `B` denote the batch size and `T` the maximum
    trajectory length.

    Parameters
    ----------
    rewards:
        The rewards (`r_t` in the literature), shape: ``[B, T]``.
    mask:
        The boolean tensor indicating which elements (or batch elements
        if distributional) are valid (True) and which are not (False),
        shape: ``[B, T]``.
        Default to ``torch.ones_like(rewards, dtype=torch.bool)``.
    discount:
        The discount factor (`gamma` in the literature).

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
        state_values=torch.zeros_like(rewards),
        rewards=rewards,
        terminals=torch.zeros_like(rewards),
        mask=mask,
        discount=discount,
        num_return_steps=rewards.shape[1],
    )
