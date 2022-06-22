# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""V-trace."""

from typing import Optional, Tuple, Union

from torch import Tensor
from torch.distributions import Distribution

from actorch.algorithms.value_estimation.generalized_estimator import (
    generalized_estimator,
)


__all__ = [
    "vtrace",
]


def vtrace(
    state_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    dones: "Tensor",
    mask: "Tensor",
    log_is_weights: "Tensor",
    next_state_values: "Optional[Union[Tensor, Distribution]]" = None,
    discount: "float" = 0.99,
    num_return_steps: "int" = 1,
    trace_decay: "float" = 1.0,
    leakage: "float" = 1.0,
    max_is_weight_trace: "float" = 1.0,
    max_is_weight_delta: "float" = 1.0,
    max_is_weight_advantage: "float" = 1.0,
    standardize_advantage: "bool" = False,
    epsilon: "float" = 1e-6,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (distributional) (leaky) V-trace targets, a.k.a. V-trace(n), and the corresponding
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
    num_return_steps:
        The number of return steps (`n` in the literature).
    trace_decay:
        The trace-decay parameter (`lambda` in the literature).
    leakage:
        The leakage factor (`alpha` in the literature).
    max_is_weight_trace:
        The maximum importance sampling weight for trace computation (`c_bar` in the literature).
    max_is_weight_delta:
        The maximum importance sampling weight for delta computation (`rho_bar` in the literature).
    max_is_weight_advantage:
        The maximum importance sampling weight for advantage computation.
    standardize_advantage:
        True to standardize the advantages, False otherwise.
    epsilon:
        The term added to the denominators to improve numerical stability.

    Returns
    -------
        - The (distributional) (leaky) V-trace targets
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    References
    ----------
    .. [1] L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron, V. Firoiu,
           T. Harley, I. Dunning, S. Legg, and K. Kavukcuoglu. "IMPALA: Scalable Distributed
           Deep-RL with Importance Weighted Actor-Learner Architectures". In: ICML. 2018.
           URL: https://arxiv.org/abs/1802.01561
    .. [2] T. Zahavy, Z. Xu, V. Veeriah, M. Hessel, J. Oh, H. Van Hasselt, D. Silver, and S. Singh.
           "A Self-Tuning Actor-Critic Algorithm". In: NeurIPS. 2020, pp. 20913-20924.
           URL: https://arxiv.org/abs/2002.12928

    """
    if leakage < 0.0 or leakage > 1.0:
        raise ValueError(f"`leakage` ({discount}) must be in the interval [0, 1]")
    if max_is_weight_trace <= 0.0:
        raise ValueError(
            f"`max_is_weight_trace` ({max_is_weight_trace}) must be in the interval (0, inf)"
        )
    if max_is_weight_delta <= 0.0:
        raise ValueError(
            f"`max_is_weight_delta` ({max_is_weight_delta}) must be in the interval (0, inf)"
        )
    if max_is_weight_advantage <= 0.0:
        raise ValueError(
            f"`max_is_weight_advantage` ({max_is_weight_advantage}) must be in the interval (0, inf)"
        )
    is_weights = log_is_weights.exp()
    trace_weights = (
        leakage * trace_decay * is_weights.clamp(max=max_is_weight_trace)
        + (1 - leakage) * is_weights
    )
    delta_weights = (
        leakage * is_weights.clamp(max=max_is_weight_delta) + (1 - leakage) * is_weights
    )
    advantage_weights = (
        leakage * is_weights.clamp(max=max_is_weight_advantage)
        + (1 - leakage) * is_weights
    )
    return generalized_estimator(
        state_values=state_values,
        rewards=rewards,
        dones=dones,
        mask=mask,
        trace_weights=trace_weights,
        delta_weights=delta_weights,
        advantage_weights=advantage_weights,
        next_state_values=next_state_values,
        discount=discount,
        num_return_steps=num_return_steps,
        trace_decay=trace_decay,
        standardize_advantage=standardize_advantage,
        epsilon=epsilon,
    )
