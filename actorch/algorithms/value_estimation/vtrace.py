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
    terminals: "Tensor",
    log_is_weights: "Tensor",
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    num_return_steps: "int" = 1,
    trace_decay: "float" = 1.0,
    leakage: "float" = 1.0,
    max_is_weight_trace: "float" = 1.0,
    max_is_weight_delta: "float" = 1.0,
    max_is_weight_advantage: "float" = 1.0,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (possibly distributional) (leaky) V-trace targets, a.k.a.
    V-trace(n), and the corresponding advantages of a trajectory.

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

    Returns
    -------
        - The (possibly distributional) (leaky) V-trace targets
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    References
    ----------
    .. [1] L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron,
           V. Firoiu, T. Harley, I. Dunning, S. Legg, and K. Kavukcuoglu.
           "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures".
           In: ICML. 2018.
           URL: https://arxiv.org/abs/1802.01561
    .. [2] T. Zahavy, Z. Xu, V. Veeriah, M. Hessel, J. Oh, H. Van Hasselt, D. Silver, and S. Singh.
           "A Self-Tuning Actor-Critic Algorithm".
           In: NeurIPS. 2020, pp. 20913-20924.
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
        terminals=terminals,
        trace_weights=trace_weights,
        delta_weights=delta_weights,
        advantage_weights=advantage_weights,
        mask=mask,
        discount=discount,
        num_return_steps=num_return_steps,
        trace_decay=trace_decay,
    )
