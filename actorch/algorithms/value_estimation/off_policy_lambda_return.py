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
    terminals: "Tensor",
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    trace_decay: "float" = 1.0,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (possibly distributional) off-policy lambda returns, a.k.a.
    Harutyunyan's et al. Q(lambda), and the corresponding advantages of a
    trajectory.

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
    mask:
        The boolean tensor indicating which elements (or batch elements
        if distributional) are valid (True) and which are not (False),
        shape: ``[B, T]``.
        Default to ``torch.ones_like(rewards, dtype=torch.bool)``.
    discount:
        The discount factor (`gamma` in the literature).
    trace_decay:
        The trace-decay parameter (`lambda` in the literature).

    Returns
    -------
        - The (possibly distributional) off-policy lambda returns,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    References
    ----------
    .. [1] A. Harutyunyan, M. G. Bellemare, T. Stepleton, and R. Munos.
           "Q(lambda) with off-policy corrections".
           In: Algorithmic Learning Theory (2016), pp. 305-320.
           URL: https://arxiv.org/abs/1602.04951

    """
    trace_weights = torch.full_like(rewards, trace_decay)
    delta_weights = torch.ones_like(rewards)
    advantage_weights = torch.ones_like(rewards)
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
