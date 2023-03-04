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
    terminals: "Tensor",
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    trace_decay: "float" = 1.0,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (possibly distributional) lambda returns, a.k.a. TD(lambda),
    and the corresponding advantages, a.k.a. GAE(lambda), of a trajectory.

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
        - The (possibly distributional) lambda returns,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    References
    ----------
    .. [1] R. S. Sutton and A. G. Barto.
           "Reinforcement Learning: An Introduction".
           MIT Press, 1998.
           URL: http://incompleteideas.net/sutton/book/ebook/node74.html
    .. [2] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel.
           "High-Dimensional Continuous Control Using Generalized Advantage Estimation".
           In: ICLR. 2016.
           URL: https://arxiv.org/abs/1506.02438

    """
    return vtrace(
        state_values=state_values,
        rewards=rewards,
        terminals=terminals,
        log_is_weights=torch.zeros_like(rewards),
        mask=mask,
        discount=discount,
        num_return_steps=rewards.shape[1],
        trace_decay=trace_decay,
        max_is_weight_trace=1.0,
        max_is_weight_delta=1.0,
        max_is_weight_advantage=1.0,
    )
