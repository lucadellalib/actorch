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

"""Tree-backup."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from actorch.algorithms.value_estimation.generalized_estimator import (
    generalized_estimator,
)


__all__ = [
    "tree_backup",
]


def tree_backup(
    state_values: "Union[Tensor, Distribution]",
    action_values: "Union[Tensor, Distribution]",
    rewards: "Tensor",
    terminals: "Tensor",
    target_log_probs: "Tensor",
    mask: "Optional[Tensor]" = None,
    discount: "float" = 0.99,
    trace_decay: "float" = 1.0,
) -> "Tuple[Union[Tensor, Distribution], Tensor]":
    """Compute the (possibly distributional) tree-backup targets, a.k.a. TB(lambda),
    and the corresponding advantages of a trajectory.

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
    target_log_probs:
        The log probabilities of the actions performed by `target_policy`, where
        `target_policy` (`pi` in the literature) is the policy being learned about,
        shape: ``[B, T]``.
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
        - The (possibly distributional) tree-backup targets,
          shape (or batch shape if distributional, assuming an empty event shape): ``[B, T]``;
        - the corresponding advantages, shape: ``[B, T]``.

    References
    ----------
    .. [1] D. Precup, R. S. Sutton, and S. Singh.
           "Eligibility traces for off-policy policy evaluation".
           In: ICML. 2000.
           URL: https://scholarworks.umass.edu/cs_faculty_pubs/80/

    """
    target_probs = target_log_probs.exp()
    trace_weights = trace_decay * target_probs
    delta_weights = torch.ones_like(target_probs)
    advantage_weights = torch.ones_like(target_probs)
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
