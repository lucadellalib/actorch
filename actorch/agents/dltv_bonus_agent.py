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

"""Decaying left truncated variance bonus agent."""

from functools import singledispatch
from typing import Optional, Tuple, Union

import numpy as np
import torch
from gymnasium.spaces import Space
from numpy import ndarray
from torch import Tensor, device
from torch.distributions import Distribution

from actorch.agents.stochastic_agent import StochasticAgent
from actorch.distributions import Finite
from actorch.networks import PolicyNetwork
from actorch.schedules import ConstantSchedule, LambdaSchedule, Schedule


__all__ = [
    "DLTVBonusAgent",
]


@singledispatch
def compute_left_truncated_variance(distribution: "Distribution") -> "Tensor":
    """Compute the left truncated variance
    of a distribution.

    Parameters
    ----------
    distribution:
        The distribution.

    Returns
    -------
        The left truncated variance.

    Raises
    ------
    NotImplementedError
        If the distribution type is not supported.

    Notes
    -----
    Register a custom distribution type as follows:
    >>> from torch.distributions import Distribution
    >>>
    >>> from actorch.agents.dltv_bonus_agent import compute_left_truncated_variance
    >>>
    >>>
    >>> class Custom(Distribution):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @compute_left_truncated_variance.register(Custom)
    >>> def _compute_left_truncated_variance_custom(distribution):
    >>>     # Implementation
    >>>     ...

    References
    ----------
    .. [1] B. Mavrin, S. Zhang, H. Yao, L. Kong, K. Wu, and Y. Yu.
           "Distributional Reinforcement Learning for Efficient Exploration".
           In: ICML. 2019, pp. 4424-4434.
           URL: https://arxiv.org/abs/1905.06125

    """
    raise NotImplementedError(
        f"Unsupported distribution type: "
        f"`{type(distribution).__module__}.{type(distribution).__name__}`. "
        f"Register a custom distribution type through "
        f"decorator `compute_left_truncated_variance.register`"
    )


class DLTVBonusAgent(StochasticAgent):
    """Agent that adds a decaying left truncated variance bonus to the prediction.

    References
    ----------
    .. [1] B. Mavrin, S. Zhang, H. Yao, L. Kong, K. Wu, and Y. Yu.
           "Distributional Reinforcement Learning for Efficient Exploration".
           In: ICML. 2019, pp. 4424-4434.
           URL: https://arxiv.org/abs/1905.06125

    """

    _STATE_VARS = StochasticAgent._STATE_VARS + ["suppression_coeff"]  # override

    # override
    def __init__(
        self,
        policy_network: "PolicyNetwork",
        observation_space: "Space",
        action_space: "Space",
        clip_action: "bool" = True,
        device: "Union[device, str]" = "cpu",
        num_random_timesteps: "int" = 0,
        suppression_coeff: "Optional[Union[float, Schedule]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        policy_network:
            The policy network.
        observation_space:
            The (possibly batched) observation space.
        action_space:
            The (possibly batched) action space.
        clip_action:
            True to clip the (possibly batched) action to
            the (possibly batched) action space bounds,
            False otherwise.
        device:
            The device.
        num_random_timesteps:
            The number of initial timesteps for which
            a random prediction is returned.
        suppression_coeff:
            The schedule for the suppression coefficient
            (`c_t` in the literature). If a number, it is
            wrapped in an `actorch.schedules.ConstantSchedule`.
            Default to ``LambdaSchedule(
                lambda t: 50 * np.sqrt(np.log(t + 2) / (t + 2))
            )``.

        """
        if suppression_coeff is None:
            self.suppression_coeff = LambdaSchedule(
                lambda t: 50 * np.sqrt(np.log(t + 2) / (t + 2))
            )
        else:
            if not isinstance(suppression_coeff, Schedule):
                suppression_coeff = ConstantSchedule(suppression_coeff)
            self.suppression_coeff = suppression_coeff
        super().__init__(
            policy_network,
            observation_space,
            action_space,
            clip_action,
            device,
            num_random_timesteps,
        )
        self._schedules["suppression_coeff"] = self.suppression_coeff

    # override
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        suppression_coeff = self.suppression_coeff()
        if suppression_coeff < 0.0:
            raise ValueError(
                f"`suppression_coeff` ({suppression_coeff}) must be in the interval [0, inf)"
            )
        # Add time axis
        flat_observation = flat_observation[:, None, ...]
        input = torch.as_tensor(flat_observation, device=self.device)
        with torch.no_grad():
            _, self._policy_network_state = self.policy_network(
                input, self._policy_network_state
            )
            sample = self.policy_network.sample_fn(self.policy_network.distribution)
            left_truncated_variance = compute_left_truncated_variance(
                self.policy_network.distribution
            )
            bonus = suppression_coeff * left_truncated_variance
            flat_action = self.policy_network.predict(sample + bonus).to("cpu").numpy()
        # Remove time axis
        flat_action = flat_action[:, 0, ...]
        log_prob = np.zeros(len(flat_action))
        return flat_action, log_prob

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(policy_network: {self.policy_network}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"clip_action: {self.clip_action}, "
            f"device: {self.device}, "
            f"num_random_timesteps: {self.num_random_timesteps}, "
            f"suppression_coeff: {self.suppression_coeff})"
        )


#####################################################################################################
# compute_left_truncated_variance implementation
#####################################################################################################


@compute_left_truncated_variance.register(Finite)
def _compute_left_truncated_variance_finite(distribution: "Finite") -> "Tensor":
    probs = distribution.probs
    if not (probs == probs.select(-1, 0).unsqueeze(-1)).all(dim=-1).all():
        raise ValueError(
            f"`distribution.probs` ({probs}) must "
            "be identical along the last dimension"
        )
    num_atoms = distribution.atoms.shape[-1]
    median_idx = num_atoms // 2
    median = distribution.atoms[..., median_idx]
    left_truncated_variance = (
        (distribution.atoms[..., median_idx:] - median) ** 2
    ).sum(dim=-1) / (2 * num_atoms)
    return left_truncated_variance
