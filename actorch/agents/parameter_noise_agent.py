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

"""Parameter noise agent."""

from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from gymnasium.spaces import Space
from numpy import ndarray
from torch import device
from torch.distributions import Distribution, kl_divergence

from actorch.agents.stochastic_agent import StochasticAgent
from actorch.networks import PolicyNetwork
from actorch.schedules import ConstantSchedule, Schedule


__all__ = [
    "ParameterNoiseAgent",
    "compute_policy_distance_ddpg",
    "compute_policy_distance_dqn",
    "compute_policy_distance_reinforce",
]


def compute_policy_distance_ddpg(
    policy: "Distribution",
    noisy_policy: "Distribution",
) -> "float":
    """Compute the DDPG-style distance between a
    policy and a noisy policy.

    Parameters
    ----------
    policy:
        The policy.
    noisy_policy:
        The noisy policy.

    Returns
    -------
        The DDPG-style distance.

    References
    ----------
    .. [1] M. Plappert, R. Houthooft, P. Dhariwal, S. Sidor, R. Y. Chen,
           X. Chen, T. Asfour, P. Abbeel, and M. Andrychowicz.
           "Parameter Space Noise for Exploration".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1706.01905

    """
    delta = policy.sample() - noisy_policy.sample()
    return (delta**2).mean().sqrt().item()


def compute_policy_distance_dqn(
    policy: "Distribution",
    noisy_policy: "Distribution",
) -> "float":
    """Compute the DQN-style distance between a
    policy and a noisy policy.

    Parameters
    ----------
    policy:
        The policy.
    noisy_policy:
        The noisy policy.

    Returns
    -------
        The DQN-style distance.

    References
    ----------
    .. [1] M. Plappert, R. Houthooft, P. Dhariwal, S. Sidor, R. Y. Chen,
           X. Chen, T. Asfour, P. Abbeel, and M. Andrychowicz.
           "Parameter Space Noise for Exploration".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1706.01905

    """
    softmax = policy.sample().softmax(dim=-1)
    noisy_softmax = noisy_policy.sample().softmax(dim=-1)
    return F.kl_div(softmax, noisy_softmax, reduction="none").sum(dim=-1).mean().item()


def compute_policy_distance_reinforce(
    policy: "Distribution",
    noisy_policy: "Distribution",
) -> "float":
    """Compute the REINFORCE-style distance between a
    policy and a noisy policy.

    Parameters
    ----------
    policy:
        The policy.
    noisy_policy:
        The noisy policy.

    Returns
    -------
        The REINFORCE-style distance.

    """
    return kl_divergence(policy, noisy_policy).mean().item()


class ParameterNoiseAgent(StochasticAgent):
    """Agent that adds adaptive Gaussian noise to the policy weights.

    References
    ----------
    .. [1] M. Plappert, R. Houthooft, P. Dhariwal, S. Sidor, R. Y. Chen,
           X. Chen, T. Asfour, P. Abbeel, and M. Andrychowicz.
           "Parameter Space Noise for Exploration".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1706.01905

    """

    _STATE_VARS = StochasticAgent._STATE_VARS + [
        "initial_stddev",
        "target_stddev",
        "adaption_coeff",
        "_stddev",
        "_noises",
    ]  # override

    # override
    def __init__(
        self,
        policy_network: "PolicyNetwork",
        observation_space: "Space",
        action_space: "Space",
        policy_distance_fn: "Callable[[Distribution, Distribution], float]",
        clip_action: "bool" = True,
        device: "Union[device, str]" = "cpu",
        num_random_timesteps: "int" = 0,
        initial_stddev: "Union[float, Schedule]" = 0.1,
        target_stddev: "Union[float, Schedule]" = 0.2,
        adaption_coeff: "Union[float, Schedule]" = 1.01,
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
        policy_distance_fn:
            The function that computes the distance between a
            policy and a noisy policy. It receives as arguments
            the policy and the noisy policy and returns the
            corresponding distance.
        clip_action:
            True to clip the (possibly batched) action to
            the (possibly batched) action space bounds,
            False otherwise.
        device:
            The device.
        num_random_timesteps:
            The number of initial timesteps for which
            a random prediction is returned.
        initial_stddev:
            The schedule for the noise Gaussian distribution initial
            standard deviation (`sigma_0` in the literature).
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
        target_stddev:
            The schedule for the noise Gaussian distribution target
            standard deviation (`delta` in the literature).
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
        adaption_coeff:
            The schedule for the noise Gaussian distribution
            adaption coefficient (`alpha` in the literature).
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.

        """
        self.policy_distance_fn = policy_distance_fn
        self.initial_stddev = (
            initial_stddev
            if isinstance(initial_stddev, Schedule)
            else ConstantSchedule(initial_stddev)
        )
        self.target_stddev = (
            target_stddev
            if isinstance(target_stddev, Schedule)
            else ConstantSchedule(target_stddev)
        )
        self.adaption_coeff = (
            adaption_coeff
            if isinstance(adaption_coeff, Schedule)
            else ConstantSchedule(adaption_coeff)
        )

        initial_stddev = self.initial_stddev()
        if initial_stddev <= 0.0:
            raise ValueError(
                f"`initial_stddev` ({initial_stddev}) must be in the interval (0, inf)"
            )
        self._stddev = initial_stddev
        self._noises = None
        super().__init__(
            policy_network,
            observation_space,
            action_space,
            clip_action,
            device,
            num_random_timesteps,
        )
        self._schedules["initial_stddev"] = self.initial_stddev
        self._schedules["target_stddev"] = self.target_stddev
        self._schedules["adaption_coeff"] = self.adaption_coeff

    # override
    def _reset(self, mask: "ndarray") -> "None":
        super()._reset(mask)
        if self._noises is not None and self._policy_network_state is not None:
            self._step()
        # Sample parameter noise
        with torch.no_grad():
            self._noises = [
                torch.normal(0.0, self._stddev, param.shape, device=param.device)
                for param in self.policy_network.parameters()
            ]

    # override
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        with torch.no_grad():
            # Add parameter noise
            for param, noise in zip(self.policy_network.parameters(), self._noises):
                param += noise
        flat_action, log_prob = super(StochasticAgent, self)._predict(flat_observation)
        with torch.no_grad():
            # Remove parameter noise
            for param, noise in zip(self.policy_network.parameters(), self._noises):
                param -= noise
        return flat_action, log_prob

    def _step(self) -> "None":
        initial_stddev = self.initial_stddev()
        if initial_stddev <= 0.0:
            raise ValueError(
                f"`initial_stddev` ({initial_stddev}) must be in the interval (0, inf)"
            )
        target_stddev = self.target_stddev()
        if target_stddev <= 0.0:
            raise ValueError(
                f"`target_stddev` ({target_stddev}) must be in the interval (0, inf)"
            )
        adaption_coeff = self.adaption_coeff()
        if adaption_coeff <= 1.0:
            raise ValueError(
                f"`adaption_coeff` ({adaption_coeff}) must be in the interval (1, inf)"
            )
        # For simplicity, do not sample from experience
        # replay buffer but from observation space
        flat_observation = self._flat_observation_space.sample()
        policy_network_state_backup = self._policy_network_state.clone()
        # Compute noisy policy
        self._policy_network_state[...] = 0.0
        self._stochastic_predict(flat_observation)
        noisy_policy = self.policy_network.distribution
        # Compute policy
        self._noises.clear()
        self._policy_network_state[...] = 0.0
        self._stochastic_predict(flat_observation)
        policy = self.policy_network.distribution
        # Restore state
        self._policy_network_state = policy_network_state_backup
        # Compute policy distance
        policy_distance = self.policy_distance_fn(policy, noisy_policy)
        if policy_distance < target_stddev:
            self._stddev *= adaption_coeff
        else:
            self._stddev /= adaption_coeff

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(policy_network: {self.policy_network}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"policy_distance_fn: {self.policy_distance_fn}, "
            f"clip_action: {self.clip_action}, "
            f"device: {self.device}, "
            f"num_random_timesteps: {self.num_random_timesteps}, "
            f"initial_stddev: {self.initial_stddev}, "
            f"target_stddev: {self.target_stddev}, "
            f"adaption_coeff: {self.adaption_coeff})"
        )
