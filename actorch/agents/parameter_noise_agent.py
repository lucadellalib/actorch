# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Parameter noise agent."""

from copy import deepcopy
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.deterministic_agent import DeterministicAgent
from actorch.agents.stochastic_agent import StochasticAgent


__all__ = [
    "ParameterNoiseAgent",
    "policy_distance_ddpg",
    "policy_distance_dqn",
]


def policy_distance_ddpg(
    policy: "Policy",
    noisy_policy: "Policy",
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
    .. [1] M. Plappert, R. Houthooft, P. Dhariwal, S. Sidor, R. Y. Chen, X. Chen, T. Asfour,
           P. Abbeel, and M. Andrychowicz. "Parameter Space Noise for Exploration".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1706.01905

    """
    delta = policy.distribution.sample() - noisy_policy.distribution.sample()
    return (delta ** 2).mean().sqrt().item()


def policy_distance_dqn(
    policy: "Policy",
    noisy_policy: "Policy",
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
    .. [1] M. Plappert, R. Houthooft, P. Dhariwal, S. Sidor, R. Y. Chen, X. Chen, T. Asfour,
           P. Abbeel, and M. Andrychowicz. "Parameter Space Noise for Exploration".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1706.01905

    """
    softmax = policy.distribution.sample().softmax(dim=-1)
    noisy_softmax = noisy_policy.distribution.sample().softmax(dim=-1)
    return F.kl_div(softmax, noisy_softmax, reduction="none").sum(dim=-1).mean().item()


class ParameterNoiseAgent(StochasticAgent, DeterministicAgent):
    """Agent that adds adaptive Gaussian noise to the policy weights.

    References
    ----------
    .. [1] M. Plappert, R. Houthooft, P. Dhariwal, S. Sidor, R. Y. Chen, X. Chen, T. Asfour,
           P. Abbeel, and M. Andrychowicz. "Parameter Space Noise for Exploration".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1706.01905

    """

    def __init__(
        self,
        policy: "Policy",
        observation_space: "Space",
        action_space: "Space",
        policy_distance_fn: "Callable[[Policy, Policy], float]",
        is_batched: "bool" = False,
        initial_stddev: "float" = 0.1,
        target_stddev: "float" = 0.2,
        adaption_coeff: "float" = 1.01,
        num_random_timesteps: "int" = 0,
        device: "Optional[Union[device, str]]" = "cpu",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        policy:
            The policy.
        observation_space:
            The (possibly batched) observation space.
        action_space:
            The (possibly batched) action space.
        policy_distance_fn:
            The function that computes the distance between a
            policy and a noisy policy. This function receives
            as arguments the policy and the noisy policy, and
            returns the corresponding distance.
        is_batched:
            True if `observation_space` and `action_space`
            are batched, False otherwise.
        initial_stddev:
            The noise Gaussian distribution initial standard deviation
            (`sigma_0` in the literature).
        target_stddev:
            The noise Gaussian distribution target standard deviation
            (`delta` in the literature).
        adaption_coeff:
            The noise Gaussian distribution adaption coefficient
            (`alpha` in the literature).
        num_random_timesteps:
            The number of initial timesteps for which
            a random prediction is returned.
        device:
            The device.

        Raises
        ------
        ValueError
            If `initial_stddev` or `target_stddev` are not in the interval (0, inf) or
            `adaption_coeff` is not in the interval (1, inf).

        """
        if initial_stddev <= 0.0:
            raise ValueError(
                f"`initial_stddev` ({initial_stddev}) must be in the interval (0, inf)"
            )
        if target_stddev <= 0.0:
            raise ValueError(
                f"`target_stddev` ({target_stddev}) must be in the interval (0, inf)"
            )
        if adaption_coeff <= 1.0:
            raise ValueError(
                f"`adaption_coeff` ({adaption_coeff}) must be in the interval (1, inf)"
            )
        self.policy_distance_fn = policy_distance_fn
        self.initial_stddev = initial_stddev
        self.target_stddev = target_stddev
        self.adaption_coeff = adaption_coeff
        self._stddev = self.initial_stddev
        self._noises = None
        StochasticAgent.__init__(
            self,
            policy,
            observation_space,
            action_space,
            is_batched,
            num_random_timesteps,
            device,
        )

    # override
    def _reset(self, mask: "ndarray") -> "None":
        super()._reset(mask)
        if self._noises:
            self._step()
        # Sample parameter noise
        with torch.no_grad():
            self._noises = [
                torch.normal(0.0, self._stddev, param.shape, device=param.device)
                for param in self.policy.parameters()
            ]

    # override
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        with torch.no_grad():
            # Add parameter noise
            for param, noise in zip(self.policy.parameters(), self._noises):
                param += noise
        flat_action, log_prob = DeterministicAgent._predict(self, flat_observation)
        with torch.no_grad():
            # Remove parameter noise
            for param, noise in zip(self.policy.parameters(), self._noises):
                param -= noise
        return flat_action, log_prob

    def _step(self) -> "None":
        # For simplicity, do not sample from experience
        # replay buffer but from observation space
        flat_observation = self._flat_observation_space.sample()
        policy_state_backup = self._policy_state.clone()
        # Noisy policy
        self._policy_state[...] = 0.0
        self._stochastic_predict(flat_observation)
        noisy_policy = deepcopy(self.policy)
        # Policy
        self._noises.clear()
        self._policy_state[...] = 0.0
        self._stochastic_predict(flat_observation)
        policy = self.policy
        # Restore state
        self._policy_state = policy_state_backup
        # Compute distance
        distance = self.policy_distance_fn(policy, noisy_policy)
        if distance < self.target_stddev:
            self._stddev *= self.adaption_coeff
        else:
            self._stddev /= self.adaption_coeff

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(policy: {self.policy}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"policy_distance_fn: {self.policy_distance_fn}, "
            f"is_batched: {self.is_batched}, "
            f"initial_stddev: {self.initial_stddev}, "
            f"target_stddev: {self.target_stddev}, "
            f"adaption_coeff: {self.adaption_coeff}, "
            f"num_random_timesteps: {self.num_random_timesteps}, "
            f"device: {self.device})"
        )
