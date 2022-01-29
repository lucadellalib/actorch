# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Gaussian noise agent."""

from typing import Optional, Tuple, Union

import numpy as np
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.deterministic_agent import DeterministicAgent
from actorch.agents.stochastic_agent import StochasticAgent
from actorch.registry import register


__all__ = [
    "GaussianNoiseAgent",
]


@register
class GaussianNoiseAgent(StochasticAgent, DeterministicAgent):
    """Agent that adds Gaussian noise to the action."""

    def __init__(
        self,
        policy: "Policy",
        observation_space: "Space",
        action_space: "Space",
        is_batched: "bool" = False,
        mean: "float" = 0.0,
        stddev: "float" = 0.1,
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
        is_batched:
            True if `observation_space` and `action_space`
            are batched, False otherwise.
        mean:
            The noise Gaussian distribution mean
            (`mu` in the literature).
        stddev:
            The noise Gaussian distribution standard deviation
            (`sigma` in the literature).
        num_random_timesteps:
            The number of initial timesteps for which
            a random prediction is returned.
        device:
            The device.

        Raises
        ------
        ValueError
            If `stddev` is not in the interval (0, inf).

        """
        if stddev <= 0.0:
            raise ValueError(f"`stddev` ({stddev}) must be in the interval (0, inf)")
        self.mean = mean
        self.stddev = stddev
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
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        flat_action, log_prob = DeterministicAgent._predict(self, flat_observation)
        flat_action += np.random.normal(
            self.mean,
            self.stddev,
            flat_action.shape,
        )
        return flat_action, log_prob

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(policy: {self.policy}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"is_batched: {self.is_batched}, "
            f"mean: {self.mean}, "
            f"stddev: {self.stddev}, "
            f"num_random_timesteps: {self.num_random_timesteps}, "
            f"device: {self.device})"
        )
