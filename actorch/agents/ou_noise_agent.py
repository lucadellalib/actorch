# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Ornstein-Uhlenbeck noise agent."""

from typing import Optional, Tuple, Union

import numpy as np
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.deterministic_agent import DeterministicAgent
from actorch.agents.stochastic_agent import StochasticAgent
from actorch.registry import register


__all__ = [
    "OUNoiseAgent",
]


@register
class OUNoiseAgent(StochasticAgent, DeterministicAgent):
    """Agent that adds Ornstein-Uhlenbeck noise to the action.

    References
    ----------
    .. [1] G. E. Uhlenbeck and L. S. Ornstein. "On the Theory of the Brownian Motion".
           In: Phys. Rev. 1930, pp. 823-841.
           URL: https://doi.org/10.1103/PhysRev.36.823

    """

    def __init__(
        self,
        policy: "Policy",
        observation_space: "Space",
        action_space: "Space",
        is_batched: "bool" = False,
        mean: "float" = 0.0,
        volatility: "float" = 0.1,
        reversion_speed: "float" = 0.15,
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
            The noise Ornstein-Uhlenbeck process mean
            (`mu` in the literature).
        volatility:
            The noise Ornstein-Uhlenbeck process volatility
            (`sigma` in the literature).
        reversion_speed:
            The noise Ornstein-Uhlenbeck process reversion speed
            (`theta` in the literature).
        num_random_timesteps:
            The number of initial timesteps for which
            a random prediction is returned.
        device:
            The device.

        Raises
        ------
        ValueError
            If `volatility` or `reversion_speed` are not in the interval (0, inf).

        """
        if volatility <= 0.0:
            raise ValueError(
                f"`volatility` ({volatility}) must be in the interval (0, inf)"
            )
        if reversion_speed <= 0.0:
            raise ValueError(
                f"`reversion_speed` ({reversion_speed}) must be in the interval (0, inf)"
            )
        self.mean = mean
        self.volatility = volatility
        self.reversion_speed = reversion_speed
        self._ou_state = np.full(self._batch_size or 1, mean)
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
        self._ou_state[mask] = self.mean

    # override
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        flat_action, log_prob = DeterministicAgent._predict(self, flat_observation)
        flat_action += self._ou_state
        self._step()
        return flat_action, log_prob

    def _step(self) -> "None":
        delta = np.random.normal(
            self.reversion_speed * (self.mean - self._ou_state),
            self.volatility,
            len(self._ou_state),
        )
        self._ou_state += delta

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(policy: {self.policy}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"is_batched: {self.is_batched}, "
            f"mean: {self.mean}, "
            f"volatility: {self.volatility}, "
            f"reversion_speed: {self.reversion_speed}, "
            f"num_random_timesteps: {self.num_random_timesteps}, "
            f"device: {self.device})"
        )
