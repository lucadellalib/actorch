# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Ornstein-Uhlenbeck noise agent."""

from typing import Tuple, Union

import numpy as np
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.stochastic_agent import StochasticAgent
from actorch.networks import PolicyNetwork


__all__ = [
    "OUNoiseAgent",
]


class OUNoiseAgent(StochasticAgent):
    """Agent that adds Ornstein-Uhlenbeck noise to the action.

    References
    ----------
    .. [1] G. E. Uhlenbeck and L. S. Ornstein. "On the Theory of the Brownian Motion".
           In: Phys. Rev. 1930, pp. 823-841.
           URL: https://doi.org/10.1103/PhysRev.36.823

    """

    _STATE_VARS = StochasticAgent._STATE_VARS + [
        "mean",
        "volatility",
        "reversion_speed",
        "_ou_state",
    ]  # override

    # override
    def __init__(
        self,
        policy_network: "PolicyNetwork",
        observation_space: "Space",
        action_space: "Space",
        clip_action: "bool" = True,
        device: "Union[device, str]" = "cpu",
        num_random_timesteps: "int" = 0,
        mean: "float" = 0.0,
        volatility: "float" = 0.1,
        reversion_speed: "float" = 0.15,
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
        mean:
            The noise Ornstein-Uhlenbeck process mean
            (`mu` in the literature).
        volatility:
            The noise Ornstein-Uhlenbeck process volatility
            (`sigma` in the literature).
        reversion_speed:
            The noise Ornstein-Uhlenbeck process reversion
            speed (`theta` in the literature).

        Raises
        ------
        ValueError
            If `volatility` or `reversion_speed` are not in
            the interval (0, inf).

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
        self._ou_state = None
        super().__init__(
            policy_network,
            observation_space,
            action_space,
            clip_action,
            device,
            num_random_timesteps,
        )

    # override
    def _reset(self, mask: "ndarray") -> "None":
        super()._reset(mask)
        if self._ou_state is None:
            self._ou_state = np.full(self._batch_size or 1, self.mean)
        self._ou_state[mask] = self.mean

    # override
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        flat_action, log_prob = super(StochasticAgent, self)._predict(flat_observation)
        flat_action = flat_action.astype(np.float32)
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
            f"mean: {self.mean}, "
            f"volatility: {self.volatility}, "
            f"reversion_speed: {self.reversion_speed})"
        )
