# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Stochastic agent."""

from typing import Optional, Tuple, Union

import torch
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.agent import Agent
from actorch.registry import register


__all__ = [
    "StochasticAgent",
]


@register
class StochasticAgent(Agent):
    """Agent that returns a stochastic prediction."""

    def __init__(
        self,
        policy: "Policy",
        observation_space: "Space",
        action_space: "Space",
        is_batched: "bool" = False,
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
        num_random_timesteps:
            The number of initial timesteps for which
            a random prediction is returned.
        device:
            The device.

        Raises
        ------
        ValueError
            If `num_random_timesteps` is not in the integer interval [0, inf).

        """
        if num_random_timesteps < 0 or not float(num_random_timesteps).is_integer():
            raise ValueError(
                f"`num_random_timesteps` ({num_random_timesteps}) "
                f"must be in the integer interval [0, inf)"
            )
        self.num_random_timesteps = int(num_random_timesteps)
        self._num_elapsed_timesteps = 0
        self._policy_state = None
        super().__init__(
            policy,
            observation_space,
            action_space,
            is_batched,
            device,
        )

    # override
    def _reset(self, mask: "ndarray") -> "None":
        if self._policy_state is not None:
            self._policy_state[mask] = 0.0

    # override
    def _predict(self, flat_observation: "ndarray") -> "Tuple[ndarray, ndarray]":
        if self._num_elapsed_timesteps < self.num_random_timesteps:
            random_flat_action = self._flat_action_space.sample()
            random_log_prob = self._flat_action_space.log_prob(random_flat_action)
            self._num_elapsed_timesteps += len(flat_observation)
            return random_flat_action, random_log_prob
        return self._stochastic_predict(flat_observation)

    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        # Add temporal axis
        flat_observation = flat_observation[:, None, ...]
        input = torch.as_tensor(flat_observation, device=self.device)
        with torch.no_grad():
            _, self._policy_state = self.policy(input, self._policy_state)
        prediction = self.policy.distribution.sample()
        log_prob = self.policy.distribution.log_prob(prediction)
        flat_action = self.policy.decode(prediction)
        flat_action, log_prob = (
            flat_action.to("cpu").numpy(),
            log_prob.to("cpu").numpy(),
        )
        # Remove temporal axis
        flat_action, log_prob = flat_action[:, 0, ...], log_prob[:, 0, ...]
        return flat_action, log_prob

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(policy: {self.policy}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"is_batched: {self.is_batched}, "
            f"num_random_timesteps: {self.num_random_timesteps}, "
            f"device: {self.device})"
        )
