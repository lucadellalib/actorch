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

"""Stochastic agent."""

from typing import Tuple, Union

import torch
from gymnasium.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.agent import Agent
from actorch.networks import PolicyNetwork


__all__ = [
    "StochasticAgent",
]


class StochasticAgent(Agent):
    """Agent that returns a stochastic prediction."""

    _STATE_VARS = Agent._STATE_VARS + [
        "num_random_timesteps",
        "_num_elapsed_timesteps",
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

        Raises
        ------
        ValueError
            If `num_random_timesteps` is not in
            the integer interval [0, inf).

        """
        if num_random_timesteps < 0 or not float(num_random_timesteps).is_integer():
            raise ValueError(
                f"`num_random_timesteps` ({num_random_timesteps}) "
                f"must be in the integer interval [0, inf)"
            )
        self.num_random_timesteps = int(num_random_timesteps)
        self._num_elapsed_timesteps = 0
        super().__init__(
            policy_network,
            observation_space,
            action_space,
            clip_action,
            device,
        )

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
        # Add time axis
        flat_observation = flat_observation[:, None, ...]
        input = torch.as_tensor(flat_observation, device=self.device)
        with torch.no_grad():
            _, self._policy_network_state = self.policy_network(
                input, self._policy_network_state
            )
            sample = self.policy_network.distribution.sample()
            flat_action = self.policy_network.predict(sample).to("cpu").numpy()
            log_prob = (
                self.policy_network.distribution.log_prob(sample).to("cpu").numpy()
            )
        # Remove time axis
        flat_action, log_prob = flat_action[:, 0, ...], log_prob[:, 0, ...]
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
            f"num_random_timesteps: {self.num_random_timesteps})"
        )
