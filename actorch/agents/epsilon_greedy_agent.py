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

"""Epsilon-greedy agent."""

from typing import Optional, Tuple, Union

import numpy as np
from gymnasium.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.agent import Agent
from actorch.networks import PolicyNetwork
from actorch.schedules import ConstantSchedule, LinearSchedule, Schedule


__all__ = [
    "EpsilonGreedyAgent",
]


class EpsilonGreedyAgent(Agent):
    """Agent that returns a random prediction with probability `epsilon`
    (annealed over time) and a deterministic one with probability 1 - `epsilon`.

    """

    _STATE_VARS = Agent._STATE_VARS + ["epsilon"]  # override

    # override
    def __init__(
        self,
        policy_network: "PolicyNetwork",
        observation_space: "Space",
        action_space: "Space",
        clip_action: "bool" = True,
        device: "Union[device, str]" = "cpu",
        epsilon: "Optional[Union[float, Schedule]]" = None,
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
        epsilon:
            The annealing schedule for the epsilon coefficient.
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
            Default to ``LinearSchedule(1.0, 0.05, int(1e5))``.

        """
        if epsilon is None:
            self.epsilon = LinearSchedule(1.0, 0.05, int(1e5))
        else:
            if not isinstance(epsilon, Schedule):
                epsilon = ConstantSchedule(epsilon)
            self.epsilon = epsilon
        super().__init__(
            policy_network,
            observation_space,
            action_space,
            clip_action,
            device,
        )

    # override
    def _predict(self, flat_observation: "ndarray") -> "Tuple[ndarray, ndarray]":
        epsilon = self.epsilon()
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"`epsilon` ({epsilon}) must be in the interval [0, 1]")
        self.epsilon.step()
        flat_action, log_prob = super()._predict(flat_observation)
        explore = np.random.rand(len(flat_observation)) < epsilon
        if not explore.any():
            return flat_action, log_prob
        random_flat_action = self._flat_action_space.sample()
        random_log_prob = self._flat_action_space.log_prob(random_flat_action)
        if not self._batch_size:
            # Add batch axis
            random_flat_action = random_flat_action[None]
            random_log_prob = random_log_prob[None]
        flat_action[explore] = random_flat_action[explore]
        log_prob[explore] = random_log_prob[explore]
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
            f"epsilon: {self.epsilon})"
        )
