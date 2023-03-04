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

"""Gaussian noise agent."""

from typing import Tuple, Union

import numpy as np
from gymnasium.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.stochastic_agent import StochasticAgent
from actorch.networks import PolicyNetwork
from actorch.schedules import ConstantSchedule, Schedule


__all__ = [
    "GaussianNoiseAgent",
]


class GaussianNoiseAgent(StochasticAgent):
    """Agent that adds Gaussian noise to the action."""

    _STATE_VARS = StochasticAgent._STATE_VARS + ["mean", "stddev"]  # override

    # override
    def __init__(
        self,
        policy_network: "PolicyNetwork",
        observation_space: "Space",
        action_space: "Space",
        clip_action: "bool" = True,
        device: "Union[device, str]" = "cpu",
        num_random_timesteps: "int" = 0,
        mean: "Union[float, Schedule]" = 0.0,
        stddev: "Union[float, Schedule]" = 0.1,
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
            The schedule for the noise Gaussian distribution mean
            (`mu` in the literature). If a number, it is wrapped
            in an `actorch.schedules.ConstantSchedule`.
        stddev:
            The schedule for the noise Gaussian distribution standard
            deviation (`sigma` in the literature). If a number, it is
            wrapped in an `actorch.schedules.ConstantSchedule`.

        """
        self.mean = mean if isinstance(mean, Schedule) else ConstantSchedule(mean)
        self.stddev = (
            stddev if isinstance(stddev, Schedule) else ConstantSchedule(stddev)
        )
        super().__init__(
            policy_network,
            observation_space,
            action_space,
            clip_action,
            device,
            num_random_timesteps,
        )
        self._schedules["mean"] = self.mean
        self._schedules["stddev"] = self.stddev

    # override
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        stddev = self.stddev()
        if stddev <= 0.0:
            raise ValueError(f"`stddev` ({stddev}) must be in the interval (0, inf)")
        flat_action, log_prob = super(StochasticAgent, self)._predict(flat_observation)
        flat_action = flat_action.astype(np.float32)
        flat_action += np.random.normal(
            self.mean(),
            stddev,
            flat_action.shape,
        )
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
            f"mean: {self.mean}, "
            f"stddev: {self.stddev})"
        )
