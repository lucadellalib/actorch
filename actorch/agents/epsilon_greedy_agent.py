# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Epsilon-greedy agent."""

from typing import Optional, Tuple, Union

import numpy as np
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.deterministic_agent import DeterministicAgent
from actorch.networks import PolicyNetwork
from actorch.schedules import ConstantSchedule, LinearSchedule, Schedule


__all__ = [
    "EpsilonGreedyAgent",
]


class EpsilonGreedyAgent(DeterministicAgent):
    """Agent that returns a random prediction with probability `epsilon`
    (annealed over time) and a deterministic one with probability `1 - epsilon`.

    """

    def __init__(
        self,
        policy_network: "PolicyNetwork",
        observation_space: "Space",
        action_space: "Space",
        is_batched: "bool" = False,
        epsilon: "Optional[Union[float, Schedule]]" = None,
        device: "Optional[Union[device, str]]" = "cpu",
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
        is_batched:
            True if `observation_space` and `action_space`
            are batched, False otherwise.
        epsilon:
            The epsilon annealing schedule.
            If a number, it is wrapped in a `ConstantSchedule`.
            Default to ``LinearSchedule(1., 0.05, int(1e5))``.
        device:
            The device.

        """
        if epsilon is None:
            self.epsilon = LinearSchedule(1.0, 0.05, int(1e5))
        else:
            self.epsilon = (
                epsilon if isinstance(epsilon, Schedule) else ConstantSchedule(epsilon)
            )
        super().__init__(
            policy_network,
            observation_space,
            action_space,
            is_batched,
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

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(policy_network: {self.policy_network}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"is_batched: {self.is_batched}, "
            f"epsilon: {self.epsilon}, "
            f"device: {self.device})"
        )
