# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Deterministic agent."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.agents.agent import Agent
from actorch.registry import register


__all__ = [
    "DeterministicAgent",
]


@register
class DeterministicAgent(Agent):
    """Agent that returns a deterministic prediction."""

    def __init__(
        self,
        policy: "Policy",
        observation_space: "Space",
        action_space: "Space",
        is_batched: "bool" = False,
        device: "Optional[Union[device, str]]" = "cpu",
    ) -> "None":
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
        # Add temporal axis
        flat_observation = flat_observation[:, None, ...]
        input = torch.as_tensor(flat_observation, device=self.device)
        with torch.no_grad():
            _, self._policy_state = self.policy(input, self._policy_state)
        prediction = self.policy.deterministic_prediction
        flat_action = self.policy.decode(prediction).to("cpu").numpy()
        # Remove temporal axis
        flat_action = flat_action[:, 0, ...]
        log_prob = np.zeros(len(flat_action))
        return flat_action, log_prob
