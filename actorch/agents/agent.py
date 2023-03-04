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

"""Agent."""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from gymnasium.spaces import Space
from numpy import ndarray
from torch import device

from actorch.envs import Flat, Nested
from actorch.networks import PolicyNetwork
from actorch.schedules import Schedule
from actorch.utils import CheckpointableMixin


__all__ = [
    "Agent",
]


class Agent(CheckpointableMixin):
    """Interface between policy network and environment that
    optionally implements an exploration strategy.

    """

    _STATE_VARS = ["policy_network", "_policy_network_state"]  # override

    def __init__(
        self,
        policy_network: "PolicyNetwork",
        observation_space: "Space",
        action_space: "Space",
        clip_action: "bool" = True,
        device: "Union[device, str]" = "cpu",
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

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        self.policy_network = policy_network
        self.observation_space = observation_space
        self.action_space = action_space
        self.clip_action = clip_action
        self.device = device
        self._policy_network_state = None

        # Empirically determine whether observation_space is batched:
        # No errors => observation_space is batched
        # Errors    => observation_space is not batched or observation_space
        #              and policy_network are not compatible
        try:
            self._flat_observation_space = Flat(observation_space, is_batched=True)
            self._batch_size = self._flat_observation_space.shape[0]
            # Remove the potential batch dimension and check
            # if an error is raised in the forward pass
            dummy_flat_observation = self._flat_observation_space.sample()[0]
            dummy_input = torch.as_tensor(dummy_flat_observation)
            with torch.no_grad():
                policy_network.to("cpu").eval()(dummy_input)
        except Exception:
            self._flat_observation_space = Flat(observation_space)
            self._batch_size = None

        # Check compatibility of policy_network, observation_space and action_space
        try:
            self._flat_action_space = Flat(
                action_space, is_batched=self._batch_size is not None
            )
            dummy_flat_observation = self._flat_observation_space.sample()
            dummy_input = torch.as_tensor(dummy_flat_observation)
            with torch.no_grad():
                policy_network.to("cpu").eval()(dummy_input)
                dummy_flat_action = policy_network.predict().to("cpu").numpy()
            if self.clip_action:
                if dummy_flat_action.shape != self._flat_action_space.shape:
                    raise ValueError
                dummy_flat_action = dummy_flat_action.clip(
                    self._flat_action_space.low, self._flat_action_space.high
                )
            if dummy_flat_action not in self._flat_action_space:
                raise ValueError
        except Exception:
            raise ValueError(
                f"`observation_space` ({observation_space}) and `action_space` ({action_space}) "
                f"must be both unbatched or batched with identical batch sizes and `policy_network` "
                f"({policy_network}) must implement a valid mapping between them"
            )

        # If space is already Flat, avoid unnecessary overhead
        if isinstance(observation_space, Flat):
            self._flat_observation_space = observation_space
        if isinstance(action_space, Flat):
            self._flat_action_space = action_space

        self.reset()
        self._schedules: "Dict[str, Schedule]" = {}

    @property
    def schedules(self) -> "Dict[str, Schedule]":
        """Return the agent public schedules.

        Returns
        -------
            The agent public schedules, i.e. a dict that maps
            names of the schedules to the schedules themselves.

        """
        return self._schedules

    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "None":
        """Reset the (possibly batched) agent state.

        In the following, let `B` denote the batch size
        (0 if the agent state is not batched).

        Parameters
        ----------
        mask:
            The boolean array indicating which batch elements are
            to reset (True) and which are not (False), shape: ``[B]``.
            If a scalar or a singleton, it is broadcast accordingly.
            Default to ``np.ones(B, dtype=bool)``.

        Raises
        ------
        ValueError
            If length of `mask` is not equal to the batch size.

        """
        batch_size = self._batch_size or 1
        if mask is None:
            mask = np.ones(batch_size, dtype=bool)
        else:
            mask = np.array(mask, copy=False, ndmin=1)
            if len(mask) == 1:
                mask = np.broadcast_to(mask, (batch_size,))
            if len(mask) != batch_size:
                raise ValueError(
                    f"Length of `mask` ({len(mask)}) must be "
                    f"equal to the batch size ({batch_size})"
                )
        return self._reset(mask)

    def __call__(self, observation: "Nested") -> "Tuple[Nested, ndarray]":
        """Map a (possibly batched) observation to a (possibly batched) action.

        In the following, let `B` denote the batch size (0 if the observation
        is not batched), `O` the shape of a single observation leaf value and
        `A` the shape of a single action leaf value.

        Parameters
        ----------
        observation:
            The (possibly batched) observation, shape of a leaf value: ``[B, *O]``.

        Returns
        -------
            - The (possibly batched) action, shape of a leaf value: ``[B, *A]``;
            - the (possibly batched) log probability, shape: ``[B]``.

        Raises
        ------
        ValueError
            If `observation` is not a sample from the given
            `observation_space` initialization argument.

        """
        flat_observation = (
            self._flat_observation_space.flatten(observation)
            if self.observation_space is not self._flat_observation_space
            else np.asarray(observation)  # Avoid unnecessary overhead
        )
        if flat_observation.shape != self._flat_observation_space.shape:
            raise ValueError(
                f"`observation` ({observation}) must be a sample from the given "
                f"`observation_space` initialization argument ({self.observation_space})"
            )
        self.policy_network.to(self.device, non_blocking=True)
        if self.policy_network.training:
            self.policy_network.eval()
        if not self._batch_size:
            # Add batch axis
            flat_observation = flat_observation[None]
        flat_action, log_prob = self._predict(flat_observation)
        if not self._batch_size:
            # Remove batch axis
            flat_action, log_prob = flat_action[0], log_prob[0]
        if self.clip_action:
            flat_action = flat_action.clip(
                self._flat_action_space.low,
                self._flat_action_space.high,
            )
        action = (
            self._flat_action_space.unflatten(flat_action)
            if self.action_space is not self._flat_action_space
            else flat_action  # Avoid unnecessary overhead
        )
        return action, log_prob

    def _reset(self, mask: "ndarray") -> "None":
        """See documentation of `reset`."""
        if self._policy_network_state is not None:
            # Copy mask to avoid UserWarning: The given NumPy array is not writable
            self._policy_network_state[np.array(mask)] = 0.0

    def _predict(self, flat_observation: "ndarray") -> "Tuple[ndarray, ndarray]":
        """Map a batched flat observation to a batched flat action and its
        corresponding batched log probability.

        In the following, let `B` denote the batch size, `O_f` the size of
        a single flat observation and `A_f` the size of a single flat action.

        Parameters
        ----------
        flat_observation:
            The batched flat observation, shape: ``[B, O_f]``.

        Returns
        -------
            - The batched flat action, shape: ``[B, A_f]``;
            - the batched log probability, shape: ``[B]``.

        """
        # Add time axis
        flat_observation = flat_observation[:, None, ...]
        input = torch.as_tensor(flat_observation, device=self.device)
        with torch.no_grad():
            _, self._policy_network_state = self.policy_network(
                input, self._policy_network_state
            )
            flat_action = self.policy_network.predict().to("cpu").numpy()
        # Remove time axis
        flat_action = flat_action[:, 0, ...]
        log_prob = np.zeros(len(flat_action))
        return flat_action, log_prob

    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(policy_network: {self.policy_network}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"clip_action: {self.clip_action}, "
            f"device: {self.device})"
        )
