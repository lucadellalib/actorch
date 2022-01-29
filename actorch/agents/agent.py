# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Agent."""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from gym.spaces import Space
from numpy import ndarray
from torch import device

from actorch.envs import Flat, Nested
from actorch.schedules import Schedule


__all__ = [
    "Agent",
]


class Agent(ABC):
    """Interface between policy and environment that
    optionally implements an exploration strategy.

    """

    def __init__(
        self,
        policy: "Policy",
        observation_space: "Space",
        action_space: "Space",
        is_batched: "bool" = False,
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
        device:
            The device.

        Raises
        ------
        ValueError
            If batch sizes of observation and action are not equal.

        """
        self.policy = policy  # Check consitency withpolicy shapes
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_batched = is_batched
        self.device = device
        self._flat_observation_space = Flat(observation_space, is_batched)
        self._flat_action_space = Flat(action_space, is_batched)
        self._batch_size = None
        if is_batched:
            if (
                self._flat_observation_space.shape[0]
                != self._flat_action_space.shape[0]
            ):
                raise ValueError(
                    f"Batch sizes of observation ({self._flat_observation_space.shape[0]}) "
                    f"and action ({self._flat_action_space.shape[0]}) must be equal"
                )
            self._batch_size = self._flat_observation_space.shape[0]
        self.reset()

    @property
    def schedules(self) -> "List[Schedule]":
        """Return the agent public schedules.

        Returns
        -------
            The agent public schedules.

        """
        return []

    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "None":
        """Reset the (possibly batched) agent state.

        In the following, let `B` denote the batch size.

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

        In the following, let `B` denote the batch size, `O` the shape of a single
        observation leaf value and `A` the shape of a single action leaf value.

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
            If batch size of `observation` is not equal to the batch
            size of the given `observation_space` initialization argument.

        """
        flat_observation = self._flat_observation_space.flatten(observation)
        if self._batch_size and len(flat_observation) != self._batch_size:
            raise ValueError(
                f"Batch size of `observation` ({len(flat_observation)}) must "
                f"be equal to the batch size of the given `observation_space` "
                f"initialization argument ({self._batch_size})"
            )
        self.policy.to(self.device)
        if self.policy.training:
            self.policy.eval()
        if not self._batch_size:
            # Add batch axis
            flat_observation = flat_observation[None]
        flat_action, log_prob = self._predict(flat_observation)
        if not self._batch_size:
            # Remove batch axis
            flat_action, log_prob = flat_action[0], log_prob[0]
        action = self._flat_action_space.unflatten(flat_action)
        return action, log_prob

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(policy: {self.policy}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"is_batched: {self.is_batched}, "
            f"device: {self.device})"
        )

    def _reset(self, mask: "ndarray") -> "None":
        """See documentation of method `reset`."""
        pass

    @abstractmethod
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
        raise NotImplementedError
