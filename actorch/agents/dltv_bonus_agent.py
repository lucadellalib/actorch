# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Decaying left truncated variance bonus agent."""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from gym.spaces import Space
from numpy import ndarray
from torch import Tensor, device
from torch.distributions import Distribution

from actorch.agents.deterministic_agent import DeterministicAgent
from actorch.agents.stochastic_agent import StochasticAgent
from actorch.distributions import Finite
from actorch.registry import register
from actorch.schedules import LambdaSchedule, Schedule
from actorch.utils import is_identical


__all__ = [
    "DLTVBonusAgent",
    "left_truncated_variance_quantile",
]


def left_truncated_variance_quantile(distribution: "Finite") -> "Tensor":
    """Compute the left truncated variance
    of a quantile distribution.

    Parameters
    ----------
    distribution:
        The quantile distribution.

    Returns
    -------
        The left truncated variance.

    Raises
    ------
    ValueError
        If `distribution.probs` is not identical
        along the last dimension.

    References
    ----------
    .. [1] B. Mavrin, S. Zhang, H. Yao, L. Kong, K. Wu, and Y. Yu.
           "Distributional Reinforcement Learning for Efficient Exploration".
           In: ICML. 2019, pp. 4424-4434.
           URL: https://arxiv.org/abs/1905.06125

    """
    if not is_identical(distribution.probs, dim=-1):
        raise ValueError(
            f"`distribution.probs` ({distribution.probs}) "
            f"must be identical along the last dimension"
        )
    num_atoms = distribution.atoms.shape[-1]
    median_idx = num_atoms // 2
    median = distribution.atoms[..., median_idx]
    left_truncated_variance = (
        (distribution.atoms[..., median_idx:] - median) ** 2
    ).sum(dim=-1) / (2 * num_atoms)
    return left_truncated_variance


@register
class DLTVBonusAgent(StochasticAgent, DeterministicAgent):
    """Agent that adds a decaying left truncated variance bonus to the prediction.

    References
    ----------
    .. [1] B. Mavrin, S. Zhang, H. Yao, L. Kong, K. Wu, and Y. Yu.
           "Distributional Reinforcement Learning for Efficient Exploration".
           In: ICML. 2019, pp. 4424-4434.
           URL: https://arxiv.org/abs/1905.06125

    """

    def __init__(
        self,
        policy: "Policy",
        observation_space: "Space",
        action_space: "Space",
        is_batched: "bool" = False,
        suppression_coeff: "Optional[Schedule]" = None,
        left_truncated_variance_fn: "Callable[[Distribution], Tensor]" = None,
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
        suppression_coeff:
            The suppression coefficient schedule
            (`c_t` in the literature).
            Default to ``LambdaSchedule(
                lambda t: 50 * np.sqrt(np.log(t + 2) / (t + 2))
            )``.
        left_truncated_variance_fn:
            The function that computes the left truncated variance
            of a distribution. This function receives as an argument
            the distribution and returns its left truncated variance.
            Default to ``left_truncated_variance_quantile``.
        num_random_timesteps:
            The number of initial timesteps for which
            a random prediction is returned.
        device:
            The device.

        """
        self.suppression_coeff = suppression_coeff or LambdaSchedule(
            lambda t: 50 * np.sqrt(np.log(t + 2) / (t + 2))
        )
        self.left_truncated_variance_fn = (
            left_truncated_variance_fn or left_truncated_variance_quantile
        )
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
    @property
    def schedules(self) -> "List[Schedule]":
        return [self.suppression_coeff]

    # override
    def _stochastic_predict(
        self, flat_observation: "ndarray"
    ) -> "Tuple[ndarray, ndarray]":
        suppression_coeff = self.suppression_coeff()
        if suppression_coeff < 0.0:
            raise ValueError(
                f"`suppression_coeff` ({suppression_coeff}) must be in the interval [0, inf)"
            )
        # Add temporal axis
        flat_observation = flat_observation[:, None, ...]
        input = torch.as_tensor(flat_observation, device=self.device)
        with torch.no_grad():
            _, self._policy_state = self.policy(input, self._policy_state)
        prediction = self.policy.distribution.deterministic_prediction
        left_truncated_variance = self.left_truncated_variance_fn(
            self.policy.distribution
        )
        bonus = suppression_coeff * left_truncated_variance
        flat_action = self.policy.decode(prediction + bonus).to("cpu").numpy()
        # Remove temporal axis
        flat_action = flat_action[:, 0, ...]
        log_prob = np.zeros(len(flat_action))
        return flat_action, log_prob

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(policy: {self.policy}, "
            f"observation_space: {self.observation_space}, "
            f"action_space: {self.action_space}, "
            f"is_batched: {self.is_batched}, "
            f"suppression_coeff: {self.suppression_coeff}, "
            f"left_truncated_variance_fn: {self.left_truncated_variance_fn}, "
            f"device: {self.device})"
        )
