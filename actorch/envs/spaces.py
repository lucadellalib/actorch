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

"""Spaces."""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from gymnasium import spaces
from gymnasium.utils.seeding import RandomNumberGenerator
from numpy import ndarray

from actorch.envs.utils import (
    Nested,
    batch_space,
    flatten,
    get_log_prob,
    get_space_bounds,
    unflatten,
    unnest_space,
)


__all__ = [
    "Flat",
]


class Flat(spaces.Box):
    """Space wrapper that flattens a base space."""

    # override
    def __init__(
        self,
        space: "spaces.Space",
        is_batched: "bool" = False,
        seed: "Optional[Union[int, RandomNumberGenerator]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        space:
            The base space to flatten.
        is_batched:
            True if `space` is batched, False otherwise.
        seed:
            The seed for generating random numbers.

        Raises
        ------
        ValueError
            If `space` is not batched when `is_batched` is True.

        """
        if is_batched:
            try:
                flatten(
                    space, space.sample(), is_batched, copy=False, validate_args=True
                )
            except Exception:
                raise ValueError(
                    f"`space` ({space}) must be batched when "
                    f"`is_batched` ({is_batched}) is True"
                )
        self.space = space
        self.is_batched = is_batched
        low, high = get_space_bounds(space)
        low, high = self.flatten(low), self.flatten(high)
        self._unnested = unnest_space(space)
        self._cached_flat_unflat = None, None
        super().__init__(low, high, dtype=low.dtype, seed=seed)

    @property
    def np_random(self) -> "RandomNumberGenerator":
        return self.space.np_random

    @property
    def unnested(self) -> "Dict[str, spaces.Space]":
        """Return the unnested base space.

        Returns
        -------
            The unnested base space.

        See Also
        --------
        actorch.envs.utils.unnest_space

        """
        return self._unnested

    # override
    def sample(self, mask: "None" = None) -> "ndarray":
        unflat = self.space.sample(mask)
        flat = self.flatten(unflat)
        self._cached_flat_unflat = flat, unflat
        return flat

    # override
    def seed(self, seed: "Optional[int]" = None) -> "List[int]":
        return self.space.seed(seed)

    # override
    def contains(self, x: "ndarray") -> "bool":
        return self.space.contains(self.unflatten(x))

    # override
    def to_jsonable(
        self,
        sample_n: "Union[Sequence[ndarray], ndarray]",
    ) -> "List":
        return self.space.to_jsonable([self.unflatten(x) for x in sample_n])

    # override
    def from_jsonable(self, sample_n: "List") -> "ndarray":
        return np.asarray([self.flatten(x) for x in self.space.from_jsonable(sample_n)])

    def flatten(self, x: "Nested") -> "ndarray":
        """Flatten a sample from the underlying
        space (without copying if possible).

        Parameters
        ----------
        x:
            The sample.

        Returns
        -------
            The flat sample.

        """
        return flatten(self.space, x, self.is_batched, copy=False)

    def unflatten(self, x: "ndarray") -> "Nested":
        """Unflatten a flat sample from the underlying
        space (without copying if possible).

        Parameters
        ----------
        x:
            The flat sample.

        Returns
        -------
            The sample.

        """
        return unflatten(self.space, x, self.is_batched, copy=False)

    def log_prob(self, x: "ndarray") -> "ndarray":
        """Return the log probability of a flat
        sample from the underlying space.

        Parameters
        ----------
        x:
            The flat sample.

        Returns
        -------
            The log probability.

        """
        flat, unflat = self._cached_flat_unflat
        if x is not flat:
            unflat = self.unflatten(x)
        return get_log_prob(self.space, unflat, self.is_batched)

    # override
    def __eq__(self, other: "Any") -> "bool":
        return (
            isinstance(other, Flat)
            and self.space == other.space
            and self.is_batched == other.is_batched
        )

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(space: {self.space}, "
            f"is_batched: {self.is_batched})"
        )


#####################################################################################################
# Flat batch_space implementation
#####################################################################################################


@batch_space.register(Flat)
def _batch_space_flat(space: "Flat", batch_size: "int" = 1) -> "Flat":
    return Flat(batch_space(space.space, batch_size), is_batched=True)


#####################################################################################################
# Flat get_log_prob implementation
#####################################################################################################


@get_log_prob.register(Flat)
def _get_log_prob_flat(
    space: "Flat",
    x: "ndarray",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    if validate_args:
        if x not in space:
            raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
        if is_batched and not space.is_batched:
            raise ValueError(f"`x` ({x}) and `space` ({space}) must be batched")
    result = space.log_prob(x)
    return result.sum(axis=tuple(range(is_batched, result.ndim)))
