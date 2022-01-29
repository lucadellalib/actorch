# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Spaces."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from gym import spaces
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

    def __init__(
        self,
        space: "spaces.Space",
        is_batched: "bool" = False,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        space:
            The base space to flatten.
        is_batched:
            True if `space` is batched, False otherwise.

        """
        # Check that space is batched when is_batched=True
        flatten(space, space.sample(), is_batched, copy=False, validate_args=True)
        self.space = space
        self.is_batched = is_batched
        low, high = get_space_bounds(space)
        low, high = self.flatten(low), self.flatten(high)
        unnested_space = unnest_space(space)
        self._shape_dict, self._type_dict = {}, {}
        for i, (key, space) in enumerate(unnested_space.items()):
            x = np.array(space.sample(), copy=False, ndmin=1)
            self._shape_dict[key] = x.shape
            self._type_dict[key] = type(space)
        super().__init__(low, high, dtype=low.dtype)
        self._cached_flat_unflat = None, None

    @property
    def shape_dict(self) -> "Dict[str, Tuple[int, ...]]":
        return self._shape_dict

    @property
    def type_dict(self) -> "Dict[str, Type[spaces.Space]]":
        return self._type_dict

    # override
    def sample(self) -> "ndarray":
        unflat = self.space.sample()
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
        return unflatten(self.space, x, copy=False)

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

    def __eq__(self, other: "Any") -> "bool":
        return (
            isinstance(other, Flat)
            and self.space == other.space
            and self.is_batched == other.is_batched
        )

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
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
