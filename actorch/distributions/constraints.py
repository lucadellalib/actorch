# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Constraints."""

import torch
from torch import Tensor
from torch.distributions import constraints


__all__ = [
    "cat",
    "ordered_real_vector",
    "real_set",
]


class _Cat(constraints.cat):
    """Extended version of `torch.distributions.constraints.cat` that
    implements property `is_discrete` and method `check` correctly.

    """

    # override
    @property
    def is_discrete(self) -> "bool":
        return all(c.is_discrete for c in self.cseq)

    # override
    def check(self, value: "Tensor") -> "Tensor":
        if self.dim < -value.ndim or self.dim >= value.ndim:
            raise IndexError(
                f"`dim` ({self.dim}) must be in the integer interval [-{value.ndim}, {value.ndim})"
            )
        chunks = value.split(self.lengths, dim=self.dim)
        checks = [c.check(chunks[i]) for i, c in enumerate(self.cseq)]
        return torch.stack(checks).all(dim=0)

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__[1:]}"
            f"({self.cseq}, "
            f"dim: {self.dim}, "
            f"lengths: {self.lengths})"
        )


class _OrderedRealVector(constraints.Constraint):
    """Constrain to a real-valued vector whose elements are sorted in (strict) ascending order."""

    event_dim = 1

    # override
    def check(self, value: "Tensor") -> "Tensor":
        return (value[..., 1:] > value[..., :-1]).all(dim=-1)


class _RealSet(constraints.Constraint):
    """Constrain to a set (i.e. no duplicates allowed) of real values."""

    is_discrete = True

    def __init__(self, values: "Tensor") -> "None":
        """Initialize the object.

        Parameters
        ----------
        values:
            The set of real values. All dimensions except for the
            last one are interpreted as batch dimensions.

        Raises
        ------
        ValueError
            If `values` contains duplicates.

        """
        sorted_values = values.sort(dim=-1).values
        if (sorted_values[..., 1:] == sorted_values[..., :-1]).any(dim=-1).any():
            raise ValueError(f"`values` ({values}) must not contain duplicates")
        self._values = values

    # override
    def check(self, value: "Tensor") -> "Tensor":
        # Add event dimension
        value = value[..., None].expand(*value.shape, self._values.shape[-1])
        expanded_support = self._values.expand_as(value)
        return (expanded_support == value).any(dim=-1)

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__[1:]}(values: {self._values})"


# Public interface
cat = _Cat
ordered_real_vector = _OrderedRealVector()
real_set = _RealSet
