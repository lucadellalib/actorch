# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Concatenated distribution."""

from typing import List, Optional, Sequence

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, constraints, utils
from torch.distributions.constraints import Constraint

from actorch.distributions.constraints import cat
from actorch.distributions.utils import is_discrete


__all__ = [
    "CatDistribution",
]


class CatDistribution(Distribution):
    """Concatenate a sequence of base distributions with identical
    batch shapes along one of their event dimensions.

    """

    has_enumerate_support = False
    arg_constraints = {}

    def __init__(
        self,
        base_distributions: "Sequence[Distribution]",
        dim: "int" = 0,
        validate_args: "Optional[bool]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        base_distributions:
            The base distributions.
        dim:
            The event dimension along which to concatenate.
        validate_args:
            True to validate the arguments, False otherwise.
            Default to ``__debug__``.

        Raises
        ------
        IndexError
            If `dim` is out of range or not integer.
        ValueError
            If batch shapes of the base distributions are not identical, or the
            corresponding expanded event shapes differ along dimensions other
            than the `dim`-th.

        """
        self.base_dists = base_distributions
        event_ndims = max(len(d.event_shape or [1]) for d in base_distributions)
        self._expanded_event_shapes = [
            torch.Size(list(d.event_shape) + [1] * (event_ndims - len(d.event_shape)))
            for d in base_distributions
        ]
        if dim < -event_ndims or dim >= event_ndims or not float(dim).is_integer():
            raise IndexError(
                f"`dim` ({dim}) must be in the integer interval [-{event_ndims}, {event_ndims})"
            )
        self.dim = dim = int(dim) % event_ndims
        batch_shape = base_distributions[0].batch_shape
        for base_dist in base_distributions[1:]:
            if base_dist.batch_shape != batch_shape:
                raise ValueError(
                    f"Batch shapes of all base distributions "
                    f"({[d.batch_shape for d in base_distributions]}) "
                    f"must be identical"
                )
        event_shape = list(self._expanded_event_shapes[0])
        for expanded_event_shape in self._expanded_event_shapes[1:]:
            if (list(expanded_event_shape[:dim] + expanded_event_shape[dim + 1 :])) != (
                event_shape[:dim] + event_shape[dim + 1 :]
            ):
                raise ValueError(
                    f"Expanded event shapes of all base distributions "
                    f"({self._expanded_event_shapes}) must be identical "
                    f"except for the `dim`-th ({dim}) dimension"
                )
            event_shape[dim] += expanded_event_shape[dim]
        super().__init__(batch_shape, torch.Size(event_shape), validate_args)

    @utils.lazy_property
    def is_discrete(self) -> "bool":
        """Whether the distribution is discrete.

        Returns
        -------
            True if the distribution is discrete, False otherwise.

        """
        return all(is_discrete(d) for d in self.base_dists)

    # override
    @property
    def support(self) -> "Constraint":
        return cat(
            [
                constraints.independent(
                    d.support, len(self.event_shape) - len(d.event_shape)
                )
                for d in self.base_dists
            ],
            dim=self.dim - len(self.event_shape),
            lengths=[shape[self.dim] for shape in self._expanded_event_shapes],
        )

    # override
    @property
    def mean(self) -> "Tensor":
        return self._cat([d.mean for d in self.base_dists])

    # override
    @property
    def variance(self) -> "Tensor":
        return self._cat([d.variance for d in self.base_dists])

    # override
    def sample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        return self._cat([d.sample(sample_shape) for d in self.base_dists])

    # override
    def rsample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        return self._cat([d.rsample(sample_shape) for d in self.base_dists])

    # override
    def log_prob(self, value: "Tensor") -> "Tensor":
        chunks = self._split(value)
        return torch.stack(
            [d.log_prob(c) for d, c in zip(self.base_dists, chunks)]
        ).sum(dim=0)

    # override
    def cdf(self, value: "Tensor") -> "Tensor":
        chunks = self._split(value)
        return torch.stack([d.cdf(c) for d, c in zip(self.base_dists, chunks)]).prod(
            dim=0
        )

    # override
    def has_rsample(self) -> "bool":
        return all(d.has_rsample for d in self.base_dists)

    # override
    def entropy(self) -> "Tensor":
        return torch.stack([d.entropy() for d in self.base_dists]).sum(dim=0)

    def _cat(self, inputs: "Sequence[Tensor]") -> "Tensor":
        inputs = [
            x[(...,) + (None,) * (len(self.event_shape) - len(d.event_shape))]
            for x, d in zip(inputs, self.base_dists)
        ]
        return torch.cat(inputs, dim=self.dim - len(self.event_shape))

    def _split(self, input: "Tensor") -> "List[Tensor]":
        split_sizes = [shape[self.dim] for shape in self._expanded_event_shapes]
        chunks = input.split(split_sizes, dim=self.dim - len(self.event_shape))
        return [
            chunk.reshape(
                *input.shape[: input.ndim - len(self.event_shape)],
                *d.event_shape,
            )
            for chunk, d in zip(chunks, self.base_dists)
        ]

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}({self.base_dists}, dim: {self.dim})"
