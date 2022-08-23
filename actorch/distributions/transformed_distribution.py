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

"""Transformed distribution."""

import torch
from torch import Size, Tensor
from torch.distributions import (
    ComposeTransform,
    Distribution,
    IndependentTransform,
    Transform,
)
from torch.distributions import TransformedDistribution as TorchTransformedDistribution
from torch.distributions import utils
from torch.distributions.constraints import Constraint

from actorch.distributions.registries import reduction_registry  # Avoid circular import
from actorch.distributions.transforms import SumTransform
from actorch.distributions.utils import is_affine, is_discrete


__all__ = [
    "TransformedDistribution",
]


class TransformedDistribution(TorchTransformedDistribution):
    """Extended version of `torch.distributions.TransformedDistribution` that
    implements additional properties and methods (e.g. `mean`,`stddev`, `entropy`,
    etc.) and handles discrete base distributions correctly.

    """

    @property
    def is_discrete(self) -> "bool":
        """Whether the distribution is discrete.

        Returns
        -------
            True if the distribution is discrete, False otherwise.

        """
        if self.reduced_dist is not self:
            return is_discrete(self.reduced_dist)
        return is_discrete(self.base_dist)

    # override
    @property
    def support(self) -> "Constraint":
        if self.reduced_dist is not self:
            return self.reduced_dist.support
        return super().support

    # override
    @property
    def mean(self) -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.mean
        if self._is_affine_or_sum_transform:
            return self._transform_base_mean()
        return super().mean

    # override
    @property
    def mode(self) -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.mode
        if self._is_affine_transform:
            return self._transform_base_mode()
        return super().mode

    # override
    @property
    def variance(self) -> "Tensor":
        return self.stddev**2

    # override
    @property
    def stddev(self) -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.stddev
        if self._is_affine_or_sum_transform:
            return self._transform_base_stddev()
        return super().stddev

    # override
    def sample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        if self.reduced_dist is not self:
            return self.reduced_dist.sample(sample_shape)
        return super().sample(sample_shape)

    # override
    def rsample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        if self.reduced_dist is not self:
            return self.reduced_dist.rsample(sample_shape)
        return super().rsample(sample_shape)

    # override
    def log_prob(self, value: "Tensor") -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.log_prob(value)
        if self.is_discrete and self._is_bijective_transform:
            return self.base_dist.log_prob(self._transform.inv(value.float()))
        return super().log_prob(value)

    # override
    def cdf(self, value: "Tensor") -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.cdf(value)
        return super().cdf(value)

    # override
    def icdf(self, value: "Tensor") -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.icdf(value)
        return super().icdf(value)

    # override
    @property
    def has_enumerate_support(self) -> "bool":
        if self.reduced_dist is not self:
            return self.reduced_dist.has_enumerate_support
        return super().has_enumerate_support

    # override
    def enumerate_support(self, expand: "bool" = True) -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.enumerate_support(expand)
        return super().enumerate_support(expand)

    # override
    def entropy(self) -> "Tensor":
        if self.reduced_dist is not self:
            return self.reduced_dist.entropy()
        if not self._is_bijective_transform:
            return super().entropy()
        if self.is_discrete:
            return self.base_dist.entropy()
        if not self._is_affine_transform:
            return super().entropy()
        return self._transform_base_entropy()

    @property
    def reduced_dist(self) -> "Distribution":
        """Reduce the transformed distribution based on the applied transforms.

        Returns
        -------
            The reduced distribution.

        See Also
        --------
        actorch.distributions.registries.reduction_registry.reduce

        """
        # Contextually check for incompatibilities
        # (e.g. non-maskable batch transforms applied to masked distributions)
        reduced_distribution = reduction_registry.reduce(
            self.base_dist,
            self._transform,
            validate_args=self._validate_args,
        )
        if isinstance(reduced_distribution, TransformedDistribution):
            # Avoid recursion
            return self
        return reduced_distribution

    @property
    def _is_bijective_transform(self) -> "bool":
        return self._transform.bijective

    @property
    def _is_affine_transform(self) -> "bool":
        return is_affine(self._transform)

    @property
    def _is_affine_or_sum_transform(self) -> "bool":
        def check(transform: "Transform") -> "bool":
            if isinstance(transform, ComposeTransform):
                return all(check(p) for p in transform.parts)
            if isinstance(transform, IndependentTransform):
                return check(transform.base_transform)
            return is_affine(transform) or isinstance(transform, SumTransform)

        # Avoid reference cycle
        # (see https://github.com/pytorch/pytorch/blob/c4196bee9324699fa0533b4e4037a9a9894af88d/torch/nn/parallel/scatter_gather.py#L30)
        try:
            result = check(self._transform)
        finally:
            check = None
        return result

    def _transform_base_mean(self) -> "Tensor":
        return self._transform(self.base_dist.mean)

    def _transform_base_mode(self) -> "Tensor":
        return self._transform(self.base_dist.mode)

    def _transform_base_stddev(self) -> "Tensor":
        def apply(transform: "Transform", x: "Tensor") -> "Tensor":
            if isinstance(transform, ComposeTransform):
                for part in transform.parts:
                    x = apply(part, x)
                return x
            if isinstance(transform, IndependentTransform):
                return apply(transform.base_transform, x)
            if is_affine(transform):
                shift = transform(torch.zeros_like(x))
                return transform(x) - shift
            if isinstance(transform, SumTransform):
                return transform(x**2).sqrt()
            raise NotImplementedError

        # Avoid reference cycle
        # (see https://github.com/pytorch/pytorch/blob/c4196bee9324699fa0533b4e4037a9a9894af88d/torch/nn/parallel/scatter_gather.py#L30)
        try:
            result = apply(self._transform, self.base_dist.stddev)
        finally:
            apply = None
        return result

    def _transform_base_entropy(self) -> "Tensor":
        # If Y = f(X) where f is a diffeomorphism and X a continuous
        # random variable, it can be shown that:
        # H[Y] = H[X] + E_X[(log o abs o det o J o f)(X)],
        # where J is the Jacobian matrix
        # (see https://en.wikipedia.org/wiki/Differential_entropy#Properties_of_differential_entropy)
        # If the Jacobian matrix is constant then:
        # E_X[(log o abs o det o J o f)(X)] = (log o abs o det o J o f)(c),
        # where c can be any value in the distribution support
        event_dim = len(self.base_dist.event_shape)
        result = self.base_dist.entropy()
        x = self.base_dist.sample()
        for transform in self.transforms:
            y = transform(x)
            event_dim += transform.codomain.event_dim - transform.domain.event_dim
            result += utils._sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.codomain.event_dim,
            )
            x = y
        return result

    @property
    def _transform(self) -> "Transform":
        if len(self.transforms) > 1:
            return ComposeTransform(self.transforms)
        return self.transforms[0]

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"({self.base_dist}, "
            f"transforms: {self.transforms})"
        )
