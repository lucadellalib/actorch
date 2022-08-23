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

"""Masked distribution."""

from typing import Optional, Union

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.constraints import Constraint

from actorch.distributions.utils import is_discrete


__all__ = [
    "MaskedDistribution",
]


class MaskedDistribution(Distribution):
    """Mask a base distribution along its batch dimensions."""

    arg_constraints = {
        "mask": constraints.boolean,
    }

    # override
    def __init__(
        self,
        base_distribution: "Distribution",
        mask: "Union[bool, Tensor]",
        validate_args: "Optional[bool]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        base_distribution:
            The base distribution to mask.
        mask:
            The boolean scalar/tensor indicating which batch
            elements are valid (True) and which are not (False).
            Must be broadcastable to the base distribution batch shape.
        validate_args:
            True to validate the arguments, False otherwise.
            Default to ``__debug__``.

        """
        mask = torch.as_tensor(mask)
        batch_shape = torch.broadcast_shapes(base_distribution.batch_shape, mask.shape)
        if mask.shape != batch_shape:
            mask = mask.expand(batch_shape)
        if base_distribution.batch_shape != batch_shape:
            base_distribution = base_distribution.expand(batch_shape)
        self.base_dist = base_distribution
        self.mask = mask
        super().__init__(batch_shape, base_distribution.event_shape, validate_args)
        # Ensure mask validation is not bypassed
        self.mask = self.mask.bool()

    # override
    def expand(
        self,
        batch_shape: "Size" = torch.Size(),  # noqa: B008
        _instance: "Optional[MaskedDistribution]" = None,
    ) -> "MaskedDistribution":
        new = self._get_checked_instance(MaskedDistribution, _instance)
        new.base_dist = self.base_dist.expand(batch_shape)
        new.mask = self.mask.expand(batch_shape)
        super(MaskedDistribution, new).__init__(batch_shape, self.event_shape, False)
        new._validate_args = self._validate_args
        return new

    @property
    def is_discrete(self) -> "bool":
        """Whether the distribution is discrete.

        Returns
        -------
            True if the distribution is discrete, False otherwise.

        """
        return is_discrete(self.base_dist)

    # override
    @property
    def support(self) -> "Constraint":
        return self.base_dist.support

    # override
    @property
    def mean(self) -> "Tensor":
        return self.base_dist.mean

    # override
    @property
    def mode(self) -> "Tensor":
        return self.base_dist.mode

    # override
    @property
    def variance(self) -> "Tensor":
        return self.base_dist.variance

    # override
    def sample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        return self.base_dist.sample(sample_shape)

    # override
    def rsample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        return self.base_dist.rsample(sample_shape)

    # override
    def log_prob(self, value: "Tensor") -> "Tensor":
        result = self.base_dist.log_prob(value)
        return result.masked_fill(~self.mask, 0.0)

    # override
    def cdf(self, value: "Tensor") -> "Tensor":
        result = self.base_dist.cdf(value)
        return result.masked_fill(~self.mask, 0.0)

    # override
    @property
    def has_rsample(self) -> "bool":
        return self.base_dist.has_rsample

    # override
    @property
    def has_enumerate_support(self) -> "bool":
        return self.base_dist.has_enumerate_support

    # override
    def enumerate_support(self, expand: "bool" = True) -> "Tensor":
        return self.base_dist.enumerate_support(expand)

    # override
    def entropy(self) -> "Tensor":
        result = self.base_dist.entropy()
        return result.masked_fill(~self.mask, 0.0)

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"({self.base_dist}, "
            f"mask: {self.mask if self.mask.numel() == 1 else self.mask.shape})"
        )
