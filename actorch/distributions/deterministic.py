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

"""Deterministic distribution."""

from typing import Optional, Union

import torch
from torch import Size, Tensor
from torch.distributions import constraints

from actorch.distributions.finite import Finite


__all__ = [
    "Deterministic",
]


class Deterministic(Finite):
    """Distribution that returns a single deterministic value with probability equal to 1.

    Examples
    --------
    >>> from actorch.distributions import Deterministic
    >>>
    >>>
    >>> value = 1.0
    >>> distribution = Deterministic(value)

    """

    has_rsample = True
    arg_constraints = {
        "value": constraints.real,
    }

    # override
    def __init__(
        self,
        value: "Union[int, float, Tensor]",
        validate_args: "Optional[bool]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        value:
            The deterministic value to return.
        validate_args:
            True to validate the arguments, False otherwise.
            Default to ``__debug__``.

        """
        self.value = torch.as_tensor(value)
        atoms = self.value[..., None]
        probs = torch.ones_like(atoms)
        super().__init__(probs, atoms=atoms, validate_args=validate_args)

    # override
    def expand(
        self,
        batch_shape: "Size" = torch.Size(),  # noqa: B008
        _instance: "Optional[Deterministic]" = None,
    ) -> "Deterministic":
        new = self._get_checked_instance(Deterministic, _instance)
        new.value = self.value.expand(batch_shape)
        atoms = new.value[..., None]
        probs = torch.ones_like(atoms)
        super(Deterministic, new).__init__(probs, atoms=atoms, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # override
    @property
    def mean(self) -> "Tensor":
        return self.value

    # override
    @property
    def mode(self) -> "Tensor":
        return self.value

    # override
    @property
    def variance(self) -> "Tensor":
        return torch.zeros_like(self.value)

    # override
    def sample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        with torch.no_grad():
            return self.rsample(sample_shape)

    # override
    def rsample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        shape = self._extended_shape(sample_shape)
        return self.value.expand(shape)

    # override
    def log_prob(self, value: "Tensor") -> "Tensor":
        if self._validate_args:
            self._validate_sample(value)
        expanded_value = self.value.expand_as(value)
        return (expanded_value == value).type(value.type()).log()

    # override
    def cdf(self, value: "Tensor") -> "Tensor":
        if self._validate_args:
            self._validate_sample(value)
        expanded_value = self.value.expand_as(value)
        return (expanded_value <= value).type(value.type())

    # override
    def enumerate_support(self, expand: "bool" = True) -> "Tensor":
        try:
            return super().enumerate_support(expand)
        except NotImplementedError:
            raise NotImplementedError(
                "`enumerate_support` does not support inhomogeneous values"
            )

    # override
    def entropy(self) -> "Tensor":
        return torch.zeros_like(self.value)

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(value: {self.value if self.value.numel() == 1 else self.value.shape})"
        )
