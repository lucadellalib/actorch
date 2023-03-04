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

"""Finite distribution."""

from typing import Optional

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, OneHotCategorical, constraints
from torch.distributions.constraints import Constraint

from actorch.distributions.constraints import ordered_real_vector, real_set


__all__ = [
    "Finite",
]


class Finite(Distribution):
    """Distribution defined over an arbitrary finite support.

    References
    ----------
    .. [1] M. G. Bellemare, W. Dabney, and R. Munos.
           "A Distributional Perspective on Reinforcement Learning".
           In: ICML. 2017, pp. 449-458.
           URL: https://arxiv.org/abs/1707.06887
    .. [2] G. Barth-Maron, M. W. Hoffman, D. Budden, W. Dabney,
           D. Horgan, D. TB, A. Muldal, N. Heess, and T. Lillicrap.
           "Distributed Distributional Deterministic Policy Gradients".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1804.08617

    Examples
    --------
    >>> from actorch.distributions import Finite
    >>>
    >>>
    >>> logits = torch.as_tensor([0.25, 0.15, 0.10, 0.30, 0.20])
    >>> atoms = torch.as_tensor([5.0, 7.5, 10.0, 12.5, 15.0])
    >>> distribution = Finite(logits, atoms=atoms)

    """

    has_enumerate_support = True
    arg_constraints = {
        "probs": constraints.simplex,
        "logits": constraints.real_vector,
        "atoms": ordered_real_vector,
    }
    is_discrete = True
    """Whether the distribution is discrete."""

    # override
    def __init__(
        self,
        probs: "Optional[Tensor]" = None,
        logits: "Optional[Tensor]" = None,
        atoms: "Optional[Tensor]" = None,
        validate_args: "Optional[bool]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        probs:
            The event probabilities.
            Must be None if `logits` is given.
        logits:
            The event log probabilities (unnormalized).
            Must be None if `probs` is given.
        atoms:
            The atoms that form the support of the distribution, sorted in (strict) ascending order.
            Default to `{0, ..., N - 1}` where `N` is ``probs.shape[-1]`` or ``logits.shape[-1]``.
        validate_args:
            True to validate the arguments, False otherwise.
            Default to ``__debug__``.

        """
        param = probs if probs is not None else logits
        if atoms is None:
            atoms = torch.arange(param.shape[-1], device=param.device)
        atoms = atoms.type(param.type())
        param, self.atoms = torch.broadcast_tensors(param, atoms)
        if probs is not None:
            probs = probs.expand_as(param)
        if logits is not None:
            logits = logits.expand_as(param)
        self._one_hot_categorical = OneHotCategorical(probs, logits, validate_args)
        super().__init__(
            self._one_hot_categorical.batch_shape, validate_args=validate_args
        )

    # override
    def expand(
        self,
        batch_shape: "Size" = torch.Size(),  # noqa: B008
        _instance: "Optional[Finite]" = None,
    ) -> "Finite":
        new = self._get_checked_instance(Finite, _instance)
        param_shape = batch_shape + self.atoms.shape[-1:]
        new.atoms = self.atoms.expand(param_shape)
        new._one_hot_categorical = self._one_hot_categorical.expand(batch_shape)
        super(Finite, new).__init__(batch_shape, self.event_shape, False)
        new._validate_args = self._validate_args
        return new

    # override
    @property
    def support(self) -> "Constraint":
        return real_set(self.atoms)

    # override
    @property
    def probs(self) -> "Tensor":
        return self._one_hot_categorical.probs

    # override
    @property
    def logits(self) -> "Tensor":
        return self._one_hot_categorical.logits

    # override
    @property
    def mean(self) -> "Tensor":
        return (self.probs * self.atoms).sum(dim=-1)

    # override
    @property
    def mode(self) -> "Tensor":
        return self.atoms.gather(
            -1,
            self.probs.argmax(dim=-1, keepdim=True),
        )[..., 0]

    # override
    @property
    def variance(self) -> "Tensor":
        return (self.probs * (self.atoms**2)).sum(dim=-1) - self.mean**2

    # override
    def sample(self, sample_shape: "Size" = torch.Size()) -> "Tensor":  # noqa: B008
        one_hot_sample = self._one_hot_categorical.sample(sample_shape)
        return (self.atoms * one_hot_sample).sum(dim=-1)

    # override
    def log_prob(self, value: "Tensor") -> "Tensor":
        if self._validate_args:
            self._validate_sample(value)
        # Add event dimension
        value = value[..., None].expand(*value.shape, self.atoms.shape[-1])
        expanded_atoms = self.atoms.expand_as(value)
        mask = expanded_atoms == value
        result = (self.logits * mask).sum(dim=-1)
        result[mask.sum(dim=-1) == 0] = -float("inf")
        return result

    # override
    def cdf(self, value: "Tensor") -> "Tensor":
        if self._validate_args:
            self._validate_sample(value)
        # Add event dimension
        value = value[..., None].expand(*value.shape, self.atoms.shape[-1])
        expanded_atoms = self.atoms.expand_as(value)
        mask = expanded_atoms <= value
        return (self.probs * mask).sum(dim=-1)

    # override
    def enumerate_support(self, expand: "bool" = True) -> "Tensor":
        if not (
            self.atoms == self.atoms[(slice(1),) * len(self.batch_shape) + (...,)]
        ).all():
            raise NotImplementedError(
                "`enumerate_support` does not support inhomogeneous atoms"
            )
        values = self.atoms.movedim(-1, 0)
        if not expand:
            values = values[
                (...,)
                + (
                    slice(
                        1,
                    ),
                )
                * len(self.batch_shape)
            ]
        return values

    # override
    def entropy(self) -> "Tensor":
        return self._one_hot_categorical.entropy()

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(probs: {self.probs if self.probs.numel() == 1 else self.probs.shape}, "
            f"atoms: {self.atoms if self.atoms.numel() == 1 else self.atoms.shape})"
        )
