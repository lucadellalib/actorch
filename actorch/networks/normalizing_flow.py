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

"""Normalizing flow."""

from abc import ABC, abstractmethod
from typing import Any

from torch import Size, Tensor
from torch.distributions import Transform, constraints
from torch.nn import Module


__all__ = [
    "NormalizingFlow",
]


class NormalizingFlow(ABC, Module, Transform):
    """Transform with learnable parameters.

    Useful to learn arbitrarily complex distributions while retaining the ease of reparametrization of simple ones.
    Derived classes must implement `domain`, `codomain`, `forward_shape`, `inverse_shape` and `_call` to allow
    for sampling from the corresponding transformed distribution. If the transform is invertible, `_inverse` and
    `log_abs_det_jacobian` should be implemented to allow for computing the log probability of samples drawn from
    the corresponding transformed distribution. Note that, even when `_inverse` cannot be implemented (e.g. no
    closed-form solution exists), the log probability of samples drawn from the corresponding transformed
    distribution can still be computed by enabling caching.

    Warnings
    --------
    Tto correctly track the device, all tensors involved in the
    computations must be registered as parameters or buffers.

    See Also
    --------
    torch.distributions.transforms.Transform

    References
    ----------
    .. [1] I. Kobyzev, S. Prince, and M. Brubaker.
           "Normalizing Flows: An Introduction and Review of Current Methods".
           In: IEEE Transactions on Pattern Analysis and Machine Intelligence. 2020, pp. 1-1.
           URL: https://arxiv.org/abs/1908.09257
    .. [2] L. Dinh, J. Sohl-Dickstein, and S. Bengio.
           "Density estimation using Real NVP".
           In: ICLR. 2017.
           URL: https://arxiv.org/abs/1605.08803
    .. [3] B. Mazoure, T. Doan, A. Durand, J. Pineau, and R. D. Hjelm.
           "Leveraging exploration in off-policy algorithms via normalizing flows".
           In: Conference on Robot Learning. 2020, pp. 430-444.
           URL: https://arxiv.org/abs/1905.06893
    .. [4] P. Ward, A. Smofsky, and A. Bose.
           "Improving Exploration in Soft-Actor-Critic with Normalizing Flows Policies".
           In: ICML. 2019.
           URL: https://arxiv.org/abs/1906.02771

    """

    is_constant_jacobian = False
    """Whether the Jacobian matrix is constant (i.e. the transform is affine)."""

    # override
    def __init__(self, cache_size: "int" = 0) -> "None":
        """Initialize the object.

        Parameters
        ----------
        cache_size:
            The cache size.

        """
        super(Module, self).__init__(cache_size)
        super().__init__()

    # override
    def forward(
        self, *args: "Any", method: "str" = "_call", **kwargs: "Any"
    ) -> "Tensor":
        if method == "_call":
            return self._call(*args, **kwargs)
        elif method == "_inverse":
            return self._inverse(*args, **kwargs)
        elif method == "log_abs_det_jacobian":
            return self.log_abs_det_jacobian(*args, **kwargs)
        raise NotImplementedError

    @abstractmethod
    def _call(self, x: "Tensor") -> "Tensor":
        """See documentation of `torch.distributions.transforms.Transform._call`."""
        raise NotImplementedError

    def _inverse(self, y: "Tensor") -> "Tensor":
        """See documentation of `torch.distributions.transforms.Transform._inverse`."""
        raise NotImplementedError

    def log_abs_det_jacobian(self, x: "Tensor", y: "Tensor") -> "Tensor":
        """See documentation of `torch.distributions.transforms.Transform.log_abs_det_jacobian`."""
        raise NotImplementedError

    def __hash__(self) -> "int":
        return Module.__hash__(self)

    # override
    @property
    @abstractmethod
    def domain(self) -> "constraints.Constraint":
        raise NotImplementedError

    # override
    @property
    @abstractmethod
    def codomain(self) -> "constraints.Constraint":
        raise NotImplementedError

    # override
    @abstractmethod
    def forward_shape(self, shape: "Size") -> "Size":
        raise NotImplementedError

    # override
    @abstractmethod
    def inverse_shape(self, shape: "Size") -> "Size":
        raise NotImplementedError
