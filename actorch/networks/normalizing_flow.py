# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Normalizing flow."""

from abc import ABC, abstractmethod

from torch import Size, Tensor
from torch.distributions import Transform, constraints
from torch.nn import Module


__all__ = [
    "NormalizingFlow",
]


class NormalizingFlow(ABC, Transform, Module):
    """Transform with learnable parameters.

    Useful to learn arbitrarily complex distributions while retaining the ease of reparametrization of simple ones.
    Derived classes must implement `domain`, `codomain`, `forward_shape`, `inverse_shape` and `_call` to allow for
    sampling from the corresponding transformed distribution. If the transform is invertible, `_inverse`, and
    `log_abs_det_jacobian` should be implemented to allow for computing the log probability of samples drawn from
    the corresponding transformed distribution. Note that, even when `_inverse` cannot be implemented (e.g. no
    closed-form solution exists), the log probability of samples drawn from the corresponding transformed
    distribution can still be computed by enabling caching.

    See Also
    --------
    torch.distributions.transforms.Transform

    References
    ----------
    .. [1] I. Kobyzev, S. Prince, and M. Brubaker. "Normalizing Flows: An Introduction
           and Review of Current Methods". In: IEEE Transactions on Pattern Analysis
           and Machine Intelligence. 2020, pp. 1-1.
           URL: https://arxiv.org/abs/1908.09257
    .. [2] L. Dinh, J. Sohl-Dickstein, and S. Bengio. "Density estimation using Real NVP".
           In: ICLR. 2017.
           URL: https://arxiv.org/abs/1605.08803
    .. [3] B. Mazoure, T. Doan, A. Durand, J. Pineau, and R. D. Hjelm. "Leveraging
           exploration in off-policy algorithms via normalizing flows". In: Conference on
           Robot Learning. 2020, pp. 430-444.
           URL: https://arxiv.org/abs/1905.06893
    .. [4] P. Ward, A. Smofsky, and A. Bose. "Improving Exploration in Soft-Actor-Critic
           with Normalizing Flows Policies". In: ICML. 2019.
           URL: https://arxiv.org/abs/1906.02771

    """

    is_constant_jacobian = False
    """Whether the Jacobian matrix is constant (i.e. the transform is affine)."""

    def __hash__(self) -> "int":
        return Module.__hash__(self)

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}()"

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

    # override
    @abstractmethod
    def _call(self, x: "Tensor") -> "Tensor":
        raise NotImplementedError
