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

"""Distribution utilities."""

from functools import singledispatch

import torch
from torch import Tensor, distributions
from torch.distributions import Distribution, Transform


__all__ = [
    "is_affine",
    "is_discrete",
    "is_scaling",
    "l2_project",
]


_DISCRETE_DISTRIBUTIONS = (
    distributions.Bernoulli,
    distributions.Binomial,
    distributions.Categorical,
    distributions.Geometric,
    distributions.Multinomial,
    distributions.NegativeBinomial,
    distributions.OneHotCategorical,
    distributions.Poisson,
)


# Adapted from:
# https://github.com/deepmind/trfl/blob/08ccb293edb929d6002786f1c0c177ef291f2956/trfl/distribution_ops.py#L32
def l2_project(z_p: "Tensor", p: "Tensor", z_q: "Tensor") -> "Tensor":
    """Project distribution `(z_p, p)` onto support `z_q` under L2-metric over CDFs.

    Supports `z_p` and `z_q` are specified as tensors of distinct atoms given in ascending order.
    Let `B = {B_1, ..., B_k}` denote the batch shape, `N_p` the number of atoms in `z_p` and `N_q`
    the number of atoms in `z_q`. This projection works for any support `z_q`. In particular, `N_p`
    does not need to be equal to `N_q`.

    Parameters
    ----------
    z_p:
        The support of distribution `p`, shape: ``[*B, N_p]``.
    p:
        The probability distribution `p(z_p)`, shape: ``[*B, N_p]``.
    z_q:
        The support to project onto, shape: ``[*B, N_q]``.

    Returns
    -------
        The projection of `(z_p, p)` onto support `z_q`
        under Cramer distance, shape: ``[*B, N_q]``.

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

    """
    # Reshape
    z_p = z_p[..., None, :]  # [*B, 1, N_p]
    p = p[..., None, :]  # [*B, 1, N_p]
    z_q = z_q[..., None]  # [*B, N_q, 1]

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[..., :1, :], z_q[..., -1:, :]
    d_pos = torch.cat([z_q, vmin], dim=-2)[..., 1:, :]  # [*B, N_q, 1]
    d_neg = torch.cat([vmax, z_q], dim=-2)[..., :-1, :]  # [*B, N_q, 1]

    # Clip z_p to be in the new support range (vmin, vmax)
    z_p = z_p.min(vmax).max(vmin)  # [*B, 1, N_p]

    # Get the distance between atom values in the support
    d_pos = d_pos - z_q  # z_q[i + 1] - z_q[i], [*B, N_q, 1]
    d_neg = z_q - d_neg  # z_q[i] - z_q[i - 1], [*B, N_q, 1]

    # Ensure that we do not divide by zero, in case of atoms of identical value
    d_neg[d_neg > 0] = 1 / d_neg[d_neg > 0]  # [*B, N_q, 1]
    d_pos[d_pos > 0] = 1 / d_pos[d_pos > 0]  # [*B, N_q, 1]

    delta_qp = z_p - z_q  # [*B, N_q, N_p]
    d_sign = (delta_qp >= 0).type(p.type())  # [*B, N_q, N_p]

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i]
    delta_hat = (d_sign * delta_qp * d_pos) - (
        (1 - d_sign) * delta_qp * d_neg
    )  # [*B, N_q, N_p]
    q = ((1 - delta_hat).clamp(0, 1) * p).sum(dim=-1)  # [*B, N_q]
    return q


@singledispatch
def is_discrete(distribution: "Distribution") -> "bool":
    """Check if a distribution is discrete.

    Parameters
    ----------
    distribution:
        The distribution.

    Returns
    -------
        True if the distribution is discrete, False otherwise.

    Notes
    -----
    Register a custom distribution type as follows:
    >>> from torch.distributions import Distribution
    >>>
    >>> from actorch.distributions.utils import is_discrete
    >>>
    >>>
    >>> class Custom(Distribution):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @is_discrete.register(Custom)
    >>> def _is_discrete_custom(distribution):
    >>>     # Implementation
    >>>     ...

    """
    if hasattr(distribution, "is_discrete"):
        return distribution.is_discrete
    if hasattr(distribution, "base_dist"):
        return is_discrete(distribution.base_dist)
    return isinstance(distribution, _DISCRETE_DISTRIBUTIONS)


@singledispatch
def is_affine(transform: "Transform") -> "bool":
    """Check if a transform is affine.

    Parameters
    ----------
    transform:
        The transform.

    Returns
    -------
        True if the transform is affine, False otherwise.

    Notes
    -----
    Register a custom transform type as follows:
    >>> from torch.distributions import Transform
    >>>
    >>> from actorch.distributions.utils import is_affine
    >>>
    >>>
    >>> class Custom(Transform):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @is_affine.register(Custom)
    >>> def _is_affine_custom(transform):
    >>>     # Implementation
    >>>     ...

    """
    if hasattr(transform, "is_constant_jacobian"):
        return transform.is_constant_jacobian
    if isinstance(
        transform, (distributions.CatTransform, distributions.StackTransform)
    ):
        return all(is_affine(t) for t in transform.transforms)
    if isinstance(transform, distributions.ComposeTransform):
        return all(is_affine(t) for t in transform.parts)
    if isinstance(transform, distributions.IndependentTransform):
        return is_affine(transform.base_transform)
    return isinstance(
        transform,
        (distributions.AffineTransform, distributions.ReshapeTransform),
    )


@singledispatch
def is_scaling(transform: "Transform") -> "bool":
    """Check if a transform is a scaling.

    Parameters
    ----------
    transform:
        The transform.

    Returns
    -------
        True if the transform is a scaling, False otherwise.

    Notes
    -----
    Register a custom transform type as follows:
    >>> from torch.distributions import Transform
    >>>
    >>> from actorch.distributions.utils import is_scaling
    >>>
    >>>
    >>> class Custom(Transform):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @is_scaling.register(Custom)
    >>> def _is_scaling_custom(transform):
    >>>     # Implementation
    >>>     ...

    """
    if isinstance(
        transform, (distributions.CatTransform, distributions.StackTransform)
    ):
        return all(is_scaling(t) for t in transform.transforms)
    if isinstance(transform, distributions.ComposeTransform):
        return all(is_scaling(t) for t in transform.parts)
    if isinstance(transform, distributions.IndependentTransform):
        return is_scaling(transform.base_transform)
    return (
        isinstance(transform, distributions.AffineTransform)
        and (torch.as_tensor(transform.loc) == 0).all()
    ) or isinstance(transform, distributions.ReshapeTransform)
