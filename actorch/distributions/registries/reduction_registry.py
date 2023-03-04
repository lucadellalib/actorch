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

"""Reduction registry similar to PyTorch KL registry
(see https://github.com/pytorch/pytorch/blob/a347c747df8302acc0007a26f23ecf3355a5bef9/torch/distributions/kl.py#L36).

"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch import distributions as ds

from actorch.distributions.deterministic import Deterministic
from actorch.distributions.finite import Finite
from actorch.distributions.masked_distribution import MaskedDistribution
from actorch.distributions.transformed_distribution import TransformedDistribution
from actorch.distributions.transforms import SumTransform
from actorch.distributions.utils import is_affine, is_scaling


__all__ = [
    "reduce",
    "register_reduction",
]


_REDUCTION_REGISTRY = defaultdict(list)

_REDUCTION_MEMOIZE: "Dict[Tuple[Type[ds.Distribution], Type[ds.Transform]], List[Tuple[Optional[Callable[[ds.Distribution, ds.Transform], bool]], Callable[..., ds.Distribution]]]]" = (
    {}
)


def register_reduction(
    distribution_cls: "Type[ds.Distribution]",
    transform_cls: "Type[ds.Transform]",
    condition_fn: "Optional[Callable[[ds.Distribution, ds.Transform], bool]]" = None,
) -> "Callable[[Callable[..., ds.Distribution]], Callable[..., ds.Distribution]]":
    """Decorator that registers a reduction function that maps a distribution-transform
    pair to its corresponding reduced distribution. Optionally, a condition function
    can be given, that checks whether the reduction function is applicable to the
    distribution-transform pair (useful, for example, to avoid ambiguities).

    Parameters
    ----------
    distribution_cls:
        The distribution class to register.
    transform_cls:
        The transform class to register.
    condition_fn:
        The function invoked before applying the reduction function.
        It receives as arguments the distribution and the transform and
        returns True if the reduction function is applicable, False otherwise.

    Returns
    -------
        The decorated reduction function.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    """
    if (not isinstance(distribution_cls, type)) or (
        not issubclass(distribution_cls, ds.Distribution)
    ):
        raise ValueError(
            f"`distribution_cls` ({distribution_cls}) must be a subclass of `torch.distributions.Distribution`"
        )
    if (not isinstance(transform_cls, type)) or (
        not issubclass(transform_cls, ds.Transform)
    ):
        raise ValueError(
            f"`transform_cls` ({transform_cls}) must be a subclass of `torch.distributions.Transform`"
        )

    def register(
        reduction_fn: "Callable[..., ds.Distribution]",
    ) -> "Callable[..., ds.Distribution]":
        _REDUCTION_REGISTRY[distribution_cls, transform_cls].append(
            (condition_fn, reduction_fn)
        )
        _REDUCTION_MEMOIZE.clear()  # Reset since lookup order may have changed
        return reduction_fn

    return register


def reduce(
    distribution: "ds.Distribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    """Reduce a distribution based on the applied transform (find the most specific
    approximate match, assuming single inheritance).

    For example, if the distribution is a ``Normal(mu, sigma)`` and the applied transform
    is an ``AffineTransform(loc, scale)``, then the resulting reduced distribution is a
    ``Normal(loc + scale * mu, scale * sigma)``.

    Returns
    -------
        The reduced distribution.

    Raises
    ------
    NotImplementedError
        If no applicable reduction function exists.

    Warnings
    --------
    Reducing an `actorch.distributions.CatDistribution` whose base distributions
    are (possibly wrapped) `actorch.distributions.MaskedDistribution` might lead
    to incorrect results.

    """
    try:
        reductions = _REDUCTION_MEMOIZE[type(distribution), type(transform)]
    except KeyError:
        reductions = _dispatch_reduction(type(distribution), type(transform))
        _REDUCTION_MEMOIZE[type(distribution), type(transform)] = reductions
    for condition_fn, reduction_fn in reductions:
        if (not condition_fn) or condition_fn(distribution, transform):
            return reduction_fn(distribution, transform, **kwargs)
    raise NotImplementedError


def _dispatch_reduction(
    distribution_cls: "Type[ds.Distribution]",
    transform_cls: "Type[ds.Transform]",
) -> "List[Tuple[Optional[Callable[[ds.Distribution, ds.Transform], bool]], Callable[..., ds.Distribution]]]":
    matches = [
        ds.kl._Match(d, t)
        for d, t in _REDUCTION_REGISTRY
        if issubclass(distribution_cls, d) and issubclass(transform_cls, t)
    ]
    reductions = []
    for match in sorted(matches):
        reductions += _REDUCTION_REGISTRY[match.types]
    return reductions


def _is_degenerate_affine(
    transform: "ds.Transform",
    shape: "Tuple[int, ...]",
) -> "bool":
    if not is_affine(transform):
        return False
    # TODO: find a more robust solution
    try:
        shift = transform(torch.zeros(shape))
        scale = transform(torch.ones(shape)) - shift
    except RuntimeError:
        shift = transform(torch.zeros(shape, device="cuda:0"))
        scale = transform(torch.ones(shape, device="cuda:0")) - shift
    return (scale == 0.0).all()


def _is_identical(input: "Tensor", dim: "int" = 0) -> "bool":
    return (input == input.select(dim, 0).unsqueeze(dim)).all(dim=dim).all()


#####################################################################################################
# Degenerate affine transform
#####################################################################################################


@register_reduction(
    ds.Distribution,
    ds.Transform,
    lambda d, t: _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
def _reduction_distribution_degenerate_affine(
    distribution: "ds.Distribution", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[Deterministic, MaskedDistribution]":
    shift = transform(torch.zeros_like(distribution.sample()))
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = Deterministic(shift, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


#####################################################################################################
# Distributions whose support is finite
#####################################################################################################


@register_reduction(
    ds.Bernoulli,
    ds.Transform,
    lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0,
)
@register_reduction(
    ds.Binomial,
    ds.Transform,
    lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0,
)
@register_reduction(
    ds.Categorical,
    ds.Transform,
    lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0,
)
def _reduction_finite_support_univariate(
    distribution: "Union[ds.Bernoulli, ds.Binomial, ds.Categorical]",
    transform: "ds.Transform",
    **kwargs: "Any",
) -> "Finite":
    support = distribution.enumerate_support()
    logits = distribution.log_prob(support).movedim(0, -1)
    if isinstance(transform, ds.AffineTransform):
        transform.loc = torch.as_tensor(
            transform.loc, device=getattr(transform.scale, "device", None)
        )
        transform.scale = torch.as_tensor(
            transform.scale, device=getattr(transform.loc, "device", None)
        )
        # Expand right
        transform.loc = transform.loc[
            (...,) + (None,) * (logits.ndim - transform.loc.ndim)
        ].expand_as(logits)
        transform.scale = transform.scale[
            (...,) + (None,) * (logits.ndim - transform.scale.ndim)
        ].expand_as(logits)
    atoms = transform(support.movedim(0, -1))
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return Finite(logits=logits, atoms=atoms, validate_args=validate_args)


@register_reduction(
    Finite, ds.Transform, lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0
)
def _reduction_finite_univariate(
    distribution: "Finite", transform: "ds.Transform", **kwargs: "Any"
) -> "Finite":
    logits = distribution.logits
    if isinstance(transform, ds.AffineTransform):
        transform.loc = torch.as_tensor(
            transform.loc, device=getattr(transform.scale, "device", None)
        )
        transform.scale = torch.as_tensor(
            transform.scale, device=getattr(transform.loc, "device", None)
        )
        # Expand right
        transform.loc = transform.loc[
            (...,) + (None,) * (logits.ndim - transform.loc.ndim)
        ].expand_as(logits)
        transform.scale = transform.scale[
            (...,) + (None,) * (logits.ndim - transform.scale.ndim)
        ].expand_as(logits)
    atoms = transform(distribution.atoms)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return type(distribution)(logits=logits, atoms=atoms, validate_args=validate_args)


@register_reduction(Deterministic, ds.Transform)
def _reduction_deterministic_transform(
    distribution: "Deterministic", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[Deterministic, MaskedDistribution]":
    value = transform(distribution.value)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(value, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


#####################################################################################################
# Distributions whose probability density/mass function is closed under affine transforms
#####################################################################################################


@register_reduction(
    ds.Cauchy,
    ds.Transform,
    lambda d, t: is_affine(t)
    and not _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
@register_reduction(
    ds.Gumbel,
    ds.Transform,
    lambda d, t: is_affine(t)
    and not _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
@register_reduction(
    ds.Laplace,
    ds.Transform,
    lambda d, t: is_affine(t)
    and not _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
@register_reduction(
    ds.Normal,
    ds.Transform,
    lambda d, t: is_affine(t)
    and not _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
def _reduction_loc_scale_affine(
    distribution: "Union[ds.Cauchy, ds.Gumbel, ds.Laplace, ds.Normal]",
    transform: "ds.Transform",
    **kwargs: "Any",
) -> "Union[ds.Cauchy, ds.Gumbel, ds.Laplace, ds.Normal, MaskedDistribution]":
    loc = transform(distribution.loc)
    shift = transform(torch.zeros_like(distribution.scale))
    scale = transform(distribution.scale) - shift
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(loc, scale, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(
    ds.MixtureSameFamily,
    ds.Transform,
    lambda d, t: is_affine(t)
    and not _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
def _reduction_mixture_same_family_affine(
    distribution: "ds.MixtureSameFamily",
    transform: "ds.Transform",
    **kwargs: "Any",
) -> "Union[ds.MixtureSameFamily, MaskedDistribution]":
    mixture_distribution = distribution.mixture_distribution
    logits = mixture_distribution.logits
    if isinstance(transform, ds.AffineTransform):
        transform.loc = torch.as_tensor(
            transform.loc, device=getattr(transform.scale, "device", None)
        )
        transform.scale = torch.as_tensor(
            transform.scale, device=getattr(transform.loc, "device", None)
        )
        # Expand right
        transform.loc = transform.loc[
            (...,) + (None,) * (logits.ndim - transform.loc.ndim)
        ].expand_as(logits)
        transform.scale = transform.scale[
            (...,) + (None,) * (logits.ndim - transform.scale.ndim)
        ].expand_as(logits)
    component_distribution = reduce(
        distribution.component_distribution, transform, **kwargs
    )
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(
        mixture_distribution, component_distribution, validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.MultivariateNormal, ds.AffineTransform)
@register_reduction(ds.LowRankMultivariateNormal, ds.AffineTransform)
def _reduction_multivariate_normal_affine_loc_scale(
    distribution: "ds.MultivariateNormal",
    transform: "ds.AffineTransform",
    **kwargs: "Any",
) -> "Union[ds.MultivariateNormal, MaskedDistribution]":
    loc = transform(distribution.loc)
    covariance_matrix = (
        distribution.covariance_matrix
        * transform.scale[..., None].expand_as(distribution.covariance_matrix) ** 2
    )
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.MultivariateNormal(
        loc, covariance_matrix, validate_args=validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(
    ds.StudentT,
    ds.Transform,
    lambda d, t: is_affine(t)
    and not _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
def _reduction_student_t_affine(
    distribution: "ds.StudentT", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.StudentT, MaskedDistribution]":
    loc = transform(distribution.loc)
    shift = transform(torch.zeros_like(distribution.scale))
    scale = transform(distribution.scale) - shift
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(distribution.df, loc, scale, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(
    ds.Uniform,
    ds.Transform,
    lambda d, t: is_affine(t)
    and not _is_degenerate_affine(t, d.batch_shape + d.event_shape),
)
def _reduction_uniform_affine(
    distribution: "ds.Uniform", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Uniform, MaskedDistribution]":
    low = transform(distribution.low)
    high = transform(distribution.high)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(low, high, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


#####################################################################################################
# Distributions whose probability density/mass function is closed under scaling
#####################################################################################################


@register_reduction(ds.Exponential, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_exponential_scaling(
    distribution: "ds.Exponential", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Exponential, MaskedDistribution]":
    rate = 1 / transform(1 / distribution.rate)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(rate, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.Gamma, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_gamma_scaling(
    distribution: "ds.Gamma", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Gamma, MaskedDistribution]":
    rate = 1 / transform(1 / distribution.rate)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(
        distribution.concentration, rate, validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.HalfCauchy, ds.Transform, lambda d, t: is_scaling(t))
@register_reduction(ds.HalfNormal, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_half_cauchy_half_normal_scaling(
    distribution: "Union[ds.HalfCauchy, ds.HalfNormal]",
    transform: "ds.Transform",
    **kwargs: "Any",
) -> "Union[ds.HalfCauchy, ds.HalfNormal, MaskedDistribution]":
    scale = transform(distribution.scale)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(scale, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.Pareto, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_pareto_scaling(
    distribution: "ds.Pareto", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Pareto, MaskedDistribution]":
    scale = transform(distribution.scale)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(scale, distribution.alpha, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.Weibull, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_weibull_scaling(
    distribution: "ds.Weibull", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Weibull, MaskedDistribution]":
    scale = transform(distribution.scale)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(
        scale, distribution.concentration, validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


#####################################################################################################
# Sum transform (see https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions)
#####################################################################################################


@register_reduction(
    ds.Bernoulli,
    SumTransform,
    lambda d, t: _is_identical(d.logits, t.dim - t.domain.event_dim),
)
@register_reduction(
    ds.Geometric,
    SumTransform,
    lambda d, t: _is_identical(d.logits, t.dim - t.domain.event_dim),
)
def _reduction_bernoulli_geometric_sum(
    distribution: "Union[ds.Bernoulli, ds.Geometric]",
    transform: "SumTransform",
    **kwargs: "Any",
) -> "Union[ds.Binomial, ds.NegativeBinomial, MaskedDistribution]":
    total_count = distribution.logits.shape[transform.dim - transform.domain.event_dim]
    logits = distribution.logits.select(transform.dim - transform.domain.event_dim, 0)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    distribution_cls = (
        ds.Binomial if isinstance(distribution, ds.Bernoulli) else ds.NegativeBinomial
    )
    base_distribution = distribution_cls(
        total_count, logits=logits, validate_args=validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(
    ds.Binomial,
    SumTransform,
    lambda d, t: _is_identical(d.logits, t.dim - t.domain.event_dim),
)
@register_reduction(
    ds.NegativeBinomial,
    SumTransform,
    lambda d, t: _is_identical(d.logits, t.dim - t.domain.event_dim),
)
def _reduction_binomial_negative_binomial_sum(
    distribution: "Union[ds.Binomial, ds.NegativeBinomial]",
    transform: "SumTransform",
    **kwargs: "Any",
) -> "Union[ds.Binomial, ds.NegativeBinomial, MaskedDistribution]":
    total_count = transform(distribution.total_count)
    logits = distribution.logits.select(transform.dim - transform.domain.event_dim, 0)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(
        total_count, logits=logits, validate_args=validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.Poisson, SumTransform)
def _reduction_poisson_sum(
    distribution: "ds.Poisson", transform: "SumTransform", **kwargs: "Any"
) -> "Union[ds.Poisson, MaskedDistribution]":
    rate = transform(distribution.rate)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(rate, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.Cauchy, SumTransform)
@register_reduction(ds.Normal, SumTransform)
def _reduction_cauchy_normal_sum(
    distribution: "Union[ds.Cauchy, ds.Normal]",
    transform: "SumTransform",
    **kwargs: "Any",
) -> "Union[ds.Cauchy, ds.Normal, MaskedDistribution]":
    loc = transform(distribution.loc)
    scale = transform(distribution.scale**2).sqrt()
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(loc, scale, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(
    ds.Gamma,
    SumTransform,
    lambda d, t: _is_identical(d.rate, t.dim - t.domain.event_dim),
)
def _reduction_gamma_sum(
    distribution: "ds.Gamma", transform: "SumTransform", **kwargs: "Any"
) -> "Union[ds.Gamma, MaskedDistribution]":
    concentration = transform(distribution.concentration)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(
        concentration, distribution.rate, validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


@register_reduction(ds.Chi2, SumTransform)
def _reduction_chi2_sum(
    distribution: "ds.Chi2", transform: "SumTransform", **kwargs: "Any"
) -> "Union[ds.Chi2, MaskedDistribution]":
    df = transform(distribution.df)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(df, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


#####################################################################################################
# Special transforms
#####################################################################################################


@register_reduction(ds.Cauchy, ds.AbsTransform, lambda d, t: (d.loc == 0).all())
@register_reduction(ds.Normal, ds.AbsTransform, lambda d, t: (d.loc == 0).all())
def _reduction_cauchy_normal_abs(
    distribution: "Union[ds.Cauchy, ds.Normal]",
    transform: "ds.AbsTransform",
    **kwargs: "Any",
) -> "Union[ds.HalfCauchy, ds.HalfNormal]":
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    distribution_cls = (
        ds.HalfCauchy if isinstance(distribution, ds.Cauchy) else ds.HalfNormal
    )
    return distribution_cls(distribution.scale, validate_args)


@register_reduction(ds.Normal, ds.ExpTransform)
def _reduction_normal_exp(
    distribution: "ds.Normal", transform: "ds.ExpTransform", **kwargs: "Any"
) -> "ds.LogNormal":
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return ds.LogNormal(distribution.loc, distribution.scale, validate_args)


@register_reduction(ds.Normal, ds.StickBreakingTransform)
def _reduction_normal_stick_breaking(
    distribution: "ds.Normal", transform: "ds.StickBreakingTransform", **kwargs: "Any"
) -> "Union[ds.LogisticNormal, MaskedDistribution]":
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.LogisticNormal(
        distribution.loc, distribution.scale, validate_args
    )
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args)
    )


#####################################################################################################
# Composite distribution/transform
#####################################################################################################


@register_reduction(ds.Independent, ds.Transform)
def _reduction_independent_transform(
    distribution: "ds.Independent", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    base_distribution = reduce(distribution.base_dist, transform, **kwargs)
    ndims = len(distribution.event_shape) - transform.domain.event_dim
    ndims += transform.codomain.event_dim - len(base_distribution.event_shape)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return (
        type(distribution)(base_distribution, ndims, validate_args)
        if ndims > 0
        else base_distribution
    )


@register_reduction(
    ds.Distribution, ds.CatTransform, lambda d, t: len(t.transforms) == 1
)
def _reduction_distribution_cat(
    distribution: "ds.Distribution", transform: "ds.CatTransform", **kwargs: "Any"
) -> "ds.Distribution":
    return reduce(distribution, transform.transforms[0], **kwargs)


@register_reduction(
    MaskedDistribution,
    ds.Transform,
    lambda d, t: len(d.event_shape) >= t.domain.event_dim,
)
def _reduction_masked_event_transform(
    distribution: "MaskedDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "MaskedDistribution":
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = reduce(distribution.base_dist, transform, **kwargs)
    return type(distribution)(base_distribution, distribution.mask, validate_args)


@register_reduction(
    MaskedDistribution,
    ds.Transform,
    lambda d, t: len(d.event_shape) < t.domain.event_dim and hasattr(t, "mask"),
)
def _reduction_masked_batch_transform(
    distribution: "MaskedDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    batch_shape, event_shape = distribution.batch_shape, distribution.event_shape
    mask = distribution.mask[(...,) + (None,) * len(event_shape)].expand(
        batch_shape + event_shape
    )
    transform.mask = (
        transform.mask[(...,) + (None,) * (mask.ndim - transform.mask.ndim)].expand_as(
            mask
        )
        & mask
    )
    return reduce(distribution.base_dist, transform, **kwargs)


# Avoid matching subclasses of torch.distributions.TransformedDistribution such as Gumbel
@register_reduction(
    ds.TransformedDistribution,
    ds.Transform,
    lambda d, t: type(d) == ds.TransformedDistribution
    or isinstance(d, TransformedDistribution),
)
def _reduction_transformed_transform(
    distribution: "ds.TransformedDistribution",
    transform: "ds.Transform",
    **kwargs: "Any",
) -> "ds.Distribution":
    base_distribution = reduce(
        distribution.base_dist,
        ds.ComposeTransform(distribution.transforms)
        if len(distribution.transforms) > 1
        else distribution.transforms[0],
        **kwargs,
    )
    is_reducible_distribution = not isinstance(
        base_distribution, TransformedDistribution
    )
    return (
        reduce(base_distribution, transform, **kwargs)
        if is_reducible_distribution
        else _reduction_distribution_transform(base_distribution, transform, **kwargs)
    )


@register_reduction(ds.Distribution, ds.IndependentTransform)
def _reduction_distribution_independent(
    distribution: "ds.Distribution",
    transform: "ds.IndependentTransform",
    **kwargs: "Any",
) -> "ds.Distribution":
    return reduce(distribution, transform.base_transform, **kwargs)


@register_reduction(ds.Distribution, ds.ComposeTransform)
def _reduction_distribution_compose(
    distribution: "ds.Distribution", transform: "ds.ComposeTransform", **kwargs: "Any"
) -> "ds.Distribution":
    for part in transform.parts:
        distribution = reduce(distribution, part, **kwargs)
    return distribution


#####################################################################################################
# Generic distribution/transform
#####################################################################################################


@register_reduction(ds.Distribution, ds.Transform, lambda d, t: not hasattr(d, "mask"))
def _reduction_distribution_transform(
    distribution: "ds.Distribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    mask = getattr(transform, "mask", torch.as_tensor(True))
    base_distribution = (
        distribution
        if mask.all()
        else MaskedDistribution(distribution, mask, validate_args)
    )
    is_identity_transform = (
        isinstance(transform, ds.ComposeTransform) and not transform.parts
    )
    return (
        base_distribution
        if is_identity_transform
        else TransformedDistribution(base_distribution, transform, validate_args)
    )
