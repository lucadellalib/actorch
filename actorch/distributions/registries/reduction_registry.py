# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Reduction registry similar to PyTorch KL registry
(see https://github.com/pytorch/pytorch/blob/a347c747df8302acc0007a26f23ecf3355a5bef9/torch/distributions/kl.py#L36).

"""

from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
from torch import distributions as ds

from actorch.distributions.cat_distribution import CatDistribution
from actorch.distributions.deterministic import Deterministic
from actorch.distributions.finite import Finite
from actorch.distributions.masked_distribution import MaskedDistribution
from actorch.distributions.transformed_distribution import TransformedDistribution
from actorch.distributions.transforms import SumTransform
from actorch.distributions.utils import is_affine, is_scaling
from actorch.utils import is_identical


__all__ = [
    "reduce",
    "register_reduction",
]


_REDUCTION_REGISTRY = defaultdict(list)

_REDUCTION_MEMOIZE = {}


def register_reduction(
    distribution_cls: "Type[ds.Distribution]",
    transform_cls: "Type[ds.Transform]",
    condition_fn: "Optional[Callable[[ds.Distribution, ds.Transform], bool]]" = None,
) -> "Callable":
    """Decorator to register a reduction function that maps a distribution-transform
    pair to its corresponding reduced distribution. Optionally, a condition function
    can be provided, that checks whether the reduction function is applicable to the
    given distribution-transform pair (useful, for example, to avoid ambiguities).

    Parameters
    ----------
    distribution_cls:
        The distribution class to register.
    transform_cls:
        The transform class to register.
    condition_fn:
        The function invoked before applying the reduction function.
        This function receives as arguments the distribution and the transform
        and returns True if the reduction function is applicable, False otherwise.

    Returns
    -------
        The decorated reduction function.

    Raises
    ------
    ValueError
        If an invalid argument value is provided.

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

    def register_fn(
        reduction_fn: "Callable[..., ds.Distribution]",
    ) -> "Callable[..., ds.Distribution]":
        _REDUCTION_REGISTRY[distribution_cls, transform_cls].append(
            (condition_fn, reduction_fn)
        )
        _REDUCTION_MEMOIZE.clear()  # Reset since lookup order may have changed
        return reduction_fn

    return register_fn


def reduce(
    distribution: "ds.Distribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    """Reduce a distribution based on the applied transform.

    For example, if the distribution is a `Normal(mu, sigma)` and the applied transform
    is an `AffineTransform(loc, scale)`, then the resulting reduced distribution is a
    `Normal(loc + scale * mu, scale * sigma)`.

    Returns
    -------
        The reduced distribution.

    Raises
    ------
    NotImplementedError
        If no applicable reduction function exists.

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
) -> "List[Tuple[Callable, Callable]]":
    matches = [
        ds.kl._Match(d, t)
        for d, t in _REDUCTION_REGISTRY
        if issubclass(distribution_cls, d) and issubclass(transform_cls, t)
    ]
    reductions = []
    for match in sorted(matches):
        reductions += _REDUCTION_REGISTRY[match.types]
    return reductions


def _check_cat_cat(
    cat_distribution: "CatDistribution", cat_transform: "ds.CatTransform"
) -> "bool":
    return (
        cat_distribution.dim == cat_transform.dim
        and len(cat_distribution.base_dists) == len(cat_transform.transforms)
        and all(
            (
                d.event_shape[cat_distribution.dim]
                if -len(d.event_shape) <= cat_distribution.dim < len(d.event_shape)
                else 1
            )
            == l
            for d, l in zip(cat_distribution.base_dists, cat_transform.lengths)
        )
    )


#####################################################################################################
# Distributions whose support is finite
#####################################################################################################


@register_reduction(
    ds.Binomial,
    ds.Transform,
    lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0,
)
@register_reduction(
    ds.Bernoulli,
    ds.Transform,
    lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0,
)
@register_reduction(
    ds.Categorical,
    ds.Transform,
    lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0,
)
def _reduction_finite_support_univariate(
    distribution: "Union[ds.Binomial, ds.Bernoulli, ds.Categorical]",
    transform: "ds.Transform",
    **kwargs: "Any",
) -> "Finite":
    support = distribution.enumerate_support()
    logits = distribution.log_prob(support).movedim(-1, 0)
    atoms = transform(support.movedim(-1, 0))
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return Finite(logits=logits, atoms=atoms, validate_args=validate_args)


@register_reduction(
    Finite, ds.Transform, lambda d, t: t.domain.event_dim == t.codomain.event_dim == 0
)
def _reduction_finite_univariate(
    distribution: "Finite", transform: "ds.Transform", **kwargs: "Any"
) -> "Finite":
    atoms = transform(distribution.atoms)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return Finite(logits=distribution.logits, atoms=atoms, validate_args=validate_args)


@register_reduction(Deterministic, ds.Transform)
def _reduction_deterministic_transform(
    distribution: "Deterministic", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[Deterministic, MaskedDistribution]":
    value = transform(distribution.value)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = Deterministic(value, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


#####################################################################################################
# Distributions whose probability density/mass function is closed under affine transforms
#####################################################################################################


@register_reduction(ds.Cauchy, ds.Transform, lambda d, t: is_affine(t))
@register_reduction(ds.Gumbel, ds.Transform, lambda d, t: is_affine(t))
@register_reduction(ds.Laplace, ds.Transform, lambda d, t: is_affine(t))
@register_reduction(ds.Normal, ds.Transform, lambda d, t: is_affine(t))
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
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.StudentT, ds.Transform, lambda d, t: is_affine(t))
def _reduction_student_t_affine(
    distribution: "ds.StudentT", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.StudentT, MaskedDistribution]":
    loc = transform(distribution.loc)
    shift = transform(torch.zeros_like(distribution.scale))
    scale = transform(distribution.scale) - shift
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.StudentT(distribution.df, loc, scale, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.Uniform, ds.Transform, lambda d, t: is_affine(t))
def _reduction_uniform_affine(
    distribution: "ds.Uniform", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Uniform, MaskedDistribution]":
    low = transform(distribution.low)
    high = transform(distribution.high)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.Uniform(low, high, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
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
    base_distribution = ds.Exponential(rate, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.Gamma, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_gamma_scaling(
    distribution: "ds.Gamma", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Gamma, MaskedDistribution]":
    rate = 1 / transform(1 / distribution.rate)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.Gamma(distribution.concentration, rate, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
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
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.Pareto, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_pareto_scaling(
    distribution: "ds.Pareto", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Pareto, MaskedDistribution]":
    scale = transform(distribution.scale)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.Pareto(scale, distribution.alpha, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.Weibull, ds.Transform, lambda d, t: is_scaling(t))
def _reduction_weibull_scaling(
    distribution: "ds.Weibull", transform: "ds.Transform", **kwargs: "Any"
) -> "Union[ds.Weibull, MaskedDistribution]":
    scale = transform(distribution.scale)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.Weibull(scale, distribution.concentration, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


#####################################################################################################
# Sum transform (see https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions)
#####################################################################################################


@register_reduction(
    ds.Bernoulli,
    SumTransform,
    lambda d, t: is_identical(d.logits, t.dim - t.domain.event_dim),
)
@register_reduction(
    ds.Geometric,
    SumTransform,
    lambda d, t: is_identical(d.logits, t.dim - t.domain.event_dim),
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
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(
    ds.Binomial,
    SumTransform,
    lambda d, t: is_identical(d.logits, t.dim - t.domain.event_dim),
)
@register_reduction(
    ds.NegativeBinomial,
    SumTransform,
    lambda d, t: is_identical(d.logits, t.dim - t.domain.event_dim),
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
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.Poisson, SumTransform)
def _reduction_poisson_sum(
    distribution: "ds.Poisson", transform: "SumTransform", **kwargs: "Any"
) -> "Union[ds.Poisson, MaskedDistribution]":
    rate = transform(distribution.rate)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.Poisson(rate, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.Cauchy, SumTransform)
@register_reduction(ds.Normal, SumTransform)
def _reduction_cauchy_normal_sum(
    distribution: "Union[ds.Cauchy, ds.Normal]",
    transform: "SumTransform",
    **kwargs: "Any",
) -> "Union[ds.Cauchy, ds.Normal, MaskedDistribution]":
    loc = transform(distribution.loc)
    scale = transform(distribution.scale ** 2).sqrt()
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = type(distribution)(loc, scale, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(
    ds.Gamma,
    SumTransform,
    lambda d, t: is_identical(d.rate, t.dim - t.domain.event_dim),
)
def _reduction_gamma_sum(
    distribution: "ds.Gamma", transform: "SumTransform", **kwargs: "Any"
) -> "Union[ds.Gamma, MaskedDistribution]":
    concentration = transform(distribution.concentration)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.Gamma(concentration, distribution.rate, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
    )


@register_reduction(ds.Chi2, SumTransform)
def _reduction_chi2_sum(
    distribution: "ds.Chi2", transform: "SumTransform", **kwargs: "Any"
) -> "Union[ds.Chi2, MaskedDistribution]":
    df = transform(distribution.df)
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    base_distribution = ds.Chi2(df, validate_args)
    mask = getattr(transform, "transformed_mask", torch.as_tensor(True))
    return (
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
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
        MaskedDistribution(base_distribution, mask, validate_args)
        if not mask.all()
        else base_distribution
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
        ds.Independent(base_distribution, ndims, validate_args)
        if ndims > 0
        else base_distribution
    )


@register_reduction(CatDistribution, ds.CatTransform, _check_cat_cat)
def _reduction_cat_cat(
    distribution: "CatDistribution", transform: "ds.CatTransform", **kwargs: "Any"
) -> "CatDistribution":
    base_distributions = [
        reduce(d, t, **kwargs)
        for d, t in zip(distribution.base_dists, transform.transforms)
    ]
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return CatDistribution(base_distributions, distribution.dim, validate_args)


@register_reduction(ds.Distribution, ds.CatTransform, lambda d, t: len(t.tseq) == 1)
def _reduction_distribution_cat(
    distribution: "ds.Distribution", transform: "ds.CatTransform", **kwargs: "Any"
) -> "ds.Distribution":
    return reduce(distribution, transform.tseq[0], **kwargs)


@register_reduction(CatDistribution, ds.Transform, lambda d, t: len(d.base_dists) == 1)
def _reduction_cat_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    return reduce(distribution.base_dists[0], transform, **kwargs)


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
    return MaskedDistribution(base_distribution, distribution.mask, validate_args)


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


# Avoid matching subclasses of `distributions.TransformedDistribution` such as `Gumbel`
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
    for i, part in enumerate(transform.parts):
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
        MaskedDistribution(distribution, mask, validate_args)
        if not mask.all()
        else distribution
    )
    is_identity_transform = (
        isinstance(transform, ds.ComposeTransform) and not transform.parts
    )
    return (
        base_distribution
        if is_identity_transform
        else TransformedDistribution(base_distribution, transform, validate_args)
    )
