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

"""Concatenated distribution reduction registry."""

from typing import Any

import torch
from torch import distributions as ds

from actorch.distributions.cat_distribution import CatDistribution
from actorch.distributions.deterministic import Deterministic
from actorch.distributions.finite import Finite
from actorch.distributions.masked_distribution import MaskedDistribution
from actorch.distributions.registries.reduction_registry import (
    reduce,
    register_reduction,
)


__all__ = []


def _is_cat_cat(
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


def _has_stackable_logits(cat_distribution: "CatDistribution") -> "bool":
    dim = len(cat_distribution.batch_shape) + cat_distribution.dim + 1
    shape = cat_distribution.base_dists[0].logits.shape
    shape = shape[:dim] + shape[dim + 1 :]
    for base_dist in cat_distribution.base_dists[1:]:
        if base_dist.logits.shape[:dim] + base_dist.logits.shape[dim + 1 :] != shape:
            return False
    return True


@register_reduction(CatDistribution, ds.Transform, lambda d, t: len(d.base_dists) == 1)
def _reduction_cat_singleton_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    return reduce(distribution.base_dists[0], transform, **kwargs)


@register_reduction(CatDistribution, ds.CatTransform, _is_cat_cat)
def _reduction_cat_cat(
    distribution: "CatDistribution", transform: "ds.CatTransform", **kwargs: "Any"
) -> "CatDistribution":
    base_distributions = [
        reduce(d, t, **kwargs)
        for d, t in zip(distribution.base_dists, transform.transforms)
    ]
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    return type(distribution)(base_distributions, distribution.dim, validate_args)


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Bernoulli) for b in d.base_dists)
    and _has_stackable_logits(d),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Categorical) for b in d.base_dists)
    and _has_stackable_logits(d),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.ContinuousBernoulli) for b in d.base_dists)
    and _has_stackable_logits(d),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Geometric) for b in d.base_dists)
    and _has_stackable_logits(d),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.OneHotCategorical) for b in d.base_dists)
    and _has_stackable_logits(d),
)
def _reduction_cat_logits_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    logits = [b.logits for b in distribution.base_dists]
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    logits = torch.stack(logits, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        logits=logits, validate_args=validate_args
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Beta) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Kumaraswamy) for b in d.base_dists),
)
def _reduction_cat_concentration1_concentration0_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    concentration1s, concentration0s = zip(
        *[(b.concentration1, b.concentration0) for b in distribution.base_dists]
    )
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    concentration1 = torch.stack(concentration1s, dim=dim)
    concentration0 = torch.stack(concentration0s, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        concentration1, concentration0, validate_args
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Binomial) for b in d.base_dists)
    and _has_stackable_logits(d),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Multinomial) for b in d.base_dists)
    and _has_stackable_logits(d),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.NegativeBinomial) for b in d.base_dists)
    and _has_stackable_logits(d),
)
def _reduction_cat_total_count_logits_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    total_counts, logits = zip(
        *[(b.total_count, b.logits) for b in distribution.base_dists]
    )
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    total_count = torch.stack(total_counts, dim=dim)
    logits = torch.stack(logits, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        total_count,
        logits=logits,
        validate_args=validate_args,
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Chi2) for b in d.base_dists),
)
def _reduction_cat_chi2_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    dfs = [b.df for b in distribution.base_dists]
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    df = torch.stack(dfs, dim=dim)
    base_distribution = type(distribution.base_dists[0])(df, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Cauchy) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Gumbel) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Laplace) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Normal) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.LogNormal) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.LogisticNormal) for b in d.base_dists),
)
def _reduction_cat_loc_scale_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    locs, scales = zip(*[(b.loc, b.scale) for b in distribution.base_dists])
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    loc = torch.stack(locs, dim=dim)
    scale = torch.stack(scales, dim=dim)
    base_distribution = type(distribution.base_dists[0])(loc, scale, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, Deterministic) for b in d.base_dists),
)
def _reduction_cat_deterministic_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    values = [b.value for b in distribution.base_dists]
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    value = torch.stack(values, dim=dim)
    base_distribution = type(distribution.base_dists[0])(value, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Dirichlet) for b in d.base_dists),
)
def _reduction_cat_dirichlet_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    concentrations = [b.concentration for b in distribution.base_dists]
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    concentration = torch.stack(concentrations, dim=dim)
    base_distribution = type(distribution.base_dists[0])(concentration, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Exponential) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Poisson) for b in d.base_dists),
)
def _reduction_cat_rate_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    rates = [b.rate for b in distribution.base_dists]
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    rate = torch.stack(rates, dim=dim)
    base_distribution = type(distribution.base_dists[0])(rate, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, Finite) for b in d.base_dists)
    and _has_stackable_logits(d),
)
def _reduction_cat_finite_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    logits, atoms = zip(*[(b.logits, b.atoms) for b in distribution.base_dists])
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    logits = torch.stack(logits, dim=dim)
    atoms = torch.stack(atoms, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        logits=logits, atoms=atoms, validate_args=validate_args
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.FisherSnedecor) for b in d.base_dists),
)
def _reduction_cat_fisher_snedecor_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    df1s, df2s = zip(*[(b.df1, b.df2) for b in distribution.base_dists])
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    df1 = torch.stack(df1s, dim=dim)
    df2 = torch.stack(df2s, dim=dim)
    base_distribution = type(distribution.base_dists[0])(df1, df2, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Gamma) for b in d.base_dists),
)
def _reduction_cat_gamma_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    concentrations, rates = zip(
        *[(b.concentration, b.rate) for b in distribution.base_dists]
    )
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    concentration = torch.stack(concentrations, dim=dim)
    rate = torch.stack(rates, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        concentration, rate, validate_args
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.HalfCauchy) for b in d.base_dists),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.HalfNormal) for b in d.base_dists),
)
def _reduction_cat_scale_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    scales = [b.scale for b in distribution.base_dists]
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    scale = torch.stack(scales, dim=dim)
    base_distribution = type(distribution.base_dists[0])(scale, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, MaskedDistribution) for b in d.base_dists)
    and all((b.mask == d.base_dists[0].mask).all() for b in d.base_dists[1:]),
)
def _reduction_cat_mask_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    dim = distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    mask = distribution.base_dists[0].mask
    base_distribution = type(distribution)(
        [b.base_dist for b in distribution.base_dists],
        dim,
        validate_args,
    )
    return reduce(
        base_distribution
        if mask.all()
        else MaskedDistribution(base_distribution, mask, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(
        isinstance(b, (ds.MultivariateNormal, ds.LowRankMultivariateNormal))
        for b in d.base_dists
    ),
)
def _reduction_cat_multivariate_normal_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    locs, covariance_matrices = zip(
        *[(b.loc, b.covariance_matrix) for b in distribution.base_dists]
    )
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    loc = torch.cat(locs, dim=dim)
    covariance_matrix = torch.zeros(*loc.shape, loc.shape[-1], device=loc.device)
    row_idx = column_idx = 0
    for k in range(len(covariance_matrices)):
        height, width = covariance_matrices[k].shape[-2:]
        covariance_matrix[
            ..., row_idx : row_idx + height, column_idx : column_idx + width
        ] = covariance_matrices[k]
        row_idx += height
        column_idx += width
    return reduce(
        ds.MultivariateNormal(loc, covariance_matrix, validate_args=validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Pareto) for b in d.base_dists),
)
def _reduction_cat_pareto_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    scales, alphas = zip(*[(b.scale, b.alpha) for b in distribution.base_dists])
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    scale = torch.stack(scales, dim=dim)
    alpha = torch.stack(alphas, dim=dim)
    base_distribution = type(distribution.base_dists[0])(scale, alpha, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.RelaxedBernoulli) for b in d.base_dists)
    and all(b.temperature == d.base_dists[0].temperature for b in d.base_dists[1:])
    and _has_stackable_logits(d),
)
@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.RelaxedOneHotCategorical) for b in d.base_dists)
    and all(b.temperature == d.base_dists[0].temperature for b in d.base_dists[1:])
    and _has_stackable_logits(d),
)
def _reduction_cat_temperature_logits_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    logits = [b.logits for b in distribution.base_dists]
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    temperature = distribution.base_dists[0].temperature
    logits = torch.stack(logits, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        temperature, logits=logits, validate_args=validate_args
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.StudentT) for b in d.base_dists),
)
def _reduction_cat_student_t_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    dfs, locs, scales = zip(*[(b.df, b.loc, b.scale) for b in distribution.base_dists])
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    df = torch.stack(dfs, dim=dim)
    loc = torch.stack(locs, dim=dim)
    scale = torch.stack(scales, dim=dim)
    base_distribution = type(distribution.base_dists[0])(df, loc, scale, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Uniform) for b in d.base_dists),
)
def _reduction_cat_uniform_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    lows, highs = zip(*[(b.low, b.high) for b in distribution.base_dists])
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    low = torch.stack(lows, dim=dim)
    high = torch.stack(highs, dim=dim)
    base_distribution = type(distribution.base_dists[0])(low, high, validate_args)
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.VonMises) for b in d.base_dists),
)
def _reduction_cat_von_mises_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    locs, concentrations = zip(
        *[(b.loc, b.concentration) for b in distribution.base_dists]
    )
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    loc = torch.stack(locs, dim=dim)
    concentration = torch.stack(concentrations, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        loc, concentration, validate_args
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )


@register_reduction(
    CatDistribution,
    ds.Transform,
    lambda d, t: all(isinstance(b, ds.Weibull) for b in d.base_dists),
)
def _reduction_cat_weibull_transform(
    distribution: "CatDistribution", transform: "ds.Transform", **kwargs: "Any"
) -> "ds.Distribution":
    scales, concentrations = zip(
        *[(b.scale, b.concentration) for b in distribution.base_dists]
    )
    dim = len(distribution.batch_shape) + distribution.dim
    validate_args = kwargs.get("validate_args", distribution._validate_args)
    scale = torch.stack(scales, dim=dim)
    concentration = torch.stack(concentrations, dim=dim)
    base_distribution = type(distribution.base_dists[0])(
        scale, concentration, validate_args
    )
    return reduce(
        ds.Independent(base_distribution, 1, validate_args),
        transform,
        **kwargs,
    )
