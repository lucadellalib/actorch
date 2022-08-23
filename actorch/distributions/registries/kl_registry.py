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

"""Kullback-Leibler divergence registry."""

from typing import List, Union

import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    LowRankMultivariateNormal,
    MultivariateNormal,
    Normal,
    kl,
    kl_divergence,
    register_kl,
    utils,
)

from actorch.distributions.cat_distribution import CatDistribution
from actorch.distributions.deterministic import Deterministic
from actorch.distributions.finite import Finite
from actorch.distributions.masked_distribution import MaskedDistribution
from actorch.distributions.transformed_distribution import TransformedDistribution
from actorch.distributions.utils import l2_project


__all__: "List[str]" = []


#####################################################################################################
# Deterministic distribution
#####################################################################################################


@register_kl(Deterministic, Distribution)
@register_kl(Deterministic, TransformedDistribution)  # Avoid ambiguities
def _kl_deterministic_distribution(p: "Deterministic", q: "Distribution") -> "Tensor":
    return -q.log_prob(p.value)


#####################################################################################################
# Finite distribution
#####################################################################################################


@register_kl(Finite, Finite)
def _kl_finite_finite(p: "Finite", q: "Finite") -> "Tensor":
    projected_p = Finite(probs=l2_project(p.atoms, p.probs, q.atoms), atoms=q.atoms)
    return kl._kl_categorical_categorical(projected_p, q)


#####################################################################################################
# Independent distribution
#####################################################################################################


# Adapted from:
# https://github.com/pyro-ppl/pyro/blob/6fa4d8b2810dff9b3bcc4b7e994029cdc818520f/pyro/distributions/kl.py#L24
@register_kl(Independent, Independent)
def _kl_independent_independent(p: "Independent", q: "Independent") -> "Tensor":
    # Handle nested independent distributions
    shared_ndims = min(p.reinterpreted_batch_ndims, q.reinterpreted_batch_ndims)
    p_ndims = p.reinterpreted_batch_ndims - shared_ndims
    q_ndims = q.reinterpreted_batch_ndims - shared_ndims
    base_dist_p = Independent(p.base_dist, p_ndims) if p_ndims else p.base_dist
    base_dist_q = Independent(q.base_dist, q_ndims) if q_ndims else q.base_dist
    result = kl_divergence(base_dist_p, base_dist_q)
    if shared_ndims:
        result = utils._sum_rightmost(result, shared_ndims)
    return result


@register_kl(Independent, Distribution)
def _kl_independent_distribution(p: "Independent", q: "Distribution") -> "Tensor":
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q)
    return utils._sum_rightmost(result, p.reinterpreted_batch_ndims)


@register_kl(Independent, MultivariateNormal)
@register_kl(Independent, LowRankMultivariateNormal)
def _kl_independent_multivariate_normal(
    p: "Independent", q: "Union[MultivariateNormal, LowRankMultivariateNormal]"
) -> "Tensor":
    if isinstance(p.base_dist, Deterministic) and len(p.event_shape) == 1:
        return -q.log_prob(p.base_dist.value)
    if isinstance(p.base_dist, Normal) and len(p.event_shape) == 1:
        multivariate_normal_p = MultivariateNormal(
            p.base_dist.loc, scale_tril=p.base_dist.scale.diag_embed()
        )
        return kl_divergence(multivariate_normal_p, q)
    return _kl_independent_distribution(p, q)


@register_kl(Distribution, Independent)
def _kl_distribution_independent(p: "Distribution", q: "Independent") -> "Tensor":
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    result = kl_divergence(p, q.base_dist)
    return utils._sum_rightmost(result, q.reinterpreted_batch_ndims)


@register_kl(MultivariateNormal, Independent)
@register_kl(LowRankMultivariateNormal, Independent)
def _kl_multivariate_normal_independent(
    p: "Union[MultivariateNormal, LowRankMultivariateNormal]", q: "Independent"
) -> "Tensor":
    if isinstance(q.base_dist, Normal) and len(q.event_shape) == 1:
        multivariate_normal_q = MultivariateNormal(
            q.base_dist.loc, scale_tril=q.base_dist.scale.diag_embed()
        )
        return kl_divergence(p, multivariate_normal_q)
    return _kl_distribution_independent(p, q)


#####################################################################################################
# Transformed distribution
#####################################################################################################


@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(
    p: "TransformedDistribution", q: "TransformedDistribution"
) -> "Tensor":
    reduced_p, reduced_q = p.reduced_dist, q.reduced_dist
    if reduced_p is None and reduced_q is None:
        return kl._kl_transformed_transformed(p, q)
    return kl_divergence(reduced_p or p, reduced_q or q)


@register_kl(TransformedDistribution, Distribution)
@register_kl(TransformedDistribution, Independent)  # Avoid ambiguities
def _kl_transformed_distribution(
    p: "TransformedDistribution", q: "Distribution"
) -> "Tensor":
    reduced_p = p.reduced_dist
    return kl_divergence(reduced_p, q)


@register_kl(Distribution, TransformedDistribution)
@register_kl(Independent, TransformedDistribution)  # Avoid ambiguities
def _kl_distribution_transformed(
    p: "Distribution", q: "TransformedDistribution"
) -> "Tensor":
    reduced_q = q.reduced_dist
    return kl_divergence(p, reduced_q)


#####################################################################################################
# Masked distribution
#####################################################################################################


@register_kl(MaskedDistribution, MaskedDistribution)
def _kl_masked_masked(p: "MaskedDistribution", q: "MaskedDistribution") -> "Tensor":
    result = kl_divergence(p.base_dist, q.base_dist)
    mask = p.mask & q.mask
    return result.masked_fill(~mask, 0.0)


@register_kl(MaskedDistribution, Distribution)
def _kl_masked_distribution(p: "MaskedDistribution", q: "Distribution") -> "Tensor":
    if not p.mask.all():
        raise NotImplementedError
    return kl_divergence(p.base_dist, q)


@register_kl(Distribution, MaskedDistribution)
def _kl_distribution_masked(p: "Distribution", q: "MaskedDistribution") -> "Tensor":
    if not q.mask.all():
        raise NotImplementedError
    return kl_divergence(p, q.base_dist)


#####################################################################################################
# Concatenated distribution
#####################################################################################################


@register_kl(CatDistribution, CatDistribution)
def _kl_cat_cat(p: "CatDistribution", q: "CatDistribution") -> "Tensor":
    if (p.dim != q.dim) or (len(p.base_dists) != len(q.base_dists)):
        raise NotImplementedError
    return torch.stack(
        [kl_divergence(d, q.base_dists[i]) for i, d in enumerate(p.base_dists)]
    ).sum(dim=0)


@register_kl(CatDistribution, Distribution)
def _kl_cat_distribution(p: "CatDistribution", q: "Distribution") -> "Tensor":
    if len(p.base_dists) > 1:
        raise NotImplementedError
    return kl_divergence(p.base_dists[0], q)


@register_kl(Distribution, CatDistribution)
def _kl_distribution_cat(p: "Distribution", q: "CatDistribution") -> "Tensor":
    if len(q.base_dists) > 1:
        raise NotImplementedError
    return kl_divergence(p, q.base_dists[0])
