#!/usr/bin/env python3

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

"""Test concatenated distribution."""

import pytest
import torch
from torch.distributions import Categorical, Independent, Normal, kl_divergence

from actorch.distributions import CatDistribution


def test_cat_distribution() -> "None":
    loc = 0.0
    scale = 1.0
    logits = torch.as_tensor([0.25, 0.15, 0.10, 0.30, 0.20])
    try:
        _ = CatDistribution([Normal(loc, scale), Categorical(logits)], dim=3)
        _ = CatDistribution(
            [
                Normal(torch.full((2, 3), loc), torch.full((2, 3), scale)),
                Categorical(logits),
            ]
        )
        _ = CatDistribution(
            [
                Independent(
                    Normal(torch.full((2, 3, 2), loc), torch.full((2, 3, 2), scale)), 2
                ),
                Categorical(logits.expand(2, 5)),
            ],
            dim=1,
        )
    except Exception:
        pass
    distribution = CatDistribution([Normal(loc, scale), Categorical(logits)])
    print(distribution)
    print(distribution.expand((2, 3)))
    if distribution.has_rsample:
        distribution.rsample()
    else:
        distribution.sample()
    print(f"Mean: {distribution.mean}")
    print(f"Mode: {distribution.mode}")
    print(f"Standard deviation: {distribution.stddev}")
    print(f"Variance: {distribution.variance}")
    print(f"Log prob: {distribution.log_prob(distribution.sample())}")
    print(f"Entropy: {distribution.entropy()}")
    print(f"Support: {distribution.support}")
    try:
        print(f"CDF: {distribution.cdf(distribution.sample())}")
        print(f"Enumerated support: {distribution.enumerate_support()}")
        print(f"Enumerated support: {distribution.enumerate_support(False)}")
    except NotImplementedError:
        pass
    print(
        f"Kullback-Leibler divergence: "
        f"{kl_divergence(distribution, CatDistribution([Normal(loc, scale), Categorical(logits)], validate_args=True))}"
    )
    print(
        f"Kullback-Leibler divergence: "
        f"{kl_divergence(CatDistribution([Normal(loc, scale)]), Normal(loc, scale))}"
    )
    print(
        f"Kullback-Leibler divergence: "
        f"{kl_divergence(Normal(loc, scale), CatDistribution([Normal(loc, scale)]))}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
