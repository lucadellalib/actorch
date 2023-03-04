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

"""Test finite distribution."""

import pytest
import torch
from torch.distributions import kl_divergence

from actorch.distributions import Finite


def test_finite() -> "None":
    logits = torch.as_tensor([0.25, 0.15, 0.10, 0.30, 0.20])
    atoms = torch.as_tensor([5.0, 7.5, 10.0, 12.5, 15.0])
    distribution = Finite(logits, atoms=atoms)
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
    print(f"CDF: {distribution.cdf(distribution.sample())}")
    print(f"Entropy: {distribution.entropy()}")
    print(f"Support: {distribution.support}")
    print(f"Enumerated support: {distribution.enumerate_support()}")
    try:
        print(f"Enumerated support: {distribution.enumerate_support(False)}")
    except NotImplementedError:
        pass
    print(
        f"Kullback-Leibler divergence: "
        f"{kl_divergence(distribution, Finite(atoms, validate_args=True))}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
