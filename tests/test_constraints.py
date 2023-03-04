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

"""Test constraints."""

import pytest
import torch
from torch.distributions.constraints import independent, positive, real

from actorch.distributions.constraints import cat, ordered_real_vector, real_set


def test_cat() -> "None":
    constraint = cat([independent(real, 1), independent(positive, 1)], lengths=(2, 1))
    check = constraint.check(torch.as_tensor([-0.2, -0.5, 2.3]))
    print(f"Constraint: {constraint}")
    print(f"Is discrete: {constraint.is_discrete}")
    print(f"Check: {check}")


def test_ordered_real_vector() -> "None":
    constraint = ordered_real_vector
    check = constraint.check(torch.as_tensor([0.2, 0.4, 0.8]))
    print(f"Constraint: {constraint}")
    print(f"Is discrete: {constraint.is_discrete}")
    print(f"Check: {check}")
    print(f"Is discrete: {constraint.is_discrete}")


def test_real_set() -> "None":
    constraint = real_set(torch.as_tensor([0.2, 0.4, 0.8]))
    check = constraint.check(torch.as_tensor(0.2))
    print(f"Constraint: {constraint}")
    print(f"Is discrete: {constraint.is_discrete}")
    print(f"Check: {check}")


if __name__ == "__main__":
    pytest.main([__file__])
