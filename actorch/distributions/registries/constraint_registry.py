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

"""Constraint registry."""

from typing import List

from torch.distributions import CatTransform, constraint_registry

from actorch.distributions.constraints import cat


__all__: "List[str]" = []


#####################################################################################################
# Concatenated constraint
#####################################################################################################


@constraint_registry.biject_to.register(cat)
def _biject_to_cat(constraint: "cat") -> "CatTransform":
    return constraint_registry._biject_to_cat(constraint)


@constraint_registry.transform_to.register(cat)
def _transform_to_cat(constraint: "cat") -> "CatTransform":
    return constraint_registry._transform_to_cat(constraint)
