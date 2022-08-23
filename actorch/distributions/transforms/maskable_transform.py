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

"""Maskable transform."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributions import Transform


__all__ = [
    "MaskableTransform",
]


class MaskableTransform(ABC, Transform):
    """Transform compatible with `actorch.distributions.MaskedDistribution`. When applied
    to a masked distribution, attribute `mask`, i.e. the boolean tensor indicating which
    elements of the transform input are valid (True) and which are not (False), must be
    set accordingly.

    See Also
    --------
    actorch.distributions.masked_distribution.MaskedDistribution

    """

    # override
    def __init__(self, cache_size: "int" = 0) -> "None":
        super().__init__(cache_size)
        self.mask = torch.as_tensor(True)[(None,) * self.domain.event_dim]

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(mask: {self.mask if self.mask.numel() == 1 else self.mask.shape})"
        )

    @property
    @abstractmethod
    def transformed_mask(self) -> "Tensor":
        """Return the transformed mask, i.e. the boolean tensor indicating which
        elements of the transform output are valid (True) and which are not (False).

        Returns
        -------
            The transformed mask.

        """
        raise NotImplementedError
