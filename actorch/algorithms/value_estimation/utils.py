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

"""Value estimation utilities."""

from typing import Any, Optional

import torch
from torch import Tensor
from torch.distributions import Distribution


__all__ = [
    "distributional_gather",
]


def distributional_gather(
    distribution: "Distribution",
    dim: "int",
    index: "Tensor",
    mask: "Optional[Tensor]" = None,
) -> "Distribution":
    """Gather batch elements of a distribution at the given
    batched index along a batch dimension.

    In the following, let `B = {B_1, ..., B_k}` denote the
    distribution batch shape, `E` the distribution event shape
    and `I = {I_1, ..., I_k} s.t. I_h = B_h for each h s.t. h != dim`
    the batched index shape.

    Parameters
    ----------
    distribution:
        The distribution, shape: ``[*B, *E]``.
    dim:
        The batch dimension.
    index:
        The batched index, shape: ``[*I]``.
    mask:
        The boolean tensor indicating which batch elements are
        valid (True) and which are not (False), shape: ``[*B]``.
        Default to ``torch.ones(I, dtype=torch.bool)``.

    Returns
    -------
        The gathered distribution, shape: ``[*I, *E]``.

    See Also
    --------
    torch.gather

    """
    expand_backup = torch.Tensor.expand

    def expand(input: "Tensor", *args: "Any", **kwargs: "Any") -> "Tensor":
        expanded_index = expand_backup(
            index[(...,) + (None,) * (input.ndim - index.ndim)],
            (-1,) * index.ndim + input.shape[index.ndim :],
        )
        result = input.gather(dim, expanded_index)
        if mask is not None:
            expanded_mask = mask[(...,) + (None,) * (input.ndim - mask.ndim)]
            expanded_mask = expanded_mask.expand_as(result)
            result *= expanded_mask
        return result

    torch.Tensor.expand = expand
    try:
        return distribution.expand(index.shape)
    finally:
        torch.Tensor.expand = expand_backup
