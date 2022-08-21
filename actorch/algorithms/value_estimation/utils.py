# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
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
