# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Tensor utilities."""

import math
from functools import reduce
from typing import Sequence, Union

import torch
from torch import Tensor


__all__ = [
    "base2dec",
    "broadcast_cat",
    "dec2base",
    "is_identical",
]


def broadcast_cat(
    inputs: "Sequence[Tensor]",
    dim: "int" = 0,
) -> "Tensor":
    """Concatenate a sequence of tensors along a dimension.

    Tensor shapes are expanded to the right, and broadcasting
    is applied along all dimensions except for the `dim`-th.

    Parameters
    ----------
    inputs:
        The tensors.
    dim:
        The dimension.

    Returns
    -------
        The broadcast concatenation of `inputs`.

    Raises
    ------
    IndexError
        If `dim` is out of range or not integer.

    """
    max_ndims = max([x.ndim for x in inputs])
    if dim < -max_ndims or dim >= max_ndims or not float(dim).is_integer():
        raise IndexError(
            f"`dim` ({dim}) must be in the integer interval [-{max_ndims}, {max_ndims})"
        )
    dim = (int(dim) + max_ndims) % max_ndims
    expanded_shapes = [x.shape + (1,) * (max_ndims - x.ndim) for x in inputs]
    # Broadcast along all dimensions except for the dim-th
    target_shape = [
        reduce(lambda x, y: x * y // math.gcd(x, y), sizes) if i != dim else -1
        for i, sizes in enumerate(zip(*expanded_shapes))
    ]
    return torch.cat(
        [
            x[(...,) + (None,) * (len(target_shape) - x.ndim)].repeat(
                [
                    target_size // expanded_size if target_size != -1 else 1
                    for expanded_size, target_size in zip(
                        expanded_shapes[i], target_shape
                    )
                ]
            )
            for i, x in enumerate(inputs)
        ],
        dim=dim,
    )


def is_identical(input: "Tensor", dim: "int" = 0) -> "bool":
    """Check if a tensor is identical along a dimension.

    Parameters
    ----------
    input:
        The tensor.
    dim:
        The dimension.

    Returns
    -------
        True if the tensor is identical along
        the `dim`-th dimension, False otherwise.

    Raises
    ------
    IndexError
        If `dim` is out of range or not integer.

    """
    if dim < -input.ndim or dim >= input.ndim or not float(dim).is_integer():
        raise IndexError(
            f"`dim` ({dim}) must be in the integer interval [-{input.ndim}, {input.ndim})"
        )
    dim = int(dim)
    return (input == input.select(dim, 0).unsqueeze(dim)).all(dim=dim).all()


def dec2base(
    input: "Tensor",
    base: "Union[int, Sequence[int], Tensor]" = 2,
) -> "Tensor":
    """Convert a tensor from base-10 to a given
    (possibly mixed) base.

    Parameters
    ----------
    input:
        The tensor.
    base:
        The (possibly mixed) base.

    Returns
    -------
        The tensor in the given base.

    Raises
    ------
    ValueError
        If an invalid argument value is provided.
    OverflowError
        If `input` is too large to be represented
        in the given base.

    """
    input = torch.atleast_1d(input)
    if ((input < 0) | (input != input.int())).any():
        raise ValueError(f"`input` ({input}) must be in the integer interval [0, inf)")
    base = torch.as_tensor(base)
    if ((base < 1) | (base != base.int())).any():
        raise ValueError(f"`base` ({base}) must be in the integer interval [1, inf)")
    if base.ndim > input.ndim + 1:
        raise ValueError(
            f"`base.ndim` ({base.ndim}) must be at most "
            f"equal to `input.ndim + 1` ({input.ndim + 1})"
        )
    base = torch.atleast_1d(base)
    base = base[(None,) * (input.ndim + 1 - base.ndim) + (...,)].expand(
        *input.shape[: input.ndim + 1 - base.ndim], *base.shape
    )
    n = base.prod(dim=-1)
    if (input >= n).any():
        # Overflow
        if base.shape[-1] == 1 and (base[..., 0] != 1).all():
            num_digits = (input.log() / base[..., 0].log()).floor() + 1
            base = base.repeat_interleave(int(num_digits.max()), dim=-1)
        else:
            raise OverflowError(
                f"`input` ({input}) must be in the integer "
                f"interval [{torch.zeros_like(n)}, {n})"
            )
    weights = base.flip(dims=(-1,)).roll(1, dims=-1)
    weights[..., 0] = 1
    weights = weights.cumprod(dim=-1).flip(dims=(-1,)).movedim(-1, 0)
    digits = torch.zeros((*input.shape, weights.shape[0] + 1))
    digits[..., -1] = input
    for i, weight in enumerate(weights):
        quotient = digits[..., -(i + 1)].div(weight, rounding_mode="trunc")
        remainder = digits[..., -(i + 1)] - (weight * quotient)
        digits[..., -(i + 1)], digits[..., -(i + 2)] = quotient, remainder
    return digits[..., 1:].flip(dims=(-1,)).int()


def base2dec(
    input: "Tensor",
    base: "Union[int, Sequence[int], Tensor]" = 2,
) -> "Tensor":
    """Convert a tensor from a given
    (possibly mixed) base to base-10.

    Parameters
    ----------
    input:
        The tensor.
    base:
        The (possibly mixed) base.

    Returns
    -------
        The tensor in base-10.

    Raises
    ------
    ValueError
        If an invalid argument value is provided.
    OverflowError
        If `input` is too large to be represented
        in the given base.

    """
    input = torch.atleast_1d(input)
    if ((input < 0) | (input != input.int())).any():
        raise ValueError(f"`input` ({input}) must be in the integer interval [0, inf)")
    base = torch.as_tensor(base)
    if ((base < 1) | (base != base.int())).any():
        raise ValueError(f"`base` ({base}) must be in the integer interval [1, inf)")
    try:
        base = base.expand_as(input)
    except Exception:
        raise ValueError(
            f"`base` ({base}) must be broadcastable "
            f"to the shape of `input` ({input.shape})"
        )
    if (input >= base).any():
        raise OverflowError(
            f"`input` ({input}) must be in the integer "
            f"interval [{torch.zeros_like(base)}, {base})"
        )
    weights = base.flip(dims=(-1,)).roll(1, dims=-1)
    weights[..., 0] = 1
    weights = weights.cumprod(dim=-1).flip(dims=(-1,))
    return (weights * input).sum(dim=-1)
