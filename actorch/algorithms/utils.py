# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Algorithm utilities."""

from contextlib import contextmanager
from typing import Iterator, Tuple

import torch
from torch.nn import Module


__all__ = [
    "count_params",
    "freeze_params",
    "sync_polyak",
]


def count_params(module: "Module") -> "Tuple[int, int]":
    """Return the number of trainable and non-trainable
    parameters in `module`.

    Parameters
    ----------
    module:
        The module.

    Returns
    -------
        The number of trainable parameters;
        the number of non-trainable parameters.

    """
    num_trainable, num_non_trainable = 0, 0
    for param in module.parameters():
        if param.requires_grad:
            num_trainable += param.numel()
        else:
            num_non_trainable += param.numel()
    return num_trainable, num_non_trainable


@contextmanager
def freeze_params(*modules: "Module") -> "Iterator[None]":
    """Context manager that stops the gradient
    flow in `modules`.

    Parameters
    ----------
    modules:
        The modules.

    """
    params = [
        param
        for module in modules
        for param in module.parameters()
        if param.requires_grad
    ]
    try:
        for param in params:
            param.requires_grad = False
        yield
    finally:
        for param in params:
            param.requires_grad = True


def sync_polyak(
    source_module: "Module",
    target_module: "Module",
    polyak_weight: "float" = 0.001,
) -> "None":
    """Synchronize `source_module` with `target_module`
    via Polyak averaging.

    For each `target_param` in `target_module`,
    for each `source_param` in `source_module`:
    ``target_param = (
        (1 - polyak_weight) * target_param +
        polyak_weight * source_param
    )``.

    Parameters
    ----------
    source_module:
        The source module.
    target_module:
        The target module.
    polyak_weight:
        The Polyak weight.

    Raises
    ------
    ValueError
        If `polyak_weight` is not in the interval [0, 1].

    References
    ----------
    .. [1] B. T. Polyak and A. B. Juditsky.
           "Acceleration of Stochastic Approximation by Averaging".
           In: SIAM Journal on Control and Optimization. 1992, pp. 838-855.
           URL: https://doi.org/10.1137/0330046

    """
    if polyak_weight < 0.0 or polyak_weight > 1.0:
        raise ValueError(
            f"`polyak_weight` ({polyak_weight}) must be in the interval [0, 1]"
        )
    if polyak_weight == 0.0:
        return
    if polyak_weight == 1.0:
        target_module.load_state_dict(source_module.state_dict())
        return
    target_params = target_module.parameters()
    source_params = source_module.parameters()
    with torch.no_grad():
        for target_param, source_param in zip(target_params, source_params):
            # Update in-place
            target_param *= (1 - polyak_weight) / polyak_weight
            target_param += source_param
            target_param *= polyak_weight
