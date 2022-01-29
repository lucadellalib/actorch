# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Common utilities."""

import importlib
import os
import random

import numpy as np
import torch

import actorch.resources as actorch_resources


__all__ = [
    "get_resource",
    "normalize_byte_size",
    "set_seed",
]


def get_resource(path: "str") -> "str":
    """Return the absolute path to a resource.

    Parameters
    ----------
    path:
        The path to the resource, relative
        to directory `resources`.

    Returns
    -------
        The absolute path to the resource.

    """
    subpackage_name, resource_name = f"{os.sep}{os.path.normpath(path)}".rsplit(
        os.sep, 1
    )
    package_name = actorch_resources.__name__ + subpackage_name.replace(os.sep, ".")
    if package_name.endswith("."):
        package_name = package_name[:-1]
    with importlib.resources.path(package_name, resource_name) as realpath:
        return str(realpath)


def normalize_byte_size(byte_size: "int") -> "str":
    """Convert a size from B (bytes) to X, where X is the least multiple
    of B such that the normalized size is less than 1024 X.

    Parameters
    ----------
    byte_size:
        The size in bytes.

    Returns
    -------
        The normalized size.

    Raises
    ------
    ValueError
        If `byte_size` is not in the integer interval [0, inf).

    """
    if byte_size < 0 or not float(byte_size).is_integer():
        raise ValueError(
            f"`byte_size` ({byte_size}) must be in the integer interval [0, inf)"
        )
    byte_size = int(byte_size)
    factor = 1024
    for prefix in ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]:
        if byte_size < factor:
            break
        byte_size /= factor
    return f"{byte_size:.2f} {prefix}B"


def set_seed(seed: "int") -> "None":
    """Set the seed for generating random numbers.

    Parameters
    ----------
    seed:
        The seed.

    Raises
    ------
    ValueError
        If `seed` is not in the integer interval [0, 2^32).

    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    except (TypeError, ValueError):
        raise ValueError(f"`seed` ({seed}) must be in the integer interval [0, 2^32)")
