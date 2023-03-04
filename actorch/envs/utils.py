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

"""Environment utilities."""

from functools import singledispatch
from typing import Container, Dict, Hashable, List, Sequence, Tuple, TypeVar, Union

import numpy as np
from gymnasium import spaces
from numpy import ndarray
from scipy import stats


__all__ = [
    "Nested",
    "batch",
    "batch_space",
    "flatten",
    "get_log_prob",
    "get_space_bounds",
    "nest",
    "unbatch",
    "unflatten",
    "unnest",
    "unnest_space",
]


_T = TypeVar("_T")

Nested = Union[_T, Container["Nested[_T]"]]
"""Generic nested type."""


@singledispatch
def batch_space(space: "spaces.Space", batch_size: "int" = 1) -> "spaces.Space":
    """Stack `batch_size` copies of `space`
    to form a batched space.

    Parameters
    ----------
    space:
        The space.
    batch_size:
        The batch size.

    Returns
    -------
        The batched space.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import batch_space
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @batch_space.register(Custom)
    >>> def _batch_space_custom(space, batch_size=1):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `batch_space.register`"
    )


@singledispatch
def unnest_space(
    space: "spaces.Space",
    prefix: "str" = "",
) -> "Dict[str, spaces.Space]":
    """Unnest a space, i.e. convert it to a flat
    dict that maps names of the paths to the leaf
    spaces to the leaf spaces themselves.

    Parameters
    ----------
    space:
        The space.
    prefix:
        The prefix to prepend to each
        key in the unnested space.

    Returns
    -------
        The unnested space.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import unnest_space
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @unnest_space.register(Custom)
    >>> def _unnest_space_custom(space, prefix=""):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `unnest_space.register`"
    )


@singledispatch
def get_space_bounds(space: "spaces.Space") -> "Tuple[Nested, Nested]":
    """Return the lower and upper bounds of a space.

    Parameters
    ----------
    space:
        The space.

    Returns
    -------
        - The lower bound;
        - the upper bound.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import get_space_bounds
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @get_space_bounds.register(Custom)
    >>> def _get_space_bounds_custom(space):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `get_space_bounds.register`"
    )


def flatten(
    space: "spaces.Space",
    x: "Nested",
    is_batched: "bool" = False,
    copy: "bool" = True,
    validate_args: "bool" = False,
) -> "ndarray":
    """Flatten a (possibly batched) sample
    from a (possibly batched) space.

    Parameters
    ----------
    space:
        The (possibly batched) space.
    x:
        The (possibly batched) sample.
    is_batched:
        True to preserve the batch size (if `x`
        is a batched sample from batched space
        `space`), False otherwise.
    copy:
        True to return a copy, False otherwise
        (if possible according to NumPy copying semantics).
    validate_args:
        True to validate the arguments, False otherwise.

    Returns
    -------
        The (possibly batched) flat sample.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    """
    unnested = unnest(space, x, validate_args)
    try:
        batch_shape = (len(unnested[0]),) if is_batched else ()
    except Exception:
        raise ValueError(f"`space` ({space}) and `x` ({x}) must be batched")
    if len(unnested) == 1:
        array = np.array(unnested[0], copy=copy)
        if array.ndim < 2:
            return array
        return array.reshape(*batch_shape, -1)
    arrays = []
    ndmin = 1
    for v in unnested:
        try:
            array = np.asarray(v).reshape(*batch_shape, -1)
        except Exception:
            raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
        if array.ndim > ndmin:
            ndmin = array.ndim
        arrays.append(array)
    arrays = [np.array(v.T, copy=False, ndmin=ndmin).T for v in arrays]
    return np.concatenate(arrays, axis=-1)  # Copy


_UNFLATTEN_CACHE: "Dict[Tuple[int, bool], ndarray]" = (
    {}
)  # Cache split_idxes to improve performance of unflatten


def unflatten(
    space: "spaces.Space",
    x: "ndarray",
    is_batched: "bool" = False,
    copy: "bool" = True,
    validate_args: "bool" = False,
) -> "Nested":
    """Unflatten a (possibly batched) flat sample
    from a (possibly batched) space.

    Parameters
    ----------
    space:
        The (possibly batched) space.
    x:
        The (possibly batched) flat sample.
    is_batched:
        True to preserve the batch size (if `x`
        is a batched sample from batched space
        `space`), False otherwise.
    copy:
        True to return a copy, False otherwise
        (if possible according to NumPy copying semantics).
    validate_args:
        True to validate the arguments, False otherwise.

    Returns
    -------
        The (possibly batched) sample.

    Raises
    ------
    ValueError
        If `x` is not a batched flat sample from `space`.

    """
    batch_shape = (len(x),) if is_batched else ()
    try:
        split_idxes = _UNFLATTEN_CACHE[id(space), is_batched]
    except KeyError:
        unnested = unnest(space, space.sample())
        arrays = []
        ndmin = 1
        for v in unnested:
            try:
                array = np.asarray(v).reshape(*batch_shape, -1)
            except Exception:
                raise ValueError(
                    f"`x` ({x}) must be a batched flat sample from `space` ({space})"
                )
            if array.ndim > ndmin:
                ndmin = array.ndim
            arrays.append(array)
        lengths = np.asarray(
            [np.array(v.T, copy=False, ndmin=ndmin).T.shape[-1] for v in arrays],
            dtype=np.int64,
        )
        split_idxes = lengths[:-1].cumsum()
        _UNFLATTEN_CACHE[id(space), is_batched] = split_idxes
    chunks = [
        np.array(v, copy=copy).reshape(*batch_shape, -1)
        for v in ([x] if len(split_idxes) == 0 else np.split(x, split_idxes, axis=-1))
    ]
    return nest(space, chunks, validate_args)


@singledispatch
def unnest(
    space: "spaces.Space",
    x: "Nested",
    validate_args: "bool" = False,
) -> "List":
    """Unnest a sample from a space, i.e. convert
    it to a flat list containing the leaf values.

    Parameters
    ----------
    space:
        The space.
    x:
        The sample.
    validate_args:
        True to validate the arguments, False otherwise.

    Returns
    -------
        The unnested sample.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import unnest
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @unnest.register(Custom)
    >>> def _unnest_custom(space, x, validate_args=False):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `unnest.register`"
    )


@singledispatch
def nest(
    space: "spaces.Space",
    x: "List",
    validate_args: "bool" = False,
) -> "Nested":
    """Nest an unnested sample (i.e. a flat list
    containing the leaf values) from a space.

    Parameters
    ----------
    space:
        The space.
    x:
        The unnested sample.
    validate_args:
        True to validate the arguments, False otherwise.

    Returns
    -------
        The sample.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import nest
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @nest.register(Custom)
    >>> def _nest_custom(space, x, validate_args=False):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `nest.register`"
    )


@singledispatch
def batch(
    space: "spaces.Space",
    x: "Sequence[Nested]",
    validate_args: "bool" = False,
) -> "Nested[ndarray]":
    """Batch an unbatched sample from a batched space.

    Parameters
    ----------
    space:
        The batched space.
    x:
        The unbatched sample (i.e. a sequence of samples
        from the corresponding unbatched space).
    validate_args:
        True to validate the arguments, False otherwise.

    Returns
    -------
        The batched sample.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import batch
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @batch.register(Custom)
    >>> def _batch_custom(space, x, validate_args=False):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `batch.register`"
    )


@singledispatch
def unbatch(
    space: "spaces.Space",
    x: "Nested[ndarray]",
    validate_args: "bool" = False,
) -> "List[Nested]":
    """Unbatch a batched sample from a batched space.

    Parameters
    ----------
    space:
        The batched space.
    x:
        The batched sample.
    validate_args:
        True to validate the arguments, False otherwise.

    Returns
    -------
        The unbatched sample (i.e. a list of samples
        from the corresponding unbatched space).

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import unbatch
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @unbatch.register(Custom)
    >>> def _unbatch_custom(space, x, validate_args=False):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `unbatch.register`"
    )


@singledispatch
def get_log_prob(
    space: "spaces.Space",
    x: "Nested",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    """Return the log probability of a (possibly batched)
    sample from a (possibly batched) space.

    Parameters
    ----------
    space:
        The (possibly batched) space.
    x:
        The (possibly batched) sample.
    is_batched:
        True to preserve the batch size (if `x`
        is a batched sample from batched space
        `space`), False otherwise.
    validate_args:
        True to validate the arguments, False otherwise.

    Returns
    -------
        The (possibly batched) log probability.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.

    Notes
    -----
    Register a custom space type as follows:
    >>> from gymnasium import spaces
    >>>
    >>> from actorch.envs.utils import get_log_prob
    >>>
    >>>
    >>> class Custom(spaces.Space):
    >>>     # Implementation
    >>>     ...
    >>>
    >>>
    >>> @unbatch.register(Custom)
    >>> def _get_log_prob_custom(space, x, is_batched=False, validate_args=False):
    >>>     # Implementation
    >>>     ...

    """
    raise NotImplementedError(
        f"Unsupported space type: "
        f"`{type(space).__module__}.{type(space).__name__}`. "
        f"Register a custom space type through "
        f"decorator `get_log_prob.register`"
    )


#####################################################################################################
# batch_space implementation
#####################################################################################################


@batch_space.register(spaces.Box)
def _batch_space_box(space: "spaces.Box", batch_size: "int" = 1) -> "spaces.Box":
    if batch_size < 1 or not float(batch_size).is_integer():
        raise ValueError(
            f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
        )
    batch_size = int(batch_size)
    reps = (batch_size,) + (1,) * len(space.shape)
    low, high = np.tile(space.low, reps), np.tile(space.high, reps)
    return spaces.Box(low, high, dtype=space.dtype)


@batch_space.register(spaces.Discrete)
def _batch_space_discrete(
    space: "spaces.Discrete", batch_size: "int" = 1
) -> "spaces.MultiDiscrete":
    if batch_size < 1 or not float(batch_size).is_integer():
        raise ValueError(
            f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
        )
    batch_size = int(batch_size)
    return spaces.MultiDiscrete(np.full((batch_size,), space.n, space.dtype))


@batch_space.register(spaces.MultiBinary)
def _batch_space_multi_binary(
    space: "spaces.MultiBinary", batch_size: "int" = 1
) -> "spaces.MultiBinary":
    if batch_size < 1 or not float(batch_size).is_integer():
        raise ValueError(
            f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
        )
    batch_size = int(batch_size)
    return spaces.MultiBinary((batch_size,) + space.shape)


@batch_space.register(spaces.MultiDiscrete)
def _batch_space_multi_discrete(
    space: "spaces.MultiDiscrete",
    batch_size: "int" = 1,
) -> "spaces.MultiDiscrete":
    if batch_size < 1 or not float(batch_size).is_integer():
        raise ValueError(
            f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
        )
    batch_size = int(batch_size)
    reps = (batch_size,) + (1,) * len(space.shape)
    nvec = np.tile(space.nvec, reps)
    return spaces.MultiDiscrete(nvec, space.dtype)


@batch_space.register(spaces.Tuple)
def _batch_space_tuple(
    space: "spaces.Tuple",
    batch_size: "int" = 1,
) -> "spaces.Tuple":
    return spaces.Tuple([batch_space(v, batch_size) for v in space])


@batch_space.register(spaces.Dict)
def _batch_space_dict(
    space: "spaces.Dict",
    batch_size: "int" = 1,
) -> "spaces.Dict":
    return spaces.Dict({k: batch_space(v, batch_size) for k, v in space.items()})


#####################################################################################################
# unnest_space implementation
#####################################################################################################


@unnest_space.register(spaces.Box)
@unnest_space.register(spaces.Discrete)
@unnest_space.register(spaces.MultiBinary)
@unnest_space.register(spaces.MultiDiscrete)
def _unnest_space_base(
    space: "Union[spaces.Box, spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete]",
    prefix: "str" = "",
) -> "Dict[str, Union[spaces.Box, spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete]]":
    return {prefix: space}


@unnest_space.register(spaces.Tuple)
def _unnest_space_tuple(
    space: "spaces.Tuple",
    prefix: "str" = "",
) -> "Dict[str, spaces.Space]":
    result = {}
    for i, v in enumerate(space):
        result.update(unnest_space(v, f"{prefix}/{i}"))
    return result


@unnest_space.register(spaces.Dict)
def _unnest_space_dict(
    space: "spaces.Dict",
    prefix: "str" = "",
) -> "Dict[str, spaces.Space]":
    result = {}
    for k, v in space.items():
        result.update(unnest_space(v, f"{prefix}/{k}"))
    return result


#####################################################################################################
# get_space_bounds implementation
#####################################################################################################


@get_space_bounds.register(spaces.Box)
def _get_space_bounds_box(space: "spaces.Box") -> "Tuple[ndarray, ndarray]":
    return space.low, space.high


@get_space_bounds.register(spaces.Discrete)
def _get_space_bounds_discrete(space: "spaces.Discrete") -> "Tuple[int, int]":
    return 0, space.n - 1


@get_space_bounds.register(spaces.MultiBinary)
def _get_space_bounds_multi_binary(
    space: "spaces.MultiBinary",
) -> "Tuple[ndarray, ndarray]":
    low = np.zeros(space.shape, space.dtype)
    high = np.ones_like(low)
    return low, high


@get_space_bounds.register(spaces.MultiDiscrete)
def _get_space_bounds_multi_discrete(
    space: "spaces.MultiDiscrete",
) -> "Tuple[ndarray, ndarray]":
    high = space.nvec - 1
    low = np.zeros_like(high)
    return low, high


@get_space_bounds.register(spaces.Tuple)
def _get_space_bounds_tuple(
    space: "spaces.Tuple",
) -> "Tuple[Tuple[Nested, ...], Tuple[Nested, ...]]":
    lows, highs = [], []
    for v in space:
        low, high = get_space_bounds(v)
        lows.append(low)
        highs.append(high)
    return tuple(lows), tuple(highs)


@get_space_bounds.register(spaces.Dict)
def _get_space_bounds_dict(
    space: "spaces.Dict",
) -> "Tuple[Dict[Hashable, Nested], Dict[Hashable, Nested]]":
    lows, highs = {}, {}
    for k, v in space.items():
        lows[k], highs[k] = get_space_bounds(v)
    return lows, highs


#####################################################################################################
# unnest implementation
#####################################################################################################


@unnest.register(spaces.Box)
@unnest.register(spaces.Discrete)
@unnest.register(spaces.MultiBinary)
@unnest.register(spaces.MultiDiscrete)
def _unnest_base(
    space: "Union[spaces.Box, spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete]",
    x: "Union[int, ndarray]",
    validate_args: "bool" = False,
) -> "List[Union[int, ndarray]]":
    if validate_args and x not in space:
        raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
    return [x]


@unnest.register(spaces.Tuple)
def _unnest_tuple(
    space: "spaces.Tuple",
    x: "Tuple[Nested, ...]",
    validate_args: "bool" = False,
) -> "List":
    if validate_args and not (isinstance(x, tuple) and len(x) == len(space)):
        raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
    result = []
    for i, v in enumerate(space):
        result += unnest(v, x[i], validate_args)
    return result


@unnest.register(spaces.Dict)
def _unnest_dict(
    space: "spaces.Dict",
    x: "Dict[Hashable, Nested]",
    validate_args: "bool" = False,
) -> "List":
    if validate_args and not (isinstance(x, dict) and all(k in x for k in space)):
        raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
    result = []
    for k, v in space.items():
        result += unnest(v, x[k], validate_args)
    return result


#####################################################################################################
# nest implementation
#####################################################################################################


@nest.register(spaces.Box)
@nest.register(spaces.MultiBinary)
@nest.register(spaces.MultiDiscrete)
def _nest_box_multi_binary_multi_discrete(
    space: "Union[spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete]",
    x: "List[ndarray]",
    validate_args: "bool" = False,
) -> "ndarray":
    try:
        result = x[0].reshape(space.shape).astype(space.dtype)
        if validate_args and result not in space:
            raise ValueError
        return result
    except Exception:
        raise ValueError(f"`x` ({x}) must be an unnested sample from `space` ({space})")


@nest.register(spaces.Discrete)
def _nest_discrete(
    space: "spaces.Discrete",
    x: "List[int]",
    validate_args: "bool" = False,
) -> "int":
    try:
        result = float(x[0])
        if not result.is_integer():
            raise ValueError
        result = int(result)
        if validate_args and result not in space:
            raise ValueError
        return result
    except Exception:
        raise ValueError(f"`x` ({x}) must be an unnested sample from `space` ({space})")


_NEST_CACHE: "Dict[int, ndarray]" = (
    {}
)  # Cache split_idxes to improve performance of nest


@nest.register(spaces.Tuple)
def _nest_tuple(
    space: "spaces.Tuple",
    x: "List",
    validate_args: "bool" = False,
) -> "Tuple[Nested, ...]":
    try:
        split_idxes = _NEST_CACHE[id(space)]
    except KeyError:
        lengths = [len(unnest(v, v.sample())) for v in space]
        split_idxes = np.asarray([0] + lengths, dtype=np.int64).cumsum()
        _NEST_CACHE[id(space)] = split_idxes
    try:
        result = tuple(
            nest(v, x[split_idxes[i] : split_idxes[i + 1]], validate_args)
            for i, v in enumerate(space)
        )
        return result
    except Exception:
        raise ValueError(f"`x` ({x}) must be an unnested sample from `space` ({space})")


@nest.register(spaces.Dict)
def _nest_dict(
    space: "spaces.Dict",
    x: "List",
    validate_args: "bool" = False,
) -> "Dict[Hashable, Nested]":
    try:
        split_idxes = _NEST_CACHE[id(space)]
    except KeyError:
        lengths = [len(unnest(v, v.sample())) for v in space.values()]
        split_idxes = np.asarray([0] + lengths, dtype=np.int64).cumsum()
        _NEST_CACHE[id(space)] = split_idxes
    try:
        result = {
            k: nest(v, x[split_idxes[i] : split_idxes[i + 1]], validate_args)
            for i, (k, v) in enumerate(space.items())
        }
        return result
    except Exception:
        raise ValueError(f"`x` ({x}) must be an unnested sample from `space` ({space})")


#####################################################################################################
# batch implementation
#####################################################################################################


@batch.register(spaces.Box)
@batch.register(spaces.MultiBinary)
@batch.register(spaces.MultiDiscrete)
def _batch_base(
    space: "Union[spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete]",
    x: "Sequence[ndarray]",
    validate_args: "bool" = False,
) -> "ndarray":
    try:
        result = np.stack(x)
        if validate_args and result not in space:
            raise ValueError
        return result
    except Exception:
        raise ValueError(
            f"`x` ({x}) must be an unbatched sample from `space` ({space})"
        )


@batch.register(spaces.Tuple)
def _batch_tuple(
    space: "spaces.Tuple",
    x: "Sequence[Tuple[Nested, ...]]",
    validate_args: "bool" = False,
) -> "Tuple[Nested[ndarray], ...]":
    try:
        batch_size = len(x)
        result = tuple(
            batch(v, [x[i][j] for i in range(batch_size)], validate_args)
            for j, v in enumerate(space)
        )
        if validate_args and not (
            isinstance(result, tuple) and len(result) == len(space)
        ):
            raise ValueError
        return result
    except Exception:
        raise ValueError(
            f"`x` ({x}) must be an unbatched sample from `space` ({space})"
        )


@batch.register(spaces.Dict)
def _batch_dict(
    space: "spaces.Dict",
    x: "Sequence[Dict[Hashable, Nested]]",
    validate_args: "bool" = False,
) -> "Dict[Hashable, Nested[ndarray]]":
    try:
        batch_size = len(x)
        result = {
            k: batch(v, [x[i][k] for i in range(batch_size)], validate_args)
            for k, v in space.items()
        }
        if validate_args and not (
            isinstance(result, dict) and all(k in result for k in space)
        ):
            raise ValueError
        return result
    except Exception:
        raise ValueError(
            f"`x` ({x}) must be an unbatched sample from `space` ({space})"
        )


#####################################################################################################
# unbatch implementation
#####################################################################################################


@unbatch.register(spaces.Box)
@unbatch.register(spaces.MultiBinary)
@unbatch.register(spaces.MultiDiscrete)
def _unbatch_base(
    space: "Union[spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete]",
    x: "ndarray",
    validate_args: "bool" = False,
) -> "List[ndarray]":
    if validate_args:
        if x not in space:
            raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
        if len(space.shape) == 0:
            raise ValueError(f"`space` ({space}) and `x` ({x}) must be batched")
    return list(x)


@unbatch.register(spaces.Tuple)
def _unbatch_tuple(
    space: "spaces.Tuple",
    x: "Tuple[Nested[ndarray], ...]",
    validate_args: "bool" = False,
) -> "List[Tuple[Nested, ...]]":
    if validate_args and not (isinstance(x, tuple) and len(x) == len(space)):
        raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
    result = [unbatch(v, x[i], validate_args) for i, v in enumerate(space)]
    batch_size = len(result[0])
    if validate_args and any(len(v) != batch_size for v in result[1:]):
        raise ValueError(
            f"Batch sizes ({[len(v) for v in result]}) must be "
            f"equal across samples from different subspaces"
        )
    return [tuple(result[j][i] for j in range(len(space))) for i in range(batch_size)]


@unbatch.register(spaces.Dict)
def _unbatch_dict(
    space: "spaces.Dict",
    x: "Dict[Hashable, Nested[ndarray]]",
    validate_args: "bool" = False,
) -> "List[Dict[Hashable, Nested]]":
    if validate_args and not (isinstance(x, dict) and all(k in x for k in space)):
        raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
    result = [unbatch(v, x[k], validate_args) for k, v in space.items()]
    batch_size = len(result[0])
    if validate_args and any(len(v) != batch_size for v in result[1:]):
        raise ValueError(
            f"Batch sizes ({[len(v) for v in result]}) must be "
            f"equal across samples from different subspaces"
        )
    return [{k: result[j][i] for j, k in enumerate(space)} for i in range(batch_size)]


#####################################################################################################
# get_log_prob implementation
#####################################################################################################


@get_log_prob.register(spaces.Box)
def _get_log_prob_box(
    space: "spaces.Box",
    x: "ndarray",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    if validate_args:
        if x not in space:
            raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
        if is_batched and len(space.shape) == 0:
            raise ValueError(f"`space` ({space}) and `x` ({x}) must be batched")
    unbounded = ~space.bounded_below & ~space.bounded_above
    lower_bounded = space.bounded_below & ~space.bounded_above
    upper_bounded = ~space.bounded_below & space.bounded_above
    bounded = space.bounded_below & space.bounded_above
    result = np.empty_like(x)
    result[unbounded] = stats.norm().logpdf(x[unbounded])
    result[lower_bounded] = stats.expon().logpdf(
        x[lower_bounded] - space.low[lower_bounded]
    )
    result[upper_bounded] = stats.expon().logpdf(
        space.high[upper_bounded] - x[upper_bounded]
    )
    result[bounded] = stats.uniform(
        loc=space.low[bounded],
        scale=space.high[bounded] - space.low[bounded],
    ).logpdf(x[bounded])
    return result.sum(axis=tuple(range(is_batched, result.ndim)))


@get_log_prob.register(spaces.Discrete)
def _get_log_prob_discrete(
    space: "spaces.Discrete",
    x: "int",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    if validate_args:
        if x not in space:
            raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
        if is_batched:
            raise ValueError(
                f"`is_batched` ({is_batched}) must be "
                f"False for space of type {type(space)}"
            )
    return stats.randint(0, space.n).logpmf(x)


@get_log_prob.register(spaces.MultiBinary)
def _get_log_prob_multi_binary(
    space: "spaces.MultiBinary",
    x: "ndarray",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    if validate_args:
        if x not in space:
            raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
        if is_batched and len(space.shape) == 0:
            raise ValueError(f"`space` ({space}) and `x` ({x}) must be batched")
    result = np.log(np.full(x.shape, 0.5))
    return result.sum(axis=tuple(range(is_batched, result.ndim)))


@get_log_prob.register(spaces.MultiDiscrete)
def _get_log_prob_multi_discrete(
    space: "spaces.MultiDiscrete",
    x: "ndarray",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    if validate_args:
        if x not in space:
            raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
        if is_batched and len(space.shape) == 0:
            raise ValueError(f"`space` ({space}) and `x` ({x}) must be batched")
    result = stats.randint(np.zeros_like(space.nvec), space.nvec).logpmf(x)
    return result.sum(axis=tuple(range(is_batched, result.ndim)))


@get_log_prob.register(spaces.Tuple)
def _get_log_prob_tuple(
    space: "spaces.Tuple",
    x: "Tuple[Nested, ...]",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    if validate_args and not (isinstance(x, tuple) and len(x) == len(space)):
        raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
    arrays = [
        get_log_prob(v, x[i], is_batched, validate_args) for i, v in enumerate(space)
    ]
    try:
        return np.asarray(arrays).sum(axis=0)
    except Exception:
        raise ValueError(
            f"Batch sizes ({[len(v) for v in arrays]}) must be "
            f"equal across samples from different subspaces"
        )


@get_log_prob.register(spaces.Dict)
def _get_log_prob_dict(
    space: "spaces.Dict",
    x: "Dict[Hashable, Nested]",
    is_batched: "bool" = False,
    validate_args: "bool" = False,
) -> "ndarray":
    if validate_args and not (isinstance(x, dict) and all(k in x for k in space)):
        raise ValueError(f"`x` ({x}) must be a sample from `space` ({space})")
    arrays = [
        get_log_prob(v, x[k], is_batched, validate_args) for k, v in space.items()
    ]
    try:
        return np.asarray(arrays).sum(axis=0)
    except Exception:
        raise ValueError(
            f"Batch sizes ({[len(v) for v in arrays]}) must be "
            f"equal across samples from different subspaces"
        )
