# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Registry."""

import functools
from typing import Any, Callable, Optional, Union

from ray import cloudpickle as ray_cloudpickle
from ray.experimental import internal_kv
from ray.tune import Trainable, registry


__all__ = [
    "get",
    "register",
]


_OTHER = (
    "actorch"  # Custom category for registrable components other than tune.Trainable
)


def register(
    component: "Optional[Any]" = None,
    name: "Optional[str]" = None,
) -> "Union[Any, Callable]":
    """Decorator to register a component.

    Parameters
    ----------
    component:
        The component.
    name:
        The name to register the component with.
        Default to ``component.__qualname__``.

    Returns
    -------
        The decorated component.

    """

    @functools.wraps(component)
    def register_fn(component: "Any") -> "Any":
        component_name = name or component.__qualname__
        if hasattr(component, "__mro__") and Trainable in component.__mro__:
            registry.register_trainable(component_name, component)
            return component
        global_registry = registry._global_registry
        global_registry._to_flush[
            (_OTHER, component_name)
        ] = ray_cloudpickle.dumps_debug(component)
        if internal_kv._internal_kv_initialized():
            global_registry.flush_values()
        return component

    if component:
        return register_fn(component)

    return register_fn


def get(name: "str") -> "Any":
    """Return the component previously registered with `name`.

    Parameters
    ----------
    name:
        The component name.

    Returns
    -------
        The component previously registered with `name`.

    Raises
    ------
    KeyError
        If `name` does not exist in ACTorch registry.

    """
    global_registry = registry._global_registry
    if global_registry.contains(registry.TRAINABLE_CLASS, name):
        return global_registry.get(registry.TRAINABLE_CLASS, name)
    if global_registry.contains(_OTHER, name):
        return global_registry.get(_OTHER, name)
    raise KeyError(f"{name}")
