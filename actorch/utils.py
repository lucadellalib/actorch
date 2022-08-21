# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Common utilities."""

import importlib
import json
import os
import platform
from datetime import datetime
from functools import singledispatch, update_wrapper
from types import ModuleType
from typing import Any, Callable, Dict, Generic, Optional, Sequence, TypeVar, Union

import GPUtil
import psutil
import yaml
from ray.tune.logger import SafeFallbackEncoder


__all__ = [
    "CheckpointableMixin",
    "FutureRef",
    "get_system_info",
    "import_module",
    "pretty_print",
    "singledispatchmethod",
]


_T = TypeVar("_T")


class CheckpointableMixin:
    """Mixin that implements PyTorch-like checkpointing functionalities."""

    _STATE_VARS: "Sequence[str]"

    def state_dict(
        self, exclude_keys: "Optional[Sequence[str]]" = None
    ) -> "Dict[str, Any]":
        """Return the object state dict.

        Parameters
        ----------
        exclude_keys:
            The keys in the object state dict
            whose values must not be saved.
            Default to ``[]``.

        Returns
        -------
            The object state dict.

        """
        if exclude_keys is None:
            exclude_keys = []
        state_dict = {}
        for k in self._STATE_VARS:
            if k in exclude_keys:
                continue
            v = self.__dict__[k]
            if hasattr(v, "state_dict"):
                state_dict[k] = v.state_dict()
            else:
                state_dict[k] = v
        return state_dict

    def load_state_dict(
        self, state_dict: "Dict[str, Any]", strict: "bool" = True
    ) -> "None":
        """Load a state dict into the object.

        Parameters
        ----------
        state_dict:
            The state dict.
        strict:
            True to enforce that keys in `state_dict`
            match the keys in the object state dict,
            False otherwise.

        Raises
        ------
        RuntimeError
            If `strict` is True and keys in `state_dict`
            do not match the keys in the object state dict.

        """
        missing_keys = []
        for k in self._STATE_VARS:
            if k not in state_dict:
                missing_keys.append(k)
                continue
            v = state_dict[k]
            if hasattr(self.__dict__[k], "load_state_dict"):
                self.__dict__[k].load_state_dict(v)
            else:
                self.__dict__[k] = v
        if strict:
            error_msgs = []
            if missing_keys:
                error_msgs.append(f"Missing keys: {missing_keys}")
            unexpected_keys = list(state_dict.keys() - self._STATE_VARS)
            if unexpected_keys:
                error_msgs.append(f"Unexpected keys: {unexpected_keys}")
            if error_msgs:
                error_msg = "\n\t".join(error_msgs)
                raise RuntimeError(f"Invalid state dict:\n\t{error_msg}")


class FutureRef(Generic[_T]):
    """Reference to an object that does not exist yet."""

    def __init__(self, ref: "str") -> "None":
        """Initialize the object.

        Parameters
        ----------
        ref:
            The reference to the future object
            expressed as a string.

        """
        self.ref = ref

    def resolve(_self, **context: "Any") -> "_T":
        """Resolve the future reference based on the given context.

        Parameters
        ----------
        context:
            The context, i.e. a dict that maps names of the
            local variables that can be used for resolving
            the reference to the local variables themselves.

        Returns
        -------
            The resolved reference.

        Raises
        ------
        ValueError
            If an invalid future reference is given.

        """
        try:
            return eval(_self.ref, globals(), context)
        except Exception:
            raise ValueError(f"Invalid future reference: {_self.ref}")

    def __repr__(self) -> "str":
        return f"{type(self).__name__}(ref: {self.ref})"


def get_system_info() -> "Dict[str, Any]":
    """Return the system hardware and software information.

    Returns
    -------
        The system hardware and software information.

    """
    system_info = {}
    uname = platform.uname()
    system_info["platform"] = {
        "system": uname.system,
        "node_name": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
    }
    boot_timestamp = psutil.boot_time()
    boot_time = datetime.fromtimestamp(boot_timestamp)
    system_info["boot_time"] = boot_time.strftime("%d/%m/%Y %H:%M:%S")
    cpu_freq = psutil.cpu_freq()
    system_info["cpu"] = {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "min_freq": f"{cpu_freq.min:.2f}MHz",
        "max_freq": f"{cpu_freq.max:.2f}MHz",
        "current_freq": f"{cpu_freq.current:.2f}MHz",
        "usage_per_core": {
            f"{i}": f"{p}%"
            for i, p in enumerate(psutil.cpu_percent(percpu=True, interval=1))
        },
        "total_usage": f"{psutil.cpu_percent()}%",
    }
    virtual_memory = psutil.virtual_memory()
    system_info["memory"] = {
        "total": _format_sizeof(virtual_memory.total, "B", 1024),
        "available": _format_sizeof(virtual_memory.available, "B", 1024),
        "used": _format_sizeof(virtual_memory.used, "B", 1024),
        "percentage": f"{virtual_memory.percent}%",
    }
    system_info["gpu"] = {
        f"{gpu.id}": {
            "uuid": gpu.uuid,
            "name": gpu.name,
            "memory": {
                "total": f"{gpu.memoryTotal}MB",
                "available": f"{gpu.memoryFree}MB",
                "used": f"{gpu.memoryUsed}MB",
                "percentage": f"{gpu.load * 100}%",
            },
            "temperature": f"{gpu.temperature}C",
        }
        for gpu in GPUtil.getGPUs()
    }
    return system_info


def import_module(path: "str") -> "ModuleType":
    """Import a Python module at runtime.

    Parameters
    ----------
    path:
        The absolute or relative path to the module.

    Returns
    -------
        The imported module.

    """
    path = os.path.realpath(path)
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Adapted from:
# https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/python/ray/tune/logger.py#L748
def pretty_print(result: "Dict[str, Any]") -> "str":
    """Modified version of `ray.tune.logger.pretty_print`
    that preserves the key order.

    Parameters
    ----------
    result:
        The result.

    Returns
    -------
        The pretty-printed result.

    """
    result = result.copy()
    result.update(config=None)
    result.update(hist_stats=None)
    output = {k: v for k, v in result.items() if v is not None}
    cleaned = json.dumps(output, cls=SafeFallbackEncoder)
    return yaml.safe_dump(
        json.loads(cleaned), default_flow_style=False, sort_keys=False
    )


def singledispatchmethod(
    method: "Optional[Callable[..., _T]]" = None,
    use_weakrefs: "bool" = True,
) -> "Union[Callable[..., _T], Callable[[Callable[..., _T]], Callable[..., _T]]]":
    """Modified version of `functools.singledispatch`
    that handles instance/class methods correctly.

    Parameters
    ----------
    method:
        The instance/class method.
    use_weakrefs:
        True to use weak references, False otherwise.
        Useful, for example, to avoid pickling issues
        (some picklers such as `cloudpickle` fail to
        pickle weak reference objects).

    Returns
    -------
        The single-dispatched instance/class method.

    """

    def singledispatchmethod_fn(method: "Callable[..., _T]") -> Callable[..., _T]:
        if not use_weakrefs:
            import weakref

            WeakKeyDictionaryBackup = weakref.WeakKeyDictionary
            weakref.WeakKeyDictionary = dict
        try:
            dispatcher = singledispatch(method)

            def wrapper(*args: "Any", **kwargs: "Any") -> "_T":
                return dispatcher.dispatch(args[1].__class__)(*args, **kwargs)

            wrapper.register = dispatcher.register
            update_wrapper(wrapper, method)
            return wrapper
        finally:
            if not use_weakrefs:
                weakref.WeakKeyDictionary = WeakKeyDictionaryBackup

    if method:
        return singledispatchmethod_fn(method)

    return singledispatchmethod_fn


# Adapted from:
# https://github.com/tqdm/tqdm/blob/f3fb54eb161a9f5de13352e16a70a9960946605b/tqdm/std.py#L259
def _format_sizeof(
    num: "float",
    suffix: "str" = "",
    divisor: "float" = 1000.0,
) -> "str":
    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 999.5:
            if abs(num) < 99.95:
                if abs(num) < 9.995:
                    return f"{num:1.2f}{unit}{suffix}"
                return f"{num:2.1f}{unit}{suffix}"
            return f"{num:3.0f}{unit}{suffix}"
        num /= divisor
    return f"{num:3.1f}Y{suffix}"
