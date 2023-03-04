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

"""Common utilities."""

import importlib
import json
import os
import platform
from datetime import datetime
from functools import singledispatch, update_wrapper
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import GPUtil
import psutil
import yaml
from ray.tune.utils.util import SafeFallbackEncoder


__all__ = [
    "get_system_info",
    "import_module",
    "pretty_print",
    "singledispatchmethod",
]


_T = TypeVar("_T")


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
