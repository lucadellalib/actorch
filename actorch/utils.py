# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Common utilities."""

import importlib
import json
import os
import platform
from datetime import datetime
from types import ModuleType
from typing import Any, Dict

import GPUtil
import psutil
import yaml
from ray.tune.logger import SafeFallbackEncoder


__all__ = [
    "get_system_info",
    "import_module",
    "pretty_print",
]


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
# https://github.com/tqdm/tqdm/blob/f3fb54eb161a9f5de13352e16a70a9960946605b/tqdm/std.py#L259
def _format_sizeof(
    num: "float",
    suffix: "str" = "",
    divisor: "float" = 1000.0,
) -> "str":
    """Format a number with SI Order of Magnitude prefixes.

    Parameters
    ----------
    num:
        The number.
    suffix:
        The post-postfix.
    divisor:
        The divisor between prefixes.

    Returns
    -------
        The number with Order of Magnitude SI unit postfix.

    """
    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 999.5:
            if abs(num) < 99.95:
                if abs(num) < 9.995:
                    return f"{num:1.2f}{unit}{suffix}"
                return f"{num:2.1f}{unit}{suffix}"
            return f"{num:3.0f}{unit}{suffix}"
        num /= divisor
    return f"{num:3.1f}Y{suffix}"


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
