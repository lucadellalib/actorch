# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Runner utilities."""

import importlib
import os
import platform
from datetime import datetime
from types import ModuleType
from typing import Any, Dict

import GPUtil
import psutil

from actorch.utils import normalize_byte_size


__all__ = [
    "get_system_info",
    "import_module",
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
        "min_freq": f"{cpu_freq.min:.2f} MHz",
        "max_freq": f"{cpu_freq.max:.2f} MHz",
        "current_freq": f"{cpu_freq.current:.2f} MHz",
        "usage_per_core": {
            f"{i}": f"{p}%"
            for i, p in enumerate(psutil.cpu_percent(percpu=True, interval=1))
        },
        "total_usage": f"{psutil.cpu_percent()}%",
    }
    virtual_memory = psutil.virtual_memory()
    system_info["memory"] = {
        "total": normalize_byte_size(virtual_memory.total),
        "available": normalize_byte_size(virtual_memory.available),
        "used": normalize_byte_size(virtual_memory.used),
        "percentage": f"{virtual_memory.percent}%",
    }
    system_info["gpu"] = {
        f"{gpu.id}": {
            "uuid": gpu.uuid,
            "name": gpu.name,
            "memory": {
                "total": f"{gpu.memoryTotal} MB",
                "available": f"{gpu.memoryFree} MB",
                "used": f"{gpu.memoryUsed} MB",
                "percentage": f"{gpu.load * 100}%",
            },
            "temperature": f"{gpu.temperature} C",
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
