# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Setup script."""

import os
import re
import subprocess
import sys

from setuptools import Extension, find_packages, setup


_ROOT_DIRPATH = os.path.dirname(os.path.realpath(__file__))

_WINDOWS_DEPENDENCIES = {
    "torch": "https://download.pytorch.org/whl/cpu/torch_stable.html",
}


def _parse_requirements(requirement_filepath):
    requirement_filepath = os.path.realpath(requirement_filepath)
    is_windows = sys.platform in ["cygwin", "win32", "windows"]
    requirements = []
    with open(requirement_filepath, encoding="utf-8") as f:
        for requirement in f:
            # Ignore lines with `-f` flag
            if requirement.split(" ")[0] == "-f":
                continue
            if not is_windows:
                requirements.append(requirement)
                continue
            package_name = re.split("[=|<>!~]", requirement)[0]
            if package_name not in _WINDOWS_DEPENDENCIES:
                requirements.append(requirement)
                continue
            # Windows-specific requirement
            url = _WINDOWS_DEPENDENCIES[package_name]
            return_code = subprocess.call(
                ["pip", "install", requirement, "-f", url, "--no-cache-dir"]
            )
            if return_code != 0:
                raise RuntimeError(f"{requirement} installation failed")
    return requirements


with open(os.path.join(_ROOT_DIRPATH, "actorch", "version.py")) as f:
    tmp = {}
    exec(f.read(), tmp)
    _VERSION = tmp["VERSION"]
    del tmp

with open(os.path.join(_ROOT_DIRPATH, "README.md"), encoding="utf-8") as f:
    _README = f.read()

_REQUIREMENTS = _parse_requirements(
    os.path.join(_ROOT_DIRPATH, "requirements", "requirements.txt")
)

_REQUIREMENTS_DEV = _parse_requirements(
    os.path.join(_ROOT_DIRPATH, "requirements", "requirements-dev.txt")
)

_REQUIREMENTS_TEST = _parse_requirements(
    os.path.join(_ROOT_DIRPATH, "requirements", "requirements-test.txt")
)


setup(
    name="actorch",
    version=_VERSION,
    description="Deep reinforcement learning framework for fast prototyping based on PyTorch",
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Luca Della Libera",
    author_email="luca.dellalib@gmail.com",
    url="https://github.com/lucadellalib/actorch",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "actorch.buffers.utils",
            sources=["actorch/buffers/utils.c"],
        ),
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=["Deep reinforcement learning", "PyTorch"],
    platforms=["OS Independent"],
    include_package_data=True,
    install_requires=_REQUIREMENTS,
    entry_points={"console_scripts": ["actorch=actorch.__main__:main"]},
    extras_require={"dev": _REQUIREMENTS_DEV, "test": _REQUIREMENTS_TEST},
    python_requires=">=3.7",
)
