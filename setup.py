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

"""Setup script."""

import os
import re
import subprocess
import sys


_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

_WINDOWS_DEPENDENCIES = {
    "torch": "https://download.pytorch.org/whl/torch_stable.html",
}

_IS_WINDOWS = sys.platform in ["cygwin", "win32", "windows"]


def _preinstall_requirement(requirement, options=None):
    args = ["pip", "install", requirement, *(options or [])]
    return_code = subprocess.call(args)
    if return_code != 0:
        raise RuntimeError(f"{requirement} installation failed")


def _parse_requirements(requirements_file):
    requirements_file = os.path.realpath(requirements_file)
    requirements = []
    with open(requirements_file, encoding="utf-8") as f:
        for requirement in f:
            # Ignore lines with `-f` flag
            if requirement.split(" ")[0] == "-f":
                continue
            if not _IS_WINDOWS:
                requirements.append(requirement)
                continue
            package_name = re.split("[=|<>!~]", requirement)[0]
            if package_name not in _WINDOWS_DEPENDENCIES:
                requirements.append(requirement)
                continue
            # Windows-specific requirement
            url = _WINDOWS_DEPENDENCIES[package_name]
            _preinstall_requirement(requirement, options=["-f", url])
    return requirements


with open(os.path.join(_ROOT_DIR, "actorch", "version.py")) as f:
    tmp = {}
    exec(f.read(), tmp)
    _VERSION = tmp["VERSION"]
    del tmp

with open(os.path.join(_ROOT_DIR, "README.md"), encoding="utf-8") as f:
    _README = f.read()

_REQUIREMENTS_SETUP = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-setup.txt")
)

_REQUIREMENTS = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements.txt")
)

_REQUIREMENTS_VISTOOL = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-vistool.txt")
)

_REQUIREMENTS_TEST = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-test.txt")
)

_REQUIREMENTS_DEV = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-dev.txt")
)

# Manually preinstall setup requirements since build system specification in
# pyproject.toml is not reliable. For example, when NumPy is preinstalled,
# NumPy extensions are compiled with the latest compatible NumPy version
# rather than the one available on the system. If the two NumPy versions
# do not match, a runtime error is raised
for requirement in _REQUIREMENTS_SETUP:
    _preinstall_requirement(requirement)


import numpy as np  # noqa: E402
from setuptools import Extension, find_packages, setup  # noqa: E402


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
            include_dirs=[np.get_include()],
        ),
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    license="Apache License 2.0",
    keywords=["Deep reinforcement learning", "PyTorch"],
    platforms=["OS Independent"],
    include_package_data=True,
    install_requires=_REQUIREMENTS,
    entry_points={"console_scripts": ["actorch=actorch.__main__:main"]},
    extras_require={
        "vistool": _REQUIREMENTS_VISTOOL,
        "test": _REQUIREMENTS_TEST,
        "dev": _REQUIREMENTS_DEV,
        "all": list(
            set(_REQUIREMENTS_VISTOOL + _REQUIREMENTS_TEST + _REQUIREMENTS_DEV)
        ),
    },
    python_requires=">=3.6",
)
