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

"""Resources."""


def get(path: "str") -> "str":
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
    import importlib
    import os

    subpackage_name, resource_name = f"{os.sep}{os.path.normpath(path)}".rsplit(
        os.sep, 1
    )
    package_name = __name__ + subpackage_name.replace(os.sep, ".")
    if package_name.endswith("."):
        package_name = package_name[:-1]
    with importlib.resources.path(package_name, resource_name) as realpath:
        return str(realpath)
