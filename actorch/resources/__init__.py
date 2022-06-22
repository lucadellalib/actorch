# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
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
