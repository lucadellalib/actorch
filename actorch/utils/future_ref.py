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

"""Future reference."""

from typing import Any, Generic, TypeVar


__all__ = [
    "FutureRef",
]


_T = TypeVar("_T")


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
