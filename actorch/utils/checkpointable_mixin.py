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

"""Checkpointable mixin."""

from typing import Any, Dict, Optional, Sequence, TypeVar


__all__ = [
    "CheckpointableMixin",
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
