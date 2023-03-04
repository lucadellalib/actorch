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

"""Preconditioner."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from torch import nn
from torch.optim import Optimizer


__all__ = [
    "Preconditioner",
]


class Preconditioner(ABC, Optimizer):
    """Preconditioner that modifies in-place the
    parameter gradients of a module.

    """

    # override
    def __init__(self, module: "nn.Module") -> "None":
        """Initialize the object.

        Parameters
        ----------
        module:
            The module to precondition.

        """
        self.module = module

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(module: {self.module})"

    # override
    def state_dict(self, include_module: "bool" = True) -> "Dict[str, Any]":
        """Return the preconditioner state dict.

        Parameters
        ----------
        include_module:
            True to include the module state dict,
            False otherwise.

        Returns
        -------
            The preconditioner state dict.

        """
        state_dict = super().state_dict()
        if include_module:
            state_dict["state"]["module"] = self.module.state_dict()
        else:
            del state_dict["state"]["module"]
        return state_dict

    # override
    def load_state_dict(self, state_dict: "Dict[str, Any]") -> "None":
        state_dict = {k: v for k, v in state_dict.items()}  # Copy
        module_state = state_dict["state"].pop("module", None)
        if module_state:
            self.module.load_state_dict(module_state)
        super().load_state_dict(state_dict)

    # override
    @abstractmethod
    def step(self) -> "None":
        raise NotImplementedError

    # override
    def zero_grad(self, set_to_none: "bool" = True) -> "None":
        raise NotImplementedError
