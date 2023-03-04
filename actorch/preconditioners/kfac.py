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

"""Kronecker-factored approximate curvature preconditioner."""

# Adapted from:
# https://github.com/gpauloski/kfac_pytorch/blob/ee8c1ec78db66b3063adf27e3290c7329e6b8a67/kfac/utils.py
# https://github.com/gpauloski/kfac_pytorch/blob/ee8c1ec78db66b3063adf27e3290c7329e6b8a67/kfac/kfac_preconditioner.py

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.distributions.kl import _Match as Match

from actorch.preconditioners.preconditioner import Preconditioner
from actorch.utils import CheckpointableMixin


__all__ = [
    "KFAC",
]


class KFACModule(ABC, CheckpointableMixin, nn.Module):
    """Wrap a module to make it compatible with K-FAC preconditioner."""

    _STATE_VARS = ["a", "g", "_A", "_G", "_dA", "_dG", "_QA", "_QG"]  # override

    # override
    def __init__(self, module: "nn.Module") -> "None":
        """Initialize the object.

        Parameters
        ----------
        module:
            The module.

        """
        super().__init__()
        self.module = module
        self.a = None  # Module input saved in the forward pass
        self.g = None  # Module gradient with respect to the output saved in the backward pass
        self._A, self._G = None, None  # Fisher factors A and G
        self._dA, self._dG = None, None  # Eigenvalues of A and G
        self._QA, self._QG = None, None  # Eigenvectors of A and G

    # override
    def state_dict(
        self,
        destination: "Optional[Dict[str, Tensor]]" = None,
        prefix: "str" = "",
        keep_vars: "bool" = False,
    ) -> "Dict[str, Optional[Tensor]]":
        state_dict = super().state_dict()
        if destination is None:
            destination = {}
        for k, v in state_dict.items():
            destination[prefix + k] = v if (keep_vars or v is None) else v.detach()
        return destination

    def update_AG(self, decay: "float") -> "None":
        """Update Fisher factors A and G.

        Parameters
        ----------
        decay:
            The exponential moving average decay rate.

        """
        A, G = self._compute_A(self.a), self._compute_G(self.g)
        if self._A is None:
            self._A, self._G = (
                A.new_ones(A.shape[0]).diag(),
                G.new_ones(G.shape[0]).diag(),
            )
            self._dA, self._dG = A.new_zeros(A.shape[0]), G.new_zeros(G.shape[0])
            self._QA, self._QG = A.new_zeros(A.shape), G.new_zeros(G.shape)
        self._update_exp_moving_average(self._A, A, decay)
        self._update_exp_moving_average(self._G, G, decay)

    def update_eigen_AG(self, epsilon: "float") -> "None":
        """Update eigenvalues and eigenvectors of A and G.

        Parameters
        ----------
        epsilon:
            The term added to the denominators to improve numerical stability.

        """
        self._dA, self._QA = torch.linalg.eigh(self._A, UPLO="U")
        self._dG, self._QG = torch.linalg.eigh(self._G, UPLO="U")
        self._dA *= (self._dA > epsilon).type_as(self._dA)
        self._dG *= (self._dG > epsilon).type_as(self._dA)

    def get_preconditioned_grad(self, damping: "float") -> "Tensor":
        """Return the module preconditioned flat gradient.

        Parameters
        ----------
        damping:
            The Tikhonov damping parameter.

        Returns
        -------
            The module preconditioned flat gradient.

        """
        v1 = self._QG.t() @ self.grad @ self._QA
        v2 = v1 / (self._dG[:, None, ...] * self._dA[None] + damping)
        v = self._QG @ v2 @ self._QA.t()
        return v

    def _update_exp_moving_average(
        self, current: "Tensor", new: "Tensor", weight: "float"
    ) -> "None":
        if weight == 1.0:
            return
        current *= weight / (1 - weight)
        current += new
        current *= 1 - weight

    @property
    @abstractmethod
    def grad(self) -> "Tensor":
        """Return the module flat gradient.

        Returns
        -------
            The module flat gradient.

        """
        raise NotImplementedError

    @grad.setter
    @abstractmethod
    def grad(self, value: "Tensor") -> "None":
        """Set the module flat gradient.

        Parameters
        ----------
        value:
            The module flat gradient.

        """
        raise NotImplementedError

    @abstractmethod
    def _compute_A(self, a: "Tensor") -> "Tensor":
        """Compute Fisher factor A.

        Parameters
        ----------
        a:
            The module input.

        Returns
        -------
            The Fisher factor A.

        """
        raise NotImplementedError

    @abstractmethod
    def _compute_G(self, g: "Tensor") -> "Tensor":
        """Compute Fisher factor G.

        Parameters
        ----------
        g:
            The module gradient with respect to the output.

        Returns
        -------
            The Fisher factor G.

        """
        raise NotImplementedError


class KFACLinear(KFACModule):
    """K-FAC linear layer wrapper."""

    # override
    @property
    def grad(self) -> "Tensor":
        weight_grad = self.module.weight.grad.flatten(start_dim=1)
        if self.module.bias is not None:
            bias_grad = self.module.bias.grad[..., None]
            return torch.cat([weight_grad, bias_grad], dim=1)
        return weight_grad

    # override
    @grad.setter
    def grad(self, value: "Tensor") -> "None":
        if self.module.bias is not None:
            weight_grad = value[:, :-1].reshape_as(self.module.weight)
            bias_grad = value[:, -1:].reshape_as(self.module.bias)
            self.module.bias.grad.copy_(bias_grad)
        else:
            weight_grad = value.reshape_as(self.module.weight)
        self.module.weight.grad.copy_(weight_grad)

    # override
    def _compute_A(self, a: "Tensor") -> "Tensor":
        # Shape of a: [batch_size, in_features]
        if self.module.bias is not None:
            a = torch.cat([a, a.new_ones(a.shape[0], 1)], dim=1)
        return a.t() @ (a / a.shape[0])

    # override
    def _compute_G(self, g: "Tensor") -> "Tensor":
        # Shape of g: [batch_size, out_features]
        return g.t() @ (g / g.shape[0])


class KFACConvNd(KFACModule):
    """K-FAC N-D convolutional layer wrapper.

    References
    ----------
    .. [1] R. Grosse and J. Martens.
           "A Kronecker-factored approximate Fisher matrix for convolution layers".
           In: ICML. 2016, pp. 573-582.
           URL: https://arxiv.org/abs/1602.01407

    """

    # override
    @property
    def grad(self) -> "Tensor":
        weight_grad = self.module.weight.grad.flatten(start_dim=1)
        if self.module.bias is not None:
            bias_grad = self.module.bias.grad[..., None]
            return torch.cat([weight_grad, bias_grad], dim=1)
        return weight_grad

    # override
    @grad.setter
    def grad(self, value: "Tensor") -> "None":
        if self.module.bias is not None:
            weight_grad = value[:, :-1].reshape_as(self.module.weight)
            bias_grad = value[:, -1:].reshape_as(self.module.bias)
            self.module.bias.grad.copy_(bias_grad)
        else:
            weight_grad = value.reshape_as(self.module.weight)
        self.module.weight.grad.copy_(weight_grad)

    # override
    def _compute_A(self, a: "Tensor") -> "Tensor":
        # Shape of a: [batch_size, in_channels, *in_shape]
        a = self._extract_patches(a)
        spatial_size = a.shape[1:-1].numel()
        a = a.reshape(-1, a.shape[-1])
        if self.module.bias is not None:
            a = torch.cat([a, a.new_ones(a.shape[0], 1)], dim=1)
        a /= spatial_size
        return a.t() @ (a / a.shape[0])

    # override
    def _compute_G(self, g: "Tensor") -> "Tensor":
        # Shape of g: [batch_size, out_channels, *out_shape]
        spatial_size = g.shape[2:].numel()
        g = g.movedim(1, -1)
        g = g.contiguous()
        g = g.reshape(-1, g.shape[-1]) * spatial_size
        return g.t() @ (g / g.shape[0])

    def _extract_patches(self, x: "Tensor") -> "Tensor":
        # Shape of x: [batch_size, in_channels, *in_shape]
        kernel_size = self.module.kernel_size
        stride = self.module.stride
        padding = self.module.padding
        if sum(padding) > 0:
            x = F.pad(x, tuple([x for x in reversed(padding) for _ in range(2)]))
        for i, (size, step) in enumerate(zip(kernel_size, stride)):
            x = x.unfold(2 + i, size, step)
        dim = len(kernel_size) + 1
        x = x.movedim(1, dim).contiguous()
        x = x.reshape(*x.shape[0:dim], x.shape[dim:].numel())
        return x


class KFAC(Preconditioner):
    """Kronecker-factored approximate curvature preconditioner.

    Precondition the gradients of a module with a layer-wise Fisher
    information matrix approximation.

    References
    ----------
    .. [1] J. Martens and R. Grosse.
           "Optimizing Neural Networks with Kronecker-factored Approximate Curvature".
           In: ICML. 2015, pp. 2408-2417.
           URL: https://arxiv.org/abs/1503.05671

    Examples
    --------
    >>> import torch
    >>>
    >>> from actorch.preconditioners import KFAC
    >>>
    >>>
    >>> model = torch.nn.Linear(64, 32)
    >>> criterion = torch.nn.MSELoss()
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    >>> preconditioner = KFAC(model)
    >>> data, target = torch.rand(10, 64), torch.rand(10, 32)
    >>> optimizer.zero_grad()
    >>> output = model(data)
    >>> loss = criterion(output, target)
    >>> loss.backward()
    >>> preconditioner.step()
    >>> optimizer.step()

    """

    _SUPPORTED_MODULE_TYPES = {
        nn.Linear: KFACLinear,
        nn.Conv1d: KFACConvNd,
        nn.Conv2d: KFACConvNd,
        nn.Conv3d: KFACConvNd,
    }

    # override
    def __init__(
        self,
        module: "nn.Module",
        lr: "float" = 0.1,
        factor_decay: "float" = 0.95,
        damping: "float" = 0.001,
        kl_clip: "float" = 0.001,
        factor_update_freq: "int" = 10,
        kfac_update_freq: "int" = 100,
        grad_scaler: "Optional[GradScaler]" = None,
        epsilon: "float" = 1e-6,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        module:
            The module to perform K-FAC updates on.
        lr:
            The learning rate.
        factor_decay:
            The exponential moving average decay rate for the Kronecker factors.
        damping:
            The Tikhonov damping parameter.
        kl_clip:
            The Kullback-Leibler divergence clipping parameter for gradient scaling.
        factor_update_freq:
            Compute the Kronecker factors and update their exponential moving average
            every `factor_update_freq` iterations.
        kfac_update_freq:
            Apply gradient preconditioning every `kfac_update_freq` iterations.
        grad_scaler:
            The gradient scaler for automatic mixed precision training.
        epsilon:
            The term added to the denominators to improve numerical stability.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if lr <= 0.0:
            raise ValueError(f"`lr` ({lr}) must be in the interval (0, inf)")
        if factor_decay > 1.0 or factor_decay <= 0.0:
            raise ValueError(
                f"`factor_decay` ({factor_decay}) must be in the interval (0, 1]"
            )
        if damping <= 0.0:
            raise ValueError(f"`damping` ({damping}) must be in the interval (0, inf)")
        if kl_clip <= 0.0:
            raise ValueError(f"`kl_clip` ({kl_clip}) must be in the interval (0, inf)")
        if factor_update_freq < 1 or not float(factor_update_freq).is_integer():
            raise ValueError(
                f"`factor_update_freq` ({factor_update_freq}) must be in the integer interval [1, inf)"
            )
        if kfac_update_freq < 1 or not float(kfac_update_freq).is_integer():
            raise ValueError(
                f"`kfac_update_freq` ({kfac_update_freq}) must be in the integer interval [1, inf)"
            )
        if epsilon <= 0.0:
            raise ValueError(f"`epsilon` ({epsilon}) must be in the interval (0, inf)")
        factor_update_freq = int(factor_update_freq)
        kfac_update_freq = int(kfac_update_freq)
        if kfac_update_freq % factor_update_freq != 0:
            warnings.warn(
                f"`kfac_update_freq` ({kfac_update_freq}) should be a "
                f"multiple of `factor_update_freq` ({factor_update_freq})"
            )

        defaults = {
            "lr": lr,
            "factor_decay": factor_decay,
            "damping": damping,
            "kl_clip": kl_clip,
            "factor_update_freq": factor_update_freq,
            "kfac_update_freq": kfac_update_freq,
            "epsilon": epsilon,
        }
        # K-FAC does not register any parameters
        super(Preconditioner, self).__init__([torch.empty(1)], defaults)
        self.state["module"] = module
        self.state["grad_scaler"] = grad_scaler
        self._register_submodules(module)

    @property
    def module(self) -> "nn.Module":
        return self.state.get("module")

    @property
    def lr(self) -> "float":
        return self.param_groups[0]["lr"]

    @property
    def factor_decay(self) -> "float":
        return self.param_groups[0]["factor_decay"]

    @property
    def damping(self) -> "float":
        return self.param_groups[0]["damping"]

    @property
    def kl_clip(self) -> "float":
        return self.param_groups[0]["kl_clip"]

    @property
    def factor_update_freq(self) -> "int":
        return self.param_groups[0]["factor_update_freq"]

    @property
    def kfac_update_freq(self) -> "int":
        return self.param_groups[0]["kfac_update_freq"]

    @property
    def grad_scaler(self) -> "GradScaler":
        return self.state.get("grad_scaler")

    @property
    def epsilon(self) -> "float":
        return self.param_groups[0]["epsilon"]

    # override
    def state_dict(
        self,
        include_module: "bool" = True,
        include_grad_scaler: "bool" = True,
    ) -> "Dict[str, Any]":
        """Return the preconditioner state dict.

        Parameters
        ----------
        include_module:
            True to include the module state dict,
            False otherwise.
        include_grad_scaler:
            True to include the gradient scaler state dict,
            False otherwise.

        Returns
        -------
            The preconditioner state dict.

        """
        state_dict = super().state_dict(include_module)
        if include_grad_scaler:
            state_dict["state"]["grad_scaler"] = None
            if self.grad_scaler:
                state_dict["state"]["grad_scaler"] = self.grad_scaler.state_dict()
        else:
            del state_dict["state"]["grad_scaler"]
        state_dict["state"]["_modules"] = [
            module.state_dict() for module in self._modules.values()
        ]
        return state_dict

    # override
    def load_state_dict(self, state_dict: "Dict[str, Any]") -> "None":
        state_dict = {k: v for k, v in state_dict.items()}  # Copy
        module_state = state_dict["state"].pop("module", None)
        if module_state:
            self.module.load_state_dict(module_state)
        grad_scaler_state = state_dict["state"].pop("grad_scaler", None)
        if self.grad_scaler and grad_scaler_state:
            self.grad_scaler.load_state_dict(grad_scaler_state)
        module_states = state_dict["state"].pop("_modules")
        if len(module_states) != len(self._modules):
            raise ValueError(
                f"The number of loaded submodule states ({len(module_states)}) must be "
                f"equal to the number of submodules in the module ({len(self._modules)})"
            )
        for module, state in zip(self._modules.values(), module_states):
            module.load_state_dict(state)
        # Workaround to prevent self.module, self.grad_scaler
        # and self._modules from being deleted
        state_backup = {k: v for k, v in self.state.items()}  # Copy
        super(Preconditioner, self).load_state_dict(state_dict)
        self.state.update(state_backup)

    # override
    @torch.no_grad()
    def step(self) -> "None":
        update_factor = self._steps % self.factor_update_freq == 0
        update_eigen = self._steps % self.kfac_update_freq == 0

        vs = []
        vg_sum = 0.0
        for module in self._modules.values():
            if update_factor:
                module.update_AG(self.factor_decay)

            if update_eigen:
                module.update_eigen_AG(self.epsilon)

            v = module.get_preconditioned_grad(self.damping)
            vs.append(v)
            vg_sum += (v * module.grad * self.lr**2).sum()

        nu = (self.kl_clip / vg_sum).sqrt().clip(max=1.0).item()

        for module, v in zip(self._modules.values(), vs):
            v *= nu
            module.grad = v

        self._steps += 1

    @property
    def _modules(self) -> "Dict[nn.Module, KFACModule]":
        if "_modules" not in self.state:
            self.state["_modules"] = {}
        return self.state["_modules"]

    @property
    def _steps(self) -> "int":
        return self.state.get("_steps", 0)

    @_steps.setter
    def _steps(self, value: "int") -> "None":
        self.state["_steps"] = value

    def _register_submodules(self, module: "nn.Module") -> "None":
        cls = type(module)
        matches = [Match(t) for t in self._SUPPORTED_MODULE_TYPES if issubclass(cls, t)]
        if matches:
            match = min(matches).types[0]
            self._modules[module] = self._SUPPORTED_MODULE_TYPES[match](module)
            module.register_forward_pre_hook(self._save_input)
            module.register_full_backward_hook(self._save_grad_output)
            return
        submodules = list(module.children())
        if not submodules and any(
            x.requires_grad for x in module.parameters()
        ):  # Is a leaf module with trainable parameters
            raise NotImplementedError(
                f"Unsupported module type: `{cls.__module__}.{cls.__name__}`"
            )
        for submodule in submodules:
            self._register_submodules(submodule)

    @torch.no_grad()
    def _save_input(self, module: "nn.Module", input: "Tensor") -> "None":
        if not module.training:
            return
        if self._steps % self.factor_update_freq == 0:
            self._modules[module].a = input[0]

    @torch.no_grad()
    def _save_grad_output(
        self,
        module: "nn.Module",
        grad_input: "Tensor",
        grad_output: "Tensor",
    ) -> "None":
        if not module.training:
            return
        if self._steps % self.factor_update_freq == 0:
            g = grad_output[0]
            if self.grad_scaler is not None:
                g /= self.grad_scaler.get_scale()
            self._modules[module].g = g

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(module: {self.module}, "
            f"lr: {self.lr}, "
            f"factor_decay: {self.factor_decay}, "
            f"damping: {self.damping}, "
            f"kl_clip: {self.kl_clip}, "
            f"factor_update_freq: {self.factor_update_freq}, "
            f"kfac_update_freq: {self.kfac_update_freq}, "
            f"grad_scaler: {self.grad_scaler}, "
            f"epsilon: {self.epsilon})"
        )
