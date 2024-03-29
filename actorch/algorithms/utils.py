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

"""Algorithm utilities."""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple, Type

import ray
import ray.train.torch  # Fix missing train.torch attribute
import torch
from ray import train
from torch import Tensor
from torch import distributed as dist
from torch.nn import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel


__all__ = [
    "count_params",
    "freeze_params",
    "init_mock_train_session",
    "normalize_",
    "prepare_model",
    "sync_polyak_",
]


def normalize_(input, dim: "int" = 0, mask: "Optional[Tensor]" = None) -> "Tensor":
    """Normalize a tensor along a dimension, by subtracting the mean
    and then dividing by the standard deviation (in-place).

    Parameters
    ----------
    input:
        The tensor.
    dim:
        The dimension.
    mask:
        The boolean tensor indicating which elements
        are valid (True) and which are not (False).
        Default to ``torch.ones_like(input, dtype=torch.bool)``.

    Returns
    -------
        The normalized tensor.

    Examples
    --------
    >>> import torch
    >>>
    >>> from actorch.algorithms.utils import normalize_
    >>>
    >>>
    >>> input = torch.randn(2,3)
    >>> mask = torch.rand(2, 3) > 0.5
    >>> output = normalize_(input, mask=mask)

    """
    if mask is None:
        mask = torch.ones_like(input, dtype=torch.bool)
    else:
        input[~mask] = 0.0
    length = mask.sum(dim=dim, keepdim=True)
    input_mean = input.sum(dim=dim, keepdim=True) / length
    input -= input_mean
    input *= mask
    input_stddev = (
        ((input**2).sum(dim=dim, keepdim=True) / length).sqrt().clamp(min=1e-6)
    )
    input /= input_stddev
    input *= mask
    return input


def count_params(module: "Module") -> "Tuple[int, int]":
    """Return the number of trainable and non-trainable
    parameters in `module`.

    Parameters
    ----------
    module:
        The module.

    Returns
    -------
        - The number of trainable parameters;
        - the number of non-trainable parameters.

    Examples
    --------
    >>> import torch
    >>>
    >>> from actorch.algorithms.utils import count_params
    >>>
    >>>
    >>> model = torch.nn.Linear(4, 2)
    >>> trainable_count, non_trainable_count = count_params(model)

    """
    trainable_count, non_trainable_count = 0, 0
    for param in module.parameters():
        if param.requires_grad:
            trainable_count += param.numel()
        else:
            non_trainable_count += param.numel()
    for buffer in module.buffers():
        non_trainable_count += buffer.numel()
    return trainable_count, non_trainable_count


@contextmanager
def freeze_params(*modules: "Module") -> "Iterator[None]":
    """Context manager that stops the gradient
    flow in `modules`.

    Parameters
    ----------
    modules:
        The modules.

    Examples
    --------
    >>> import torch
    >>>
    >>> from actorch.algorithms.utils import freeze_params
    >>>
    >>>
    >>> policy_model = torch.nn.Linear(4, 2)
    >>> value_model = torch.nn.Linear(2, 1)
    >>> input = torch.randn(3, 4)
    >>> with freeze_params(value_model):
    ...     action = policy_model(input)
    ...     loss = -value_model(action).mean()
    >>> loss.backward()

    """
    params = [
        param
        for module in modules
        for param in module.parameters()
        if param.requires_grad
    ]
    try:
        for param in params:
            param.requires_grad = False
        yield
    finally:
        for param in params:
            param.requires_grad = True


def sync_polyak_(
    source_module: "Module",
    target_module: "Module",
    polyak_weight: "float" = 0.001,
) -> "Module":
    """Synchronize a source module with a target module
    through Polyak averaging (in-place).

    For each `target_parameter` in `target_module`,
    for each `source_parameter` in `source_module`:
    `target_parameter` =
        (1 - `polyak_weight`) * `target_parameter`
        + `polyak_weight` * `source_parameter`.

    Parameters
    ----------
    source_module:
        The source module.
    target_module:
        The target module.
    polyak_weight:
        The Polyak weight.

    Returns
    -------
        The synchronized target module.

    Raises
    ------
    ValueError
        If `polyak_weight` is not in the interval [0, 1].

    References
    ----------
    .. [1] B. T. Polyak and A. B. Juditsky.
           "Acceleration of Stochastic Approximation by Averaging".
           In: SIAM Journal on Control and Optimization. 1992, pp. 838-855.
           URL: https://doi.org/10.1137/0330046

    Examples
    --------
    >>> import torch
    >>>
    >>> from actorch.algorithms.utils import sync_polyak_
    >>>
    >>>
    >>> model = torch.nn.Linear(4, 2)
    >>> target_model = torch.nn.Linear(4, 2)
    >>> sync_polyak_(model, target_model, polyak_weight=0.001)

    """
    if polyak_weight < 0.0 or polyak_weight > 1.0:
        raise ValueError(
            f"`polyak_weight` ({polyak_weight}) must be in the interval [0, 1]"
        )
    if polyak_weight == 0.0:
        return target_module
    if polyak_weight == 1.0:
        target_module.load_state_dict(source_module.state_dict())
        return target_module
    # Update parameters
    target_params = target_module.parameters()
    source_params = source_module.parameters()
    with torch.no_grad():
        for target_param, source_param in zip(target_params, source_params):
            # Update in-place
            target_param *= (1 - polyak_weight) / polyak_weight
            target_param += source_param
            target_param *= polyak_weight
    # Update buffers
    target_buffers = target_module.buffers()
    source_buffers = source_module.buffers()
    for target_buffer, source_buffer in zip(target_buffers, source_buffers):
        # Update in-place
        target_buffer *= (1 - polyak_weight) / polyak_weight
        target_buffer += source_buffer
        target_buffer *= polyak_weight
    return target_module


def init_mock_train_session() -> "None":
    """Modified version of `ray.train.session.init_session` that
    initializes a mock PyTorch Ray Train session.

    Useful, for example, to enable the use of PyTorch Ray Train objects
    and functions without instantiating a `ray.train.Trainer`.

    """
    # If a session already exists, do nothing
    if train.session.get_session():
        return
    train.session.init_session(None, 0, 0, 1)
    if dist.is_available() and dist.is_initialized():
        session = train.session.get_session()
        session.world_rank = dist.get_rank()
        session.local_rank = (ray.get_gpu_ids() or [None])[0]
        session.world_size = dist.get_world_size()


def prepare_model(
    model: "Module",
    move_to_device: "bool" = True,
    dp_cls: "Optional[Type[DataParallel]]" = DataParallel,
    dp_kwargs: "Optional[Dict[str, Any]]" = None,
    ddp_cls: "Optional[Type[DistributedDataParallel]]" = DistributedDataParallel,
    ddp_kwargs: "Optional[Dict[str, Any]]" = None,
) -> "Module":
    """Modified version of `ray.train.torch.prepare_model` that additionally
    wraps the model in a subclass of `torch.nn.parallel.DataParallel` if at
    least 2 GPUs are available for each process.

    Parameters
    ----------
    model:
        The model.
    move_to_device:
        True to move the model to the correct device, False otherwise.
    dp_cls:
        The subclass of `torch.nn.parallel.DataParallel` in which to wrap
        the model if at least 2 GPUs are available for each process.
        If None, wrapping is not performed.
    dp_kwargs:
        The data parallel initialization keyword arguments
        (ignored if `dp_cls` is None or not applicable).
        Default to ``{}``.
    ddp_cls:
        The subclass of `torch.nn.parallel.DistributedDataParallel` in which
        to wrap the model if at least 2 processes have been initialized.
        If None, wrapping is not performed.
    ddp_kwargs:
        The distributed data parallel initialization keyword arguments
        (ignored if `ddp_cls` is None or not applicable).
        Default to ``{}``.

    Returns
    -------
        The prepared model.

    """
    prepared_model = train.torch.prepare_model(
        model, move_to_device, ddp_cls is not None, ddp_kwargs
    )
    if train.session.get_accelerator(train.torch.TorchAccelerator).amp_is_enabled:
        # Patch forward
        if isinstance(prepared_model, DistributedDataParallel):
            prepared_model.module.forward = torch.cuda.amp.autocast()(
                prepared_model.module._unwrapped_forward
            )
        else:
            prepared_model.forward = torch.cuda.amp.autocast()(
                prepared_model._unwrapped_forward
            )
    if isinstance(prepared_model, DistributedDataParallel):
        prepared_model.__class__ = ddp_cls
    available_gpu_ids = ray.get_gpu_ids()
    if dp_cls and len(available_gpu_ids) > 1:
        dp_kwargs = dp_kwargs or {}
        dp_kwargs = {"device_ids": available_gpu_ids, **dp_kwargs}
        if isinstance(prepared_model, ddp_cls):
            prepared_model.module = dp_cls(prepared_model.module, **dp_kwargs)
        else:
            prepared_model = dp_cls(prepared_model, **dp_kwargs)
    return prepared_model
