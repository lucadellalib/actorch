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

"""Network."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Size, Tensor, nn
from torch.distributions import Distribution, Independent, ReshapeTransform, constraints

from actorch.distributions import (
    CatDistribution,
    MaskedDistribution,
    TransformedDistribution,
)
from actorch.models import Model
from actorch.networks.normalizing_flow import NormalizingFlow
from actorch.networks.processors import Identity, Processor


__all__ = [
    "DistributionParametrization",
    "Network",
]


DistributionParametrization = Dict[
    str, Tuple[Dict[str, Tuple[int, ...]], Callable[[Dict[str, Tensor]], Tensor]]
]
"""A dict that maps names of a distribution initialization
arguments to pairs with the following values:
- a dict that maps parameters names to
  their corresponding event shapes;
- the parameter aggregation function. It receives as
  arguments the parameters and returns the corresponding
  distribution initialization argument.

"""


class Network(nn.Module):
    """Assist the creation of an `actorch.models.Model` and provide
    additional functionalities such as preprocessing of inputs,
    flattening of outputs, unflattening/flattening of states,
    and support for distributional output and normalizing flows.

         input           state
           |               |
    .______|_____.   ._____|_____.
    | preprocess |   | unflatten |
    |____________|   |___________|
           |               |
           |               |
        inputs           states
           |               |
    .______|_______________|_____.
    |            model           |
    |____________________________|
           |               |
           |               |
        outputs          states
           |               |
      .____|____.     .____|____.
      | flatten |     | flatten |
      |_________|     |_________|
           |               |
           |               |
        output           state

    """

    wrapped_model: "nn.Module"
    """The underlying wrapped model."""

    # override
    def __init__(
        self,
        preprocessors: "Dict[str, Processor]",
        model_builder: "Callable[..., Model]",
        distribution_builders: "Dict[str, Callable[..., Distribution]]",
        distribution_parametrizations: "Dict[str, DistributionParametrization]",
        model_config: "Optional[Dict[str, Any]]" = None,
        distribution_configs: "Optional[Dict[str, Dict[str, Any]]]" = None,
        normalizing_flows: "Optional[Dict[str, NormalizingFlow]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        preprocessors:
            The preprocessors, i.e. a dict that maps names of the
            input modalities to their corresponding preprocessors.
        model_builder:
            The model builder, i.e. a callable that receives keyword
            arguments from a model configuration and returns a model.
        distribution_builders:
            The distribution builders, i.e. a dict that maps names of
            the output modalities to their corresponding distribution
            builders. A distribution builder is a callable that receives
            keyword arguments from a distribution parametrization and a
            distribution configuration and returns a distribution.
        distribution_parametrizations:
            The distribution parametrizations, i.e. a dict that maps names
            of the output modalities to their corresponding distribution
            parametrizations.
        model_config:
            The model configuration.
            Arguments `in_shapes` and `out_shapes` are set internally.
            Default to ``{}``.
        distribution_configs:
            The distribution configurations, i.e. a dict that maps names
            of the output modalities to their corresponding distribution
            configurations.
            Arguments defined in `distribution_parametrizations` are set internally.
            Default to ``{}``.
        normalizing_flows:
            The normalizing flows, i.e. a dict that maps names of the
            output modalities to their corresponding normalizing flows.
            Default to ``{}``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        out_modality_names = distribution_builders.keys()
        if distribution_parametrizations.keys() != out_modality_names:
            raise ValueError(
                f"`distribution_parametrizations.keys()` ({list(distribution_parametrizations.keys())}) "
                f"must be equal to `distribution_builders.keys()` ({list(out_modality_names)})"
            )
        distribution_configs = distribution_configs or {}
        if any(k not in out_modality_names for k in distribution_configs):
            raise ValueError(
                f"`distribution_configs.keys()` ({list(distribution_configs.keys())}) "
                f"must be a subset of `distribution_builders.keys()` ({list(out_modality_names)})"
            )
        normalizing_flows = normalizing_flows or {}
        if any(k not in out_modality_names for k in normalizing_flows):
            raise ValueError(
                f"`normalizing_flows.keys()` ({list(normalizing_flows.keys())}) "
                f"must be a subset of `distribution_builders.keys()` ({list(out_modality_names)})"
            )
        super().__init__()
        self.preprocessors = preprocessors
        self.model_builder = model_builder
        self.distribution_builders = distribution_builders
        self.distribution_parametrizations = distribution_parametrizations
        self.model_config = model_config or {}
        self.distribution_configs = distribution_configs
        for k, v in normalizing_flows.items():
            normalizing_flows[k] = WrappedNormalizingFlow(v)
        self.normalizing_flows = nn.ModuleDict(normalizing_flows)
        model_in_shapes = {
            in_modality_name: preprocessor.out_shape
            for in_modality_name, preprocessor in self.preprocessors.items()
        }
        model_out_shapes = {}
        for out_modality_name in out_modality_names:
            distribution_parametrization = distribution_parametrizations[
                out_modality_name
            ]
            for params, _ in distribution_parametrization.values():
                for param_name, param_shape in params.items():
                    key = f"{out_modality_name}/{param_name}"
                    if key in model_out_shapes:
                        param_names = [
                            param_name
                            for param_name in params
                            for params, _ in distribution_parametrization.values()
                        ]
                        raise ValueError(
                            f"Parameter names related to the same "
                            f"distribution ({param_names}) must be unique"
                        )
                    model_out_shapes[key] = param_shape
        model = model_builder(model_in_shapes, model_out_shapes, **self.model_config)
        example_inputs, example_states, _ = model.get_example_inputs((2,))
        self._state_preprocessors, self._state_postprocessors = {}, {}
        for key, example_state in example_states.items():
            shape = example_state.shape[1:]
            self._state_preprocessors[key] = Identity(shape)
            self._state_postprocessors[key] = Identity(shape)
        self._output_preprocessors, self._output_postprocessors = {}, {}
        for key, model_out_shape in model_out_shapes.items():
            self._output_preprocessors[key] = Identity(model_out_shape)
            self._output_postprocessors[key] = Identity(model_out_shape)

        # Thin wrapper around the model that handles preprocessing of inputs,
        # flattening of outputs, and unflattening/flattening of states
        class WrappedModel(nn.Module):
            # override
            def __init__(self) -> "None":
                super().__init__()
                self.model = model

            # override
            def forward(
                _self,
                input: "Tensor",
                state: "Optional[Tensor]" = None,
                mask: "Optional[Tensor]" = None,
            ) -> "Tuple[Tensor, Optional[Tensor]]":
                model_inputs = self._unflatten(input, self.preprocessors)
                model_states = (
                    None
                    if state is None
                    else self._unflatten(state, self._state_preprocessors)
                )
                model_outputs, model_states = _self.model(
                    model_inputs, model_states, mask
                )
                output = self._flatten(model_outputs, self._output_postprocessors)
                state = (
                    self._flatten(model_states, self._state_postprocessors)
                    if model_states
                    else None
                )
                return output, state

        self.wrapped_model = WrappedModel()
        self._event_shapes: "Dict[str, Size]" = {}
        # Forward once to populate self._event_shapes
        with torch.no_grad():
            example_outputs, _ = model(example_inputs)
        output = self._flatten(example_outputs, self._output_postprocessors)
        self._cache = {"forward": (output, None)}
        _ = self.distribution
        # Reset cache
        self._cache = {}

    @property
    def distribution(self) -> "Distribution":
        """Return the predictive distribution, i.e. the conditional distribution
        of the flat output given the flat input, which is the concatenation of
        the flat distributions built for each output modality.

        Returns
        -------
            The predictive distribution.

        Raises
        ------
        RuntimeError
            If `forward` is not called at least once before
            accessing the property.

        """
        try:
            result = self._cache["distribution"]
        except KeyError:
            if "forward" not in self._cache:
                raise RuntimeError(
                    "`forward` must be called at least once "
                    "before accessing `distribution`"
                )
            output, mask = self._cache["forward"]
            model_outputs = self._unflatten(output, self._output_preprocessors)
            if mask is None:
                mask = torch.as_tensor(True)
            distributions = []
            for (
                out_modality_name,
                distribution_builder,
            ) in self.distribution_builders.items():
                # Build distribution
                distribution_parametrization = self.distribution_parametrizations[
                    out_modality_name
                ]
                kwargs = {
                    arg_name: aggregation_fn(
                        {
                            param_name: model_outputs[
                                f"{out_modality_name}/{param_name}"
                            ]
                            for param_name in params
                        }
                    )
                    for arg_name, (
                        params,
                        aggregation_fn,
                    ) in distribution_parametrization.items()
                }
                config = self.distribution_configs.get(out_modality_name, {})
                distribution = distribution_builder(**kwargs, **config)
                # Set event dimension
                batch_ndims = output.ndim - 1
                reinterpreted_batch_ndims = len(distribution.batch_shape) - batch_ndims
                if reinterpreted_batch_ndims > 0:
                    distribution = Independent(
                        distribution,
                        reinterpreted_batch_ndims,
                        distribution._validate_args,
                    )
                # Apply mask
                if not mask.all():
                    distribution = MaskedDistribution(
                        distribution, mask, distribution._validate_args
                    )
                # Apply normalizing flow
                try:
                    event_shape = self._event_shapes[out_modality_name]
                except KeyError:
                    event_shape = distribution.event_shape
                    if out_modality_name in self.normalizing_flows:
                        normalizing_flow = self.normalizing_flows[out_modality_name]
                        event_shape = torch.Size(
                            normalizing_flow.forward_shape(event_shape)
                        )
                    self._event_shapes[out_modality_name] = event_shape
                transforms = []
                if out_modality_name in self.normalizing_flows:
                    normalizing_flow = self.normalizing_flows[out_modality_name]
                    transforms.append(normalizing_flow)
                # Apply flatten transform
                if len(event_shape) > 1:
                    transforms.append(
                        ReshapeTransform(
                            event_shape, torch.Size((event_shape.numel(),))
                        )
                    )
                if transforms:
                    distribution = TransformedDistribution(
                        distribution, transforms, distribution._validate_args
                    ).reduced_dist
                distributions.append(distribution)
            result = (
                CatDistribution(
                    distributions,
                    validate_args=any(d._validate_args for d in distributions),
                )
                if len(distributions) > 1
                else distributions[0]
            )
            self._cache["distribution"] = result
        return result

    # override
    def forward(
        self,
        input: "Tensor",
        state: "Optional[Tensor]" = None,
        mask: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        """Forward pass.

        In the following, let `B = {B_1, ..., B_k}` denote the batch shape,
        `B_star` a subset of `B`, `I` the flat input event size, `S` the flat
        state event size, and `O` the flat output event size.

        Parameters
        ----------
        input:
            The flat input, shape: ``[*B, I]``.
        state:
            The flat state, shape: ``[*B_star, S]``.
        mask:
            The boolean tensor indicating which batch elements are
            valid (True) and which are not (False), shape: ``[*B]``.

        Returns
        -------
            - The flat output, shape: ``[*B, O]``;
            - the possibly updated flat state, shape: ``[*B_star, S]``.

        """
        output, state = self.wrapped_model(input, state, mask)
        self._cache["forward"] = (output, mask)
        # Invalidate distribution cache
        self._cache.pop("distribution", None)
        return output, state

    def _unflatten(
        self,
        input: "Tensor",
        processors: "Dict[str, Processor]",
    ) -> "Dict[str, Tensor]":
        outputs = {}
        start = stop = 0
        batch_shape = input.shape[:-1]
        for key, processor in processors.items():
            in_shape = processor.in_shape
            if len(processors) == 1 and not in_shape:
                batch_shape = input.shape
                input = input[..., None]
            stop += in_shape.numel()
            chunk = input[..., start:stop]
            outputs[key] = processor(chunk.reshape(batch_shape + in_shape)).float()
            start = stop
        return outputs

    def _flatten(
        self,
        inputs: "Dict[str, Tensor]",
        processors: "Dict[str, Processor]",
    ) -> "Tensor":
        outputs = []
        batch_shape = None
        for key, processor in processors.items():
            input = inputs[key]
            in_shape, out_shape = processor.in_shape, processor.out_shape
            if batch_shape is None:
                batch_shape = input.shape[: input.ndim - len(in_shape)]
            output = processor(input)
            if len(processors) == 1:
                return output.reshape(batch_shape + ((-1,) if out_shape else ()))
            outputs.append(output.reshape(*batch_shape, -1))
        return torch.cat(outputs, dim=-1)


class WrappedNormalizingFlow(NormalizingFlow):
    """Thin wrapper around a normalizing flow that handles
    attribute access when using parallelization strategies
    (e.g. distributed data parallel).

    """

    model: "NormalizingFlow"
    """The underlying normalizing flow."""

    # override
    def __init__(self, model: "NormalizingFlow") -> "None":
        """Initialize the object.

        Parameters
        ----------
        model:
            The normalizing flow to wrap.

        """
        super().__init__()
        self.model = model

    # override
    @property
    def is_constant_jacobian(self) -> "bool":
        if hasattr(self.model, "is_constant_jacobian"):
            return self.model.is_constant_jacobian
        return self.model.module.is_constant_jacobian

    # override
    @property
    def domain(self) -> "constraints.Constraint":
        if hasattr(self.model, "domain"):
            return self.model.domain
        return self.model.module.domain

    # override
    @property
    def codomain(self) -> "constraints.Constraint":
        if hasattr(self.model, "codomain"):
            return self.model.codomain
        return self.model.module.codomain

    # override
    def _call(self, x: "Tensor") -> "Tensor":
        return self.model(x)

    # override
    def _inverse(self, y: "Tensor") -> "Tensor":
        return self.model(y, method="_inverse")

    # override
    def log_abs_det_jacobian(self, x: "Tensor", y: "Tensor") -> "Tensor":
        return self.model(x, y, method="log_abs_det_jacobian")

    # override
    def forward_shape(self, shape: "Size") -> "Size":
        if hasattr(self.model, "forward_shape"):
            return self.model.forward_shape(shape)
        return self.model.module.forward_shape(shape)

    # override
    def inverse_shape(self, shape: "Size") -> "Size":
        if hasattr(self.model, "inverse_shape"):
            return self.model.inverse_shape(shape)
        return self.model.module.inverse_shape(shape)
