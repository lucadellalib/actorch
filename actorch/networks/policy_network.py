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

"""Policy network."""

from typing import Any, Callable, Dict, Optional

from torch import Tensor
from torch.distributions import Distribution

from actorch.models import Model
from actorch.networks.network import DistributionParametrization, Network
from actorch.networks.normalizing_flow import NormalizingFlow
from actorch.networks.processors import Identity, Processor


__all__ = [
    "PolicyNetwork",
]


class PolicyNetwork(Network):
    """Network that implements a policy to generate predictions
    from the predictive distribution.

    ._________________________.
    | predictive distribution |
    |_________________________|
                 |
                 |
               sample
                 |
         ._______|_______.
         | prediction_fn |
         |       +       |
         |  postprocess  |
         |_______________|
                 |
                 |
             prediction

    """

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
        sample_fn: "Optional[Callable[[Distribution], Tensor]]" = None,
        prediction_fn: "Optional[Callable[[Tensor], Tensor]]" = None,
        postprocessors: "Optional[Dict[str, Processor]]" = None,
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
        sample_fn:
            The function that draws samples from the predictive distribution.
            It receives as an argument the predictive distribution and returns
            the corresponding sample.
            Default to ``lambda distribution: distribution.mean``.
        prediction_fn:
            The function that returns predictions from samples drawn from
            the predictive distribution. It receives as an argument a
            sample drawn from the predictive distribution and returns
            the corresponding prediction.
            Default to ``lambda sample: sample``.
        postprocessors:
            The postprocessors, i.e. a dict that maps names of the
            output modalities to their corresponding postprocessors.
            Default to ``{}``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__(
            preprocessors,
            model_builder,
            distribution_builders,
            distribution_parametrizations,
            model_config,
            distribution_configs,
            normalizing_flows,
        )
        out_modality_names = distribution_builders.keys()
        postprocessors = postprocessors or {}
        if any(k not in out_modality_names for k in postprocessors):
            raise ValueError(
                f"`postprocessors.keys()` ({list(postprocessors.keys())}) must be a "
                f"subset of `distribution_builders.keys()` ({list(out_modality_names)})"
            )
        if postprocessors:
            # Order matters
            postprocessors = {
                out_modality_name: postprocessors.get(
                    out_modality_name,
                    Identity(self._event_shapes[out_modality_name]),
                )
                for out_modality_name in out_modality_names
            }
        self.sample_fn = sample_fn or (lambda distribution: distribution.mean)
        self.prediction_fn = prediction_fn or (lambda sample: sample)
        self.postprocessors = postprocessors
        self._prediction_preprocessors = {}
        for out_modality_name, event_shape in self._event_shapes.items():
            self._prediction_preprocessors[out_modality_name] = Identity(event_shape)

    def predict(self, sample: "Optional[Tensor]" = None) -> "Tensor":
        """Return a prediction from a sample drawn from
        the predictive distribution.

        Parameters
        ----------
        sample:
            The sample.
            Default to the sample drawn through the given
            `sample_fn` initialization argument.

        Returns
        -------
            The prediction.

        """
        if sample is None:
            sample = self.sample_fn(self.distribution)
        prediction = self.prediction_fn(sample)
        if not self.postprocessors:
            return prediction
        prediction = self._unflatten(prediction, self._prediction_preprocessors)
        prediction = self._flatten(prediction, self.postprocessors)
        return prediction
