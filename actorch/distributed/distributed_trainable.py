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

"""Distributed trainable."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import ray
from ray.tune import PlacementGroupFactory
from ray.tune.sample import Domain
from ray.tune.trainable import DistributedTrainable as TuneDistributedTrainable
from ray.util.placement_group import get_current_placement_group, remove_placement_group


__all__ = [
    "DistributedTrainable",
    "Tunable",
]


_T = TypeVar("_T")

_GridSearch = Dict[str, List[_T]]

Tunable = Union[_T, Domain, _GridSearch[_T]]
"""Ray Tune tunable argument."""


class DistributedTrainable(ABC, TuneDistributedTrainable):
    """Distributed Ray Tune trainable with configurable resource requirements.

    Derived classes must implement `step`, `save_checkpoint` and `load_checkpoint`
    (see https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-trainable-class-api).

    """

    class Config(dict):
        """Keyword arguments expected in the configuration received by `setup`."""

        def __init__(
            self,
            bundles: "Tunable[Optional[Sequence[Dict[str, Union[int, float]]]]]" = None,
            placement_strategy: "Tunable[str]" = "PACK",
        ) -> "None":
            """Initialize the object.

            Parameters
            ----------
            bundles:
                The bundles, i.e. a sequence of dicts that map resource
                names to their corresponding requested quantities.
                Default to ``[{"CPU": 1}]``.
            placement_strategy:
                The placement strategy
                (see https://docs.ray.io/en/latest/ray-core/placement-group.html).

            """
            super().__init__(
                bundles=bundles or [{"CPU": 1}],
                placement_strategy=placement_strategy,
            )

    # override
    @classmethod
    def default_resource_request(
        cls,
        config: "Dict[str, Any]",
    ) -> "PlacementGroupFactory":
        config = DistributedTrainable.Config(**config)
        bundles = config["bundles"]
        placement_strategy = config["placement_strategy"]
        return PlacementGroupFactory(bundles, placement_strategy)

    # override
    @classmethod
    def resource_help(cls, config: "Dict[str, Any]") -> "str":
        return (
            "The following keyword arguments should be given to configure the resources:"
            + "\n"
            "    bundles:" + "\n"
            "        The bundles, i.e. a sequence of dicts that map resource" + "\n"
            "        names to their corresponding requested quantities." + "\n"
            "        Default to ``[{}]``." + "\n"
            "    placement_strategy:" + "\n"
            "        The placement strategy." + "\n"
            "        (see https://docs.ray.io/en/latest/ray-core/placement-group.html)."
        )

    # override
    def setup(self, config: "Dict[str, Any]") -> "None":
        config = DistributedTrainable.Config(**config)
        self.config = deepcopy(config)
        self.bundles = config["bundles"]
        self.placement_strategy = config["placement_strategy"]
        self._placement_group = None

        # Setup placement group if necessary
        if not get_current_placement_group():
            default_resource_request = DistributedTrainable.default_resource_request(
                config
            )
            self._placement_group = default_resource_request()
            ray.get(self._placement_group.ready())

    # override
    def cleanup(self) -> "None":
        if self._placement_group:
            remove_placement_group(self._placement_group)

    # override
    @abstractmethod
    def step(self) -> "Dict[str, Any]":
        raise NotImplementedError

    # override
    @abstractmethod
    def save_checkpoint(
        self, tmp_checkpoint_dir: "str"
    ) -> "Union[str, Dict[str, Any]]":
        raise NotImplementedError

    # override
    @abstractmethod
    def load_checkpoint(self, checkpoint: "Union[str, Dict[str, Any]]") -> "None":
        raise NotImplementedError
