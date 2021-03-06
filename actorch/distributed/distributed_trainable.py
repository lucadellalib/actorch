# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Distributed trainable."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union

import ray
from ray.tune import PlacementGroupFactory
from ray.tune.trainable import DistributedTrainable as TuneDistributedTrainable
from ray.util.placement_group import get_current_placement_group, remove_placement_group


__all__ = [
    "DistributedTrainable",
]


class DistributedTrainable(ABC, TuneDistributedTrainable):
    """Distributed Ray Tune trainable with configurable resource requirements.

    Derived classes must implement `step`, `save_checkpoint` and `load_checkpoint`
    (see https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-trainable-class-api).

    """

    class Config(dict):
        """Keyword arguments expected in the configuration received by `setup`."""

        def __init__(
            self,
            bundles: "Optional[Sequence[Dict[str, Union[int, float]]]]" = None,
            placement_strategy: "str" = "PACK",
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

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}({self.config})"

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
