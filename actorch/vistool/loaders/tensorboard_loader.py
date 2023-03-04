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

"""TensorBoard performance metrics loader."""

from argparse import ArgumentParser
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from actorch.vistool.loaders.loader import Loader


__all__ = [
    "TensorBoardLoader",
]


class TensorBoardLoader(Loader):
    """Load performance metrics from TensorBoard log files."""

    # override
    @classmethod
    def load(
        cls,
        input_dir: "str",
        search_pattern: "str" = ".*",
        exclude_names: "Optional[Sequence[str]]" = None,
        **kwargs: "Any",
    ) -> "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]":
        search_pattern = (
            f"(?=.*events\\.out\\.tfevents.*)(?={search_pattern})"  # noqa: W605
        )
        return super().load(input_dir, search_pattern, exclude_names, **kwargs)

    # override
    @classmethod
    def _load_metrics(cls, file: "str", **kwargs: "Any") -> "Dict[str, ndarray]":
        metrics = {}
        event_accumulator = EventAccumulator(file).Reload()
        reference_x = np.empty(0)  # Series with maximum length
        for tag in event_accumulator.Tags()["scalars"]:
            x, y = [], []
            for scalar in event_accumulator.Scalars(tag):
                x.append(float(scalar.step))
                y.append(float(scalar.value))
            x, y = np.array(x), np.array(y)
            if len(x) > len(reference_x):
                reference_x = x
            metrics[tag] = x, y
        for tag, (x, y) in metrics.items():
            # Some data series may be shorter because the corresponding values were logged
            # less frequently, therefore they need to be extended to the maximum length
            y_with_gaps = np.full_like(reference_x, np.nan)
            y_with_gaps[np.digitize(x, reference_x) - 1] = y
            metrics[tag] = y_with_gaps
        return metrics

    # override
    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        parser_kwargs.setdefault("description", "Load TensorBoard log files")
        return super().get_default_parser(**parser_kwargs)
