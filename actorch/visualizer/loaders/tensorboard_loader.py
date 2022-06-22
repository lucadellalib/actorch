# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""TensorBoard progress loader."""

from argparse import ArgumentParser
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from actorch.visualizer.loaders.loader import Loader


__all__ = [
    "TensorBoardLoader",
]


class TensorBoardLoader(Loader):
    """Load progress data from TensorBoard progress files."""

    # override
    def load(
        self,
        input_dirpath: "str",
        search_pattern: "str" = ".*",
        exclude_names: "Optional[Sequence[str]]" = None,
        **kwargs: "Any",
    ) -> "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]":
        search_pattern = f"(?=.*events\.out\.tfevents.*)(?={search_pattern})"
        return super().load(input_dirpath, search_pattern, exclude_names, **kwargs)

    # override
    def _load_data(self, filepath: "str", **kwargs: "Any") -> "Dict[str, ndarray]":
        data = {}
        event_accumulator = EventAccumulator(filepath).Reload()
        reference_x = np.empty(0)  # Series with maximum length
        for tag in event_accumulator.Tags()["scalars"]:
            x, y = [], []
            for scalar in event_accumulator.Scalars(tag):
                x.append(float(scalar.step))
                y.append(float(scalar.value))
            x, y = np.array(x), np.array(y)
            if len(x) > len(reference_x):
                reference_x = x
            data[tag] = x, y
        for tag, (x, y) in data.items():
            # Some data series may be shorter because the corresponding values were logged
            # less frequently, therefore they need to be extended to the maximum length
            y_with_gaps = np.full_like(reference_x, np.nan)
            y_with_gaps[np.digitize(x, reference_x) - 1] = y
            data[tag] = y_with_gaps
        return data

    # override
    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        parser_kwargs.setdefault("description", "Load TensorBoard progress files")
        return super().get_default_parser(**parser_kwargs)
