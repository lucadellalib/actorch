# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""CSV loader."""

import csv
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray

from actorch.visualizer.loaders.loader import Loader


__all__ = [
    "CSVLoader",
]


class CSVLoader(Loader):
    """Load progress data from CSV progress files."""

    # override
    def load(
        self,
        input_dirpath: "str",
        search_pattern: "str" = ".*",
        exclude_names: "Optional[Sequence[str]]" = None,
        **kwargs: "Any",
    ) -> "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]":
        search_pattern = f"(?=.*\.csv)(?={search_pattern})"
        return super().load(input_dirpath, search_pattern, exclude_names, **kwargs)

    # override
    def _load_data(self, filepath: "str", **kwargs: "Any") -> "Dict[str, ndarray]":
        data = {}
        with open(filepath) as f:
            content = csv.reader(f)
            headers = list(next(content))
            bodies = list(zip(*[row for row in content]))
            for header, body in zip(headers, bodies):
                try:
                    # Replace empty strings with NaN
                    y_with_gaps = np.array(
                        list(map(lambda x: float(x) if not x == "" else np.nan, body))
                    )
                    if np.isfinite(y_with_gaps).any():
                        data[header] = y_with_gaps
                except ValueError:
                    pass
        return data
