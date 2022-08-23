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

"""CSV progress loader."""

import csv
from argparse import ArgumentParser
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
    @classmethod
    def load(
        cls,
        input_dirpath: "str",
        search_pattern: "str" = ".*",
        exclude_names: "Optional[Sequence[str]]" = None,
        **kwargs: "Any",
    ) -> "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]":
        search_pattern = f"(?=.*\\.csv)(?={search_pattern})"  # noqa: W605
        return super().load(input_dirpath, search_pattern, exclude_names, **kwargs)

    # override
    @classmethod
    def _load_data(cls, filepath: "str", **kwargs: "Any") -> "Dict[str, ndarray]":
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

    # override
    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        parser_kwargs.setdefault("description", "Load CSV progress files")
        return super().get_default_parser(**parser_kwargs)
