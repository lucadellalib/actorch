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

"""Performance metrics loader."""

import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from tqdm import tqdm


__all__ = [
    "Loader",
]


class Loader(ABC):
    """Load performance metrics from log files stored in a directory."""

    @classmethod
    def load(
        cls,
        input_dir: "str",
        search_pattern: "str" = ".*",
        exclude_names: "Optional[Sequence[str]]" = None,
        **kwargs: "Any",
    ) -> "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]":
        """Retrieve log files from an input directory and load the performance metrics.

        Parameters
        ----------
        input_dir:
            The absolute or relative path to the input directory.
        search_pattern:
            The regex pattern to search for in the paths to the log files.
        exclude_names:
            The names of the data series in the log files that must not be loaded.
            Default to ``[]``.

        Returns
        -------
            The performance metrics, i.e. a dict that maps series names to dicts
            that map trial names to pairs with the following values:
            - the series mean (over sampled series);
            - the series standard deviation (over sampled series).

        """
        input_dir = os.path.realpath(input_dir)
        search_regex = re.compile(search_pattern)
        exclude_names = exclude_names or []
        files = sorted(
            filter(
                lambda x: search_regex.match(x),
                [
                    os.path.join(subdir, filename)
                    for subdir, _, filenames in os.walk(input_dir)
                    for filename in filenames
                ],
            )
        )
        all_metrics: "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]" = defaultdict(dict)
        with tqdm(total=len(files)) as progress_bar:
            for file in files:
                head, tail = os.path.split(file)
                # Assume log files are stored in a directory named after the trial
                trial_name = os.path.basename(head)
                progress_bar.set_description(
                    f"Loading {os.path.join(trial_name, tail)}"
                )
                metrics = cls._load_metrics(file, **kwargs)
                for series_name in metrics:
                    if series_name in exclude_names:
                        continue
                    if series_name.endswith("/stddev"):
                        continue
                    mean = metrics[series_name]
                    stddev = np.zeros_like(mean)
                    if series_name.endswith("/mean"):
                        stddev_key = series_name.replace("/mean", "/stddev")
                        if stddev_key in metrics:
                            series_name = series_name.replace("/mean", "")
                            stddev = metrics[stddev_key]
                    all_metrics[series_name][trial_name] = (mean, stddev)
                progress_bar.update()
        return all_metrics

    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        """Return a default command line argument parser for `load`.

        Parameters
        ----------
        parser_kwargs:
            The parser initialization keyword arguments.

        Returns
        -------
            The default command line argument parser.

        """
        parser_kwargs.setdefault("description", "Load performance metrics")
        parser = ArgumentParser(**parser_kwargs)
        parser.add_argument(
            "--input-dir",
            default=".",
            help="absolute or relative path to the input directory",
        )
        parser.add_argument(
            "-p",
            "--search-pattern",
            default=".*",
            help="regex pattern to search for in the paths to the log files",
        )
        parser.add_argument(
            "-e",
            "--exclude-names",
            nargs="+",
            default=[],
            help="names of the data series in the log files that must not be loaded",
        )
        return parser

    @classmethod
    @abstractmethod
    def _load_metrics(cls, file: "str", **kwargs: "Any") -> "Dict[str, ndarray]":
        """Load performance metrics from a log file.

        Parameters
        ----------
        file:
            The absolute path to the log file.

        Returns
        -------
            The performance metrics, i.e. a dict that maps names of the
            data series in the log file to the data series themselves.

        """
        raise NotImplementedError
