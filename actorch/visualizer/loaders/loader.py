# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Progress loader."""

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
    """Load progress data from progress files stored in a directory."""

    @classmethod
    def load(
        cls,
        input_dirpath: "str",
        search_pattern: "str" = ".*",
        exclude_names: "Optional[Sequence[str]]" = None,
        **kwargs: "Any",
    ) -> "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]":
        """Retrieve progress files from an input directory and load the progress data.

        Parameters
        ----------
        input_dirpath:
            The absolute or relative path to the input directory.
        search_pattern:
            The regex pattern to search for in the paths to the progress files.
        exclude_names:
            The names of the data series in the progress files that must not be loaded.
            Default to ``[]``.

        Returns
        -------
            The progress data, i.e. a dict that maps series names to dicts
            that map trial names to pairs with the following values:
            - the series mean (over sampled series);
            - the series standard deviation (over sampled series).

        """
        input_dirpath = os.path.realpath(input_dirpath)
        search_regex = re.compile(search_pattern)
        exclude_names = exclude_names or []
        filepaths = sorted(
            filter(
                lambda x: search_regex.match(x),
                [
                    os.path.join(subdirpath, filename)
                    for subdirpath, _, filenames in os.walk(input_dirpath)
                    for filename in filenames
                ],
            )
        )
        all_data: "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]" = defaultdict(dict)
        with tqdm(total=len(filepaths)) as progress_bar:
            for filepath in filepaths:
                head, tail = os.path.split(filepath)
                # Assume progress files are stored in a directory named after the trial
                trial_name = os.path.basename(head)
                progress_bar.set_description(
                    f"Loading {os.path.join(trial_name, tail)}"
                )
                data = cls._load_data(filepath, **kwargs)
                for series_name in data:
                    if series_name in exclude_names:
                        continue
                    if series_name.endswith("/stddev"):
                        continue
                    mean = data[series_name]
                    stddev = np.zeros_like(mean)
                    if series_name.endswith("/mean"):
                        stddev_key = series_name.replace("/mean", "/stddev")
                        if stddev_key in data:
                            series_name = series_name.replace("/mean", "")
                            stddev = data[stddev_key]
                    all_data[series_name][trial_name] = (mean, stddev)
                progress_bar.update()
        return all_data

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
        parser_kwargs.setdefault("description", "Load progress data")
        parser = ArgumentParser(**parser_kwargs)
        parser.add_argument(
            "--input-dirpath",
            default=".",
            help="absolute or relative path to the input directory",
            metavar="input-dir",
        )
        parser.add_argument(
            "-p",
            "--search-pattern",
            default=".*",
            help="regex pattern to search for in the paths to the progress files",
        )
        parser.add_argument(
            "-e",
            "--exclude-names",
            nargs="+",
            default=[],
            help="names of the data series in the progress files that must not be loaded",
        )
        return parser

    @classmethod
    @abstractmethod
    def _load_data(cls, filepath: "str", **kwargs: "Any") -> "Dict[str, ndarray]":
        """Load progress data from a progress file.

        Parameters
        ----------
        filepath:
            The absolute path to the progress file.

        Returns
        -------
            The progress data, i.e. a dict that maps names of the data
            series in the progress file to the data series themselves.

        """
        raise NotImplementedError
