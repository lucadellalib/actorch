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

"""Performance metrics plotter."""

import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import numpy as np
from numpy import ndarray
from scipy import stats
from tqdm import tqdm


__all__ = [
    "Plotter",
]


class Plotter(ABC):
    """Plot performance metrics."""

    @classmethod
    def plot(
        cls,
        metrics: "Dict[str, Dict[str, Tuple[ndarray, ndarray]]]",
        output_dir: "str" = "plots",
        x_name: "str" = "training_iteration",
        smoothing: "float" = 0.0,
        confidence_level: "float" = 0.9545,
        opacity: "float" = 0.15,
        **kwargs: "Any",
    ) -> "None":
        """Plot performance metrics.

        Parameters
        ----------
        metrics:
            The performance metrics, i.e. a dict that maps series names to dicts
            that map trial names to pairs with the following values:
            - the series mean (over sampled series);
            - the series standard deviation (over sampled series).
        output_dir:
            The absolute or relative path to the output directory.
        x_name:
            The name of the series to plot along the x-axis.
        smoothing:
            The smoothing factor.
        confidence_level:
            The confidence level assuming normally-distributed data.
        opacity:
            The confidence interval opacity.

        Raises
        ------
        ValueError
            If an invalid argument value is given.
        RuntimeError
            If `x_name` is not in the performance metrics to plot.

        """
        for key, value in {
            "smoothing": smoothing,
            "confidence_level": confidence_level,
            "opacity": opacity,
        }.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"`{key}` ({value}) must be in the interval [0, 1]")
        output_dir = os.path.realpath(output_dir)
        if x_name not in metrics:
            raise RuntimeError(
                f"`x_name` ({x_name}) must be in the performance metrics to plot"
            )
        x_trials = metrics.pop(x_name)
        x_name = x_name.replace("/", "_").replace("_", " ").capitalize()
        num_stddevs = float(stats.norm.ppf((confidence_level + 1) / 2))
        with tqdm(total=len(metrics)) as progress_bar:
            for y_name, y_trials in metrics.items():
                progress_bar.set_description(f"Plotting {y_name}")
                output_subdir = os.path.join(
                    output_dir,
                    os.path.dirname(y_name),
                )
                os.makedirs(output_subdir, exist_ok=True)
                y_name = y_name.replace("/", "_")
                output_file = os.path.join(output_subdir, y_name)
                traces = {}
                for trial_name, (mean, stddev) in y_trials.items():
                    x = x_trials[trial_name][0]
                    mask = np.isfinite(x) & np.isfinite(mean)
                    stddev[~np.isfinite(stddev)] = 0.0
                    mean[mask] = cls._exponential_moving_average(mean[mask], smoothing)
                    shift = stddev.copy()
                    shift[mask] = num_stddevs * np.sqrt(
                        cls._exponential_moving_average(stddev[mask] ** 2, smoothing)
                    )
                    traces[trial_name] = (x[mask], mean[mask], shift[mask])
                cls._plot_traces(
                    traces,
                    x_name,
                    y_name.replace("_", " ").capitalize(),
                    output_file,
                    opacity,
                    **kwargs,
                )
                progress_bar.update()

    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        """Return a default command line argument parser for `plot`.

        Parameters
        ----------
        parser_kwargs:
            The parser initialization keyword arguments.

        Returns
        -------
            The default command line argument parser.

        """
        parser_kwargs.setdefault("description", "Plot performance metrics")
        parser = ArgumentParser(**parser_kwargs)
        parser.add_argument(
            "--output-dir",
            default="plots",
            help="absolute or relative path to the output directory",
        )
        parser.add_argument(
            "-x",
            "--x-name",
            default="training_iteration",
            help="name of the series to plot along the x-axis",
        )
        parser.add_argument(
            "-s",
            "--smoothing",
            default=0.0,
            type=float,
            help="smoothing factor",
        )
        parser.add_argument(
            "-c",
            "--confidence-level",
            default=0.9545,
            type=float,
            help="confidence level assuming normally-distributed data",
        )
        parser.add_argument(
            "-o",
            "--opacity",
            default=0.15,
            type=float,
            help="confidence interval opacity",
        )
        return parser

    @classmethod
    def _exponential_moving_average(
        cls, values: "ndarray", smoothing: "float" = 0.0
    ) -> "ndarray":
        # This is the same smoothing algorithm used by TensorBoard
        # (see https://github.com/tensorflow/tensorboard/blob/2.6/tensorboard/components/vz_line_chart2/line-chart.ts#L714)
        last = values[0]
        output = []
        for value in values:
            if np.isfinite(value):
                last = last * smoothing + (1 - smoothing) * value
                output.append(last)
        return np.array(output)

    @classmethod
    @abstractmethod
    def _plot_traces(
        cls,
        traces: "Dict[str, Tuple[ndarray, ndarray, ndarray]]",
        x_name: "str",
        y_name: "str",
        output_file: "str",
        opacity: "float",
        **kwargs: "Any",
    ) -> "None":
        """Plot traces.

        Parameters
        ----------
        traces:
            The traces, i.e. a dict that maps trial names
            to triplets with the following values:
            - the series to plot along the x-axis;
            - the mean (over sampled series) of the series to plot along the y-axis;
            - the shift (over sampled series) from the mean of the series to plot along the y-axis.
        output_file:
            The absolute path to the output file.
        x_name:
            The name of the series to plot along the x-axis.
        y_name:
            The name of the series to plot along the y-axis.
        opacity:
            The confidence interval opacity.

        """
        raise NotImplementedError
