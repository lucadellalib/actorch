# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Matplotlib progress plotter."""

import os
from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import ndarray

from actorch import resources
from actorch.visualizer.plotters.plotter import Plotter


__all__ = [
    "MatplotlibPlotter",
]


class MatplotlibPlotter(Plotter):
    """Plotter based on Matplotlib backend."""

    @classmethod
    # override
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        parser = super().get_default_parser(**parser_kwargs)
        parser.add_argument(
            "--figsize",
            nargs=2,
            default=(7.5, 6.0),
            type=float,
            help="figure size",
        )
        parser.add_argument(
            "-f",
            "--format",
            default="png",
            help="output image format",
        )
        parser.add_argument(
            "-u",
            "--usetex",
            action="store_true",
            help="render text with LaTeX",
        )
        parser.add_argument(
            "--style",
            default=resources.get("styles/default-style.mplstyle"),
            help="absolute or relative path to a Matplotlib style file or name of one of Matplotlib built-in styles",
            dest="style_filepath_or_name",
        )
        return parser

    # override
    def _plot_traces(
        self,
        traces: "Dict[str, Tuple[ndarray, ndarray, ndarray]]",
        x_name: "str",
        y_name: "str",
        output_filepath: "str",
        opacity: "float",
        figsize: "Tuple[float, float]" = (7.5, 6.0),
        format: "str" = "png",
        usetex: "bool" = False,
        style_filepath_or_name: "str" = "classic",
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
        output_filepath:
            The absolute path to the output file.
        x_name:
            The name of the series to plot along the x-axis.
        y_name:
            The name of the series to plot along the y-axis.
        opacity:
            The confidence interval opacity.
        figsize:
            The figure size.
        format:
            The output image format.
        usetex:
            True to render text with LaTeX, False otherwise.
        style_filepath_or_name:
            The absolute or relative path to a Matplotlib style file or the name of one
            of Matplotlib built-in styles
            (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

        """
        output_filepath = f"{output_filepath}.{format}"
        if os.path.isfile(style_filepath_or_name):
            style_filepath_or_name = os.path.realpath(style_filepath_or_name)
        if usetex:
            x_name = x_name.replace("_", "\_")  # noqa: W605
            y_name = y_name.replace("_", "\_")  # noqa: W605
        with plt.style.context(style_filepath_or_name):
            rc("text", usetex=usetex)
            fig = plt.figure(figsize=figsize)
            for trial_name, (x, mean, shift) in traces.items():
                # Mean
                label = (
                    trial_name.replace("_", "\_")  # noqa: W605
                    if usetex
                    else trial_name
                )
                plt.plot(x, mean, label=label)
                # Confidence interval
                plt.fill_between(
                    x,
                    y1=mean - shift,
                    y2=mean + shift,
                    alpha=opacity,
                )
            plt.legend()
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            fig.tight_layout()
            plt.savefig(output_filepath, bbox_inches="tight")
            plt.close()
