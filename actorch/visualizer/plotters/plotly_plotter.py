# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Plotly progress plotter."""

import os
from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import plotly
import yaml
from numpy import ndarray
from plotly import graph_objects as go

from actorch import resources
from actorch.visualizer.plotters.plotter import Plotter


__all__ = [
    "PlotlyPlotter",
]


class PlotlyPlotter(Plotter):
    """Plotter based on Plotly backend."""

    # override
    def _plot_traces(
        self,
        traces: "Dict[str, Tuple[ndarray, ndarray, ndarray]]",
        x_name: "str",
        y_name: "str",
        output_filepath: "str",
        opacity: "float",
        offline: "bool" = False,
        template_filepath_or_name: "str" = "plotly",
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
        offline:
            True to generate self-contained plots that can be displayed offline, False otherwise.
        template_filepath_or_name:
            The absolute or relative path to a Plotly template file or the name of one
            of Plotly built-in templates (see https://plotly.com/python/templates/).

        """
        output_filepath = f"{output_filepath}.html"
        template = template_filepath_or_name
        if os.path.isfile(template_filepath_or_name):
            with open(os.path.realpath(template_filepath_or_name)) as f:
                template = go.layout.Template(yaml.safe_load(f))
        fig = go.Figure()
        for i, (trial_name, (x, mean, shift)) in enumerate(traces.items()):
            # Mean
            color = plotly.colors.DEFAULT_PLOTLY_COLORS[
                i % len(plotly.colors.DEFAULT_PLOTLY_COLORS)
            ]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    line={"color": color},
                    legendgroup=i,
                    mode="lines",
                    name=f"{trial_name}",
                )
            )
            # Confidence interval
            color = color.replace("rgb", "rgba").replace(")", f", {opacity})")
            scatter_kwargs = {
                "legendgroup": i,
                "line": {"width": 0},
                "marker": {"color": color},
                "mode": "lines",
                "name": f"{trial_name} CI",
                "showlegend": False,
            }
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean + shift,
                    **scatter_kwargs,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean - shift,
                    fill="tonexty",
                    fillcolor=color,
                    **scatter_kwargs,
                )
            )
        fig.update_layout(
            xaxis={"title": x_name},
            yaxis={"title": y_name},
            legend={"traceorder": "normal"},
            template=template,
        )
        fig.write_html(output_filepath, include_plotlyjs=True if offline else "cdn")

    # override
    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        parser_kwargs.setdefault("description", "Plot using Plotly backend")
        parser = super().get_default_parser(**parser_kwargs)
        parser.add_argument(
            "--offline",
            action="store_true",
            help="generate self-contained plots that can be displayed offline",
        )
        parser.add_argument(
            "--template",
            default=resources.get("templates/default-template.yml"),
            help="absolute or relative path to a Plotly template file or name of one of Plotly built-in templates",
            dest="template_filepath_or_name",
        )
        return parser
