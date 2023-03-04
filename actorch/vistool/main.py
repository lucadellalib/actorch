#!/usr/bin/env python3

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

"""VisTool main script."""

from argparse import ArgumentParser, Namespace
from typing import Any

from actorch.vistool.loaders import CSVLoader, TensorBoardLoader
from actorch.vistool.plotters import MatplotlibPlotter, PlotlyPlotter


__all__ = [
    "VisTool",
]


class VisTool:
    """VisTool main script."""

    _LOADERS = {
        "csv": CSVLoader,
        "tensorboard": TensorBoardLoader,
    }

    _PLOTTERS = {
        "matplotlib": MatplotlibPlotter,
        "plotly": PlotlyPlotter,
    }

    @classmethod
    def main(cls, args: "Namespace") -> "None":
        """Script entry point.

        Parameters
        ----------
        args:
            The command line arguments obtained from an
            `argparse.ArgumentParser` (e.g. the default
            parser returned by `get_default_parser`).

        """
        input_format = getattr(args, "input_format", None)
        if input_format not in cls._LOADERS:
            raise ValueError(
                f"`input_format` ({input_format}) must be in {list(cls._LOADERS)}"
            )
        backend = getattr(args, "backend", None)
        if backend not in cls._PLOTTERS:
            raise ValueError(f"`backend` ({backend}) must be in {list(cls._PLOTTERS)}")
        loader = cls._LOADERS[input_format]()
        plotter = cls._PLOTTERS[backend]()
        del args.input_format, args.backend
        kwargs = vars(args)
        data = loader.load(**kwargs)
        plotter.plot(data, **kwargs)

    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        """Return a default command line argument parser for `main`.

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
        subparsers = parser.add_subparsers(
            title="backend", dest="backend", required=True
        )
        for backend, plotter in cls._PLOTTERS.items():
            backend_parser = plotter.get_default_parser()
            backend_help = backend_description = backend_parser.description
            if backend_help:
                backend_help = f"{backend_help[0].lower()}{backend_help[1:]}"
            backend_input_format_parser = subparsers.add_parser(
                backend, description=backend_description, help=backend_help
            )
            backend_input_format_subparsers = (
                backend_input_format_parser.add_subparsers(
                    title="input format", dest="input_format", required=True
                )
            )
            for input_format, loader in cls._LOADERS.items():
                input_format_parser = loader.get_default_parser()
                input_format_help = (
                    input_format_description
                ) = input_format_parser.description
                if input_format_help:
                    input_format_help = (
                        f"{input_format_help[0].lower()}{input_format_help[1:]}"
                    )
                    input_format_description += f" and {backend_help}"
                backend_input_format_subparsers.add_parser(
                    input_format,
                    description=input_format_description,
                    help=input_format_help,
                    parents=[
                        input_format_parser,
                        backend_parser,
                    ],
                    conflict_handler="resolve",
                )
        return parser


def main() -> "None":
    """VisTool entry point."""
    try:
        parser = VisTool.get_default_parser()
        args = parser.parse_args()
        print("------------------- Start ------------------")
        VisTool.main(args)
        print("------------------- Done -------------------")
    except KeyboardInterrupt:
        print("---- Exiting early (Keyboard Interrupt) ----")


if __name__ == "__main__":
    main()
