#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Visualize experiment progress."""

from argparse import ArgumentParser, Namespace
from typing import Any

from actorch.visualizer.loaders import CSVLoader, TensorBoardLoader
from actorch.visualizer.plotters import MatplotlibPlotter, PlotlyPlotter


__all__ = [
    "Visualize",
]


class Visualize:
    """Visualize experiment progress."""

    _LOADER_TYPES = {
        "csv": CSVLoader,
        "tensorboard": TensorBoardLoader,
    }

    _PLOTTER_TYPES = {
        "matplotlib": MatplotlibPlotter,
        "plotly": PlotlyPlotter,
    }

    def main(self, args: "Namespace") -> "None":
        """Script entry point.

        Parameters
        ----------
        args:
            The command line arguments obtained from an
            `argparse.ArgumentParser` (e.g. the default
            parser returned by `get_default_parser`).

        """
        input_format = getattr(args, "input_format", None)
        if input_format not in self._LOADER_TYPES:
            raise ValueError(
                f"`input_format` ({input_format}) must be in {list(self._LOADER_TYPES)}"
            )
        backend = getattr(args, "backend", None)
        if backend not in self._PLOTTER_TYPES:
            raise ValueError(
                f"`backend` ({backend}) must be in {list(self._PLOTTER_TYPES)}"
            )
        loader = self._LOADER_TYPES[input_format]()
        plotter = self._PLOTTER_TYPES[backend]()
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
        parser_kwargs.setdefault("description", "Visualize experiment progress")
        parser = ArgumentParser(**parser_kwargs)
        subparsers = parser.add_subparsers(
            title="backend", dest="backend", required=True
        )
        for backend, plotter_cls in cls._PLOTTER_TYPES.items():
            backend_parser = plotter_cls.get_default_parser()
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
            for input_format, loader_cls in cls._LOADER_TYPES.items():
                input_format_parser = loader_cls.get_default_parser()
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

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}()"


def _main() -> "None":
    try:
        parser = Visualize.get_default_parser()
        args = parser.parse_args()
        print("------------------- Start ------------------")
        visualize = Visualize()
        visualize.main(args)
        print("------------------- Done -------------------")
    except KeyboardInterrupt:
        print("---- Exiting early (Keyboard Interrupt) ----")


if __name__ == "__main__":
    _main()
