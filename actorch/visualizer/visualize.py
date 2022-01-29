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
            parser returned by method `get_default_parser`).

        """
        input_format = getattr(args, "input_format", None)
        if input_format not in self._LOADER_TYPES:
            raise ValueError(
                f"`input_format` ({input_format}) must be in {list(self._LOADER_TYPES.keys())}"
            )
        backend = getattr(args, "backend", None)
        if backend not in self._PLOTTER_TYPES:
            raise ValueError(
                f"`backend` ({backend}) must be in {list(self._PLOTTER_TYPES.keys())}"
            )
        loader = self._LOADER_TYPES[input_format]()
        plotter = self._PLOTTER_TYPES[backend]()
        del args.input_format, args.backend
        kwargs = vars(args)
        data = loader.load(**kwargs)
        plotter.plot(data, **kwargs)

    @classmethod
    def get_default_parser(cls, **parser_kwargs: "Any") -> "ArgumentParser":
        """Return a default command line argument parser for method `main`.

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
            backend_help = f"generate plots via {plotter_cls.__name__.replace('Plotter', '')} backend"
            backend_parser = subparsers.add_parser(backend, help=backend_help)
            backend_subparsers = backend_parser.add_subparsers(
                title="input format", dest="input_format", required=True
            )
            for input_format, loader_cls in cls._LOADER_TYPES.items():
                input_format_help = (
                    f"load {loader_cls.__name__.replace('Loader', '')} progress files"
                )
                backend_subparsers.add_parser(
                    input_format,
                    help=input_format_help,
                    parents=[
                        loader_cls.get_default_parser(),
                        plotter_cls.get_default_parser(),
                    ],
                    conflict_handler="resolve",
                )
        return parser


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
