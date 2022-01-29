#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""ACTorch main script."""

from argparse import ArgumentParser, Namespace
from typing import Any

from colorama import Fore, Style, init

from actorch.version import VERSION
from actorch.visualizer import Visualize


__all__ = [
    "ACTorch",
    "main",
]


init()


_ASCII_LOGO = (
    Fore.RED
    + Style.BRIGHT
    + f"""
    _    ____ _____              _     
   / \  / ___|_   _|__  _ __ ___| |__  
  / _ \| |     | |/ _ \| '__/ __| '_ \ 
 / ___ \ |___  | | (_) | | | (__| | | |
/_/   \_\____| |_|\___/|_|  \___|_| |_|{VERSION}

"""
    + Style.RESET_ALL
)


class ACTorch:
    """ACTorch main script."""

    _SCRIPT_TYPES = {
        # "run": Run,
        "visualize": Visualize,
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
        command = getattr(args, "command", None)
        if command not in self._SCRIPT_TYPES:
            raise ValueError(
                f"`command` ({command}) must be in {list(self._SCRIPT_TYPES.keys())}"
            )
        script = self._SCRIPT_TYPES[command]()
        del args.command
        script.main(args)

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
        parser = ArgumentParser(**parser_kwargs)
        subparsers = parser.add_subparsers(
            title="command", dest="command", required=True
        )
        for command, script_cls in cls._SCRIPT_TYPES.items():
            subparsers.add_parser(
                command,
                parents=[script_cls.get_default_parser()],
                conflict_handler="resolve",
            )
        return parser


def main() -> "None":
    """ACTorch entry point."""
    try:
        print(_ASCII_LOGO)
        parser = ACTorch.get_default_parser()
        args = parser.parse_args()
        print("------------------- Start ------------------")
        actorch = ACTorch()
        actorch.main(args)
        print("------------------- Done -------------------")
    except KeyboardInterrupt:
        print("---- Exiting early (Keyboard Interrupt) ----")


if __name__ == "__main__":
    main()
