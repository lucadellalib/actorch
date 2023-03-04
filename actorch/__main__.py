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

"""ACTorch main script."""

from argparse import ArgumentParser, Namespace
from typing import Any

from colorama import Fore, Style, init

from actorch.runner import Runner
from actorch.version import VERSION


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
   / \\  / ___|_   _|__  _ __ ___| |__  
  / _ \\| |     | |/ _ \\| '__/ __| '_ \\ 
 / ___ \\ |___  | | (_) | | | (__| | | |
/_/   \\_\\____| |_|\\___/|_|  \\___|_| |_|{VERSION}

"""
    + Style.RESET_ALL
)


class ACTorch:
    """ACTorch main script."""

    _SCRIPTS = {"run": Runner}

    try:
        from actorch.vistool import VisTool

        _SCRIPTS["vistool"] = VisTool
    except ImportError as e:
        import warnings

        warnings.warn(str(e), UserWarning)

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
        command = getattr(args, "command", None)
        if command not in cls._SCRIPTS:
            raise ValueError(f"`command` ({command}) must be in {list(cls._SCRIPTS)}")
        script = cls._SCRIPTS[command]()
        del args.command
        script.main(args)

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
        parser_kwargs.setdefault("description", "ACTorch main script")
        parser = ArgumentParser(**parser_kwargs)
        subparsers = parser.add_subparsers(
            title="command", dest="command", required=True
        )
        for command, script in cls._SCRIPTS.items():
            command_parser = script.get_default_parser()
            help = description = command_parser.description
            if help:
                help = f"{help[0].lower()}{help[1:]}"
            subparsers.add_parser(
                command,
                description=description,
                help=help,
                parents=[command_parser],
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
        ACTorch.main(args)
        print("------------------- Done -------------------")
    except KeyboardInterrupt:
        print("---- Exiting early (Keyboard Interrupt) ----")


if __name__ == "__main__":
    main()
