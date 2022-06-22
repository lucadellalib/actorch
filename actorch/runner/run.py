#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Run experiment."""

import hashlib
import logging
import os
import shutil
import socket
import subprocess
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any

import ray
from ray import tune
from ray.tune.progress_reporter import detect_reporter

from actorch.runner.loggers import TBXLoggerCallback
from actorch.runner.progress_reporters import TrialProgressCallback
from actorch.utils import get_system_info, import_module, pretty_print


__all__ = [
    "Run",
]


_LOGGER = logging.getLogger(__name__)


class Run:
    """Run experiment."""

    def main(self, args: "Namespace") -> "tune.ExperimentAnalysis":
        """Script entry point.

        Parameters
        ----------
        args:
            The command line arguments obtained from an
            `argparse.ArgumentParser` (e.g. the default
            parser returned by `get_default_parser`).

        Returns
        -------
            The Ray Tune experiment analysis.

        Raises
        ------
        RuntimeError
            If the configuration file is invalid.

        """
        if args.suppress_warnings:
            # Suppress warnings across all subprocesses
            os.environ["PYTHONWARNINGS"] = "ignore"
        config_filepath = os.path.realpath(args.config_filepath)
        experiment_name = os.path.splitext(os.path.basename(config_filepath))[0]
        config = import_module(config_filepath)
        if not hasattr(config, "experiment_params"):
            raise ValueError(
                "Invalid configuration file: `experiment_params` undefined"
            )
        experiment_params = config.experiment_params

        if "name" not in experiment_params:
            # Set run name
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(config_filepath, "rb") as f:
                md5_hash = f"md5{hashlib.md5(f.read()).hexdigest()}"
            run_name = os.path.join(experiment_name, f"{current_time}_{socket.gethostname()}_{md5_hash}")
            experiment_params["name"] = run_name

        if "local_dir" not in experiment_params:
            experiment_params["local_dir"] = "experiments"
        run_dirpath = os.path.join(experiment_params["local_dir"], experiment_params["name"])
        try:
            # Copy configuration file
            os.makedirs(run_dirpath, exist_ok=True)
            shutil.copy(config_filepath, run_dirpath)
        except Exception as e:
            _LOGGER.warning(f"Could not copy configuration file: {e}")

        try:
            # Freeze requirements
            requirements = subprocess.check_output(["pip", "freeze"])
            with open(os.path.join(run_dirpath, "requirements.txt"), "wb") as f:
                f.write(requirements)
        except Exception as e:
            _LOGGER.warning(f"Could not freeze requirements: {e}")

        # Set progress reporter
        if "progress_reporter" not in experiment_params:
            progress_reporter = detect_reporter()
            progress_reporter.add_metric_column("episodes_total", representation="eps")
            cumreward_window_size = experiment_params.get("config", {}).get("cumreward_window_size", 100)
            metric_name = f"cumreward_{cumreward_window_size}"
            progress_reporter.add_metric_column(metric_name, representation=metric_name)
            experiment_params["progress_reporter"] = progress_reporter

        # Set callbacks
        callbacks = experiment_params.get("callbacks", [])
        metric = experiment_params.get("metric")
        if not any(isinstance(c, tune.progress_reporter.TrialProgressCallback) for c in callbacks):
            callbacks.append(TrialProgressCallback(metric=metric))
        if not any(isinstance(c, tune.logger.TrialProgressCallback) for c in callbacks):
            callbacks.append(TBXLoggerCallback())
        experiment_params["callbacks"] = callbacks

        if not ray.is_initialized():
            try:
                ray.init("auto")
            except Exception:
                ray.init()

        # Log system information for each cluster node
        @ray.remote
        def remote_get_system_info():
            try:
                return get_system_info()
            except Exception as e:
                _LOGGER.warning(f"Could not retrieve system information: {e}")
                return "unknown"

        try:
            system_infos = [
                {v.replace("node:", "Node "): ray.get(remote_get_system_info.options(resources={v: 1}).remote())}
                for v in ray.cluster_resources() if v.startswith("node:")
            ]
            with open(os.path.join(run_dirpath, "system-info.txt"), "w") as f:
                f.write("\n".join([pretty_print(v) for v in system_infos]))
        except Exception as e:
            _LOGGER.warning(f"Could not log system information: {e}")

        analysis = tune.run(**experiment_params)
        ray.shutdown()
        return analysis

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
        parser_kwargs.setdefault("description", "Run experiment")
        parser = ArgumentParser(**parser_kwargs)
        parser.add_argument(
            "config_filepath",
            help=(
                "absolute or relative path to the configuration file, i.e. "
                "a Python script that defines an `experiment_params` dict "
                "with the keyword arguments to pass to `ray.tune.run` "
                "(see https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run)"
            ),
            metavar="config-file",
        )
        parser.add_argument(
            "-s",
            "--suppress-warnings",
            action="store_true",
            help="suppress warnings",
        )
        return parser

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}()"


def _main() -> "None":
    try:
        parser = Run.get_default_parser()
        args = parser.parse_args()
        print("------------------- Start ------------------")
        run = Run()
        run.main(args)
        print("------------------- Done -------------------")
    except KeyboardInterrupt:
        print("---- Exiting early (Keyboard Interrupt) ----")


if __name__ == "__main__":
    _main()
