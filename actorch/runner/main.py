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

"""Runner main script."""

import hashlib
import logging
import os
import shutil
import socket
import subprocess
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any

import psutil
import ray
from ray import tune
from ray.tune.progress_reporter import detect_reporter

from actorch.runner.loggers import TBXLoggerCallback
from actorch.runner.progress_reporters import TrialProgressCallback
from actorch.utils import get_system_info, import_module, pretty_print


__all__ = [
    "ExperimentParams",
    "Runner",
]


_LOGGER = logging.getLogger(__name__)


class ExperimentParams(dict):
    """Keyword arguments expected in the `experiment_params`
    dict of an ACTorch configuration file.

    """

    def __init__(self, **run_kwargs: "Any") -> "None":
        """Initialize the object.

        Parameters
        ----------
        run_kwargs:
            The keyword arguments to pass to `ray.tune.run`
            (see https://docs.ray.io/en/releases-1.13.0/tune/api_docs/execution.html#tune-run for Ray 1.13.0).

        """
        super().__init__(**run_kwargs)


class Runner:
    """Runner main script."""

    @classmethod
    def main(cls, args: "Namespace") -> "tune.ExperimentAnalysis":  # noqa: C901
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
        config_file = os.path.realpath(args.config_file)
        experiment_name = os.path.splitext(os.path.basename(config_file))[0]
        config = import_module(config_file)
        if not hasattr(config, "experiment_params"):
            raise ValueError(
                "Invalid configuration file: `experiment_params` undefined"
            )
        experiment_params = ExperimentParams(**config.experiment_params)

        if "name" not in experiment_params:
            # Set run name
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(config_file, "rb") as f:
                md5_hash = f"md5-{hashlib.md5(f.read()).hexdigest()}"
            run_name = os.path.join(
                experiment_name, f"{current_time}_{socket.gethostname()}_{md5_hash}"
            )
            experiment_params["name"] = run_name

        if "local_dir" not in experiment_params:
            experiment_params["local_dir"] = "experiments"
        run_dir = os.path.join(
            experiment_params["local_dir"], experiment_params["name"]
        )
        try:
            # Copy configuration file
            os.makedirs(run_dir, exist_ok=True)
            shutil.copy(config_file, run_dir)
        except Exception as e:
            _LOGGER.warning(f"Could not copy configuration file: {e}")

        try:
            # Freeze requirements
            requirements = subprocess.check_output(["pip", "freeze"])
            with open(os.path.join(run_dir, "requirements.txt"), "wb") as f:
                f.write(requirements)
        except Exception as e:
            _LOGGER.warning(f"Could not freeze requirements: {e}")

        # Set progress reporter
        if "progress_reporter" not in experiment_params:
            progress_reporter = detect_reporter()
            progress_reporter.add_metric_column("episodes_total", representation="eps")
            cumreward_window_size = experiment_params.get("config", {}).get(
                "cumreward_window_size", 100
            )
            metric_name = f"cumreward_{cumreward_window_size}"
            progress_reporter.add_metric_column(metric_name, representation=metric_name)
            experiment_params["progress_reporter"] = progress_reporter

        # Set callbacks
        callbacks = experiment_params.get("callbacks", [])
        metric = experiment_params.get("metric")
        if not any(
            isinstance(callback, tune.progress_reporter.TrialProgressCallback)
            for callback in callbacks
        ):
            callbacks.append(TrialProgressCallback(metric=metric))
        if not any(isinstance(c, tune.logger.TBXLoggerCallback) for c in callbacks):
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
                {
                    v.replace("node:", "Node "): ray.get(
                        remote_get_system_info.options(resources={v: 1}).remote()
                    )
                }
                for v in ray.cluster_resources()
                if v.startswith("node:")
            ]
            with open(os.path.join(run_dir, "system-info.txt"), "w") as f:
                f.write("\n".join([pretty_print(v) for v in system_infos]))
        except Exception as e:
            _LOGGER.warning(f"Could not log system information: {e}")

        try:
            analysis = tune.run(**experiment_params)
            return analysis
        finally:
            cls._cleanup()

            @ray.remote
            def remote_cleanup():
                cls._cleanup()

            for v in ray.cluster_resources():
                if v.startswith("node:"):
                    ray.get(remote_cleanup.options(resources={v: 1}).remote())
            ray.shutdown()

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
            "config_file",
            help=(
                "absolute or relative path to the configuration file, i.e. a "
                "Python script that defines a dict named `experiment_params` "
                "with keyword arguments to pass to `ray.tune.run` "
                f"(see https://docs.ray.io/en/releases-{ray.__version__}/tune/api_docs/execution.html#tune-run)"
            ),
            metavar="config-file",
        )
        return parser

    @classmethod
    def _cleanup(cls) -> "None":
        # Workaround to kill all Ray processes spawned by this session
        for process in psutil.Process(os.getpid()).children(recursive=True):
            if process.name() == "ray::IDLE" or (
                process.name().startswith("ray::") and "init" in process.name()
            ):
                process.kill()
        for process in psutil.process_iter():
            if process.name().startswith("ray::") and "init" in process.name():
                process.kill()


def main() -> "None":
    """Runner entry point."""
    try:
        parser = Runner.get_default_parser()
        args = parser.parse_args()
        print("------------------- Start ------------------")
        Runner.main(args)
        print("------------------- Done -------------------")
    except KeyboardInterrupt:
        print("---- Exiting early (Keyboard Interrupt) ----")


if __name__ == "__main__":
    main()
