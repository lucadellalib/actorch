# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Custom Ray Tune progress reporters."""

import os
import time
from typing import Any, Dict

from ray.tune import progress_reporter
from ray.tune.trial import DEBUG_PRINT_INTERVAL, Trial
from ray.tune.utils.log import Verbosity, has_verbosity

from actorch.utils import pretty_print


__all__ = [
    "TrialProgressCallback",
]


# Adapted from:
# https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/python/ray/tune/progress_reporter.py#L945
class TrialProgressCallback(progress_reporter.TrialProgressCallback):
    """Modified version of `ray.tune.progress_reporter.TrialProgressCallback`
    that prints values through `actorch.utils.pretty_print`.

    """

    # override
    def log_result(
        self,
        trial: "Trial",
        result: "Dict[str, Any]",
        error: "bool" = False,
    ) -> "None":
        done = result.get("done", False) is True
        last_print = self._last_print[trial]
        if done and trial not in self._completed_trials:
            self._completed_trials.add(trial)
        if has_verbosity(Verbosity.V3_TRIAL_DETAILS) and (
            done or error or time.time() - last_print > DEBUG_PRINT_INTERVAL
        ):
            pretty_printed_result = pretty_print(result).replace("\n", "\n  ")
            print(f"Result for {trial}:")
            print(f"  {pretty_printed_result}")
            self._last_print[trial] = time.time()
        elif has_verbosity(Verbosity.V2_TRIAL_NORM) and (
            done or error or time.time() - last_print > DEBUG_PRINT_INTERVAL
        ):
            info = ""
            if done:
                info = " This trial completed."

            metric_name = self._metric or "_metric"
            metric_value = result.get(metric_name, -99.0)
            print_result_str = self._print_result(result)
            self._last_result_str[trial] = print_result_str

            error_file = os.path.join(trial.logdir, "error.txt")
            if error:
                message = (
                    f"The trial {trial} errored with "
                    f"parameters={trial.config}. "
                    f"Error file: {error_file}"
                )
            elif self._metric:
                message = (
                    f"Trial {trial} reported "
                    f"{metric_name}={metric_value:.2f} "
                    f"with parameters={trial.config}.{info}"
                )
            else:
                message = (
                    f"Trial {trial} reported "
                    f"{print_result_str} "
                    f"with parameters={trial.config}.{info}"
                )

            print(message)
            self._last_print[trial] = time.time()
