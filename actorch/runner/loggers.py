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

"""Custom Ray Tune loggers."""

import logging
from typing import Any, Dict

import numpy as np
from ray.tune import logger
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.trial import Trial
from ray.tune.utils import flatten_dict
from ray.util.debug import log_once


__all__ = [
    "TBXLogger",
    "TBXLoggerCallback",
]


_LOGGER = logging.getLogger(__name__)


# Adapted from:
# https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/python/ray/tune/logger.py#L161
class TBXLogger(logger.TBXLogger):
    """Modified version of `ray.tune.logger.TBXLogger` that logs
    values without prepending ``"ray/tune"`` to their tags.

    """

    # override
    def on_result(
        self,
        result: "Dict[str, Any]",
    ) -> "None":
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        tmp = result.copy()
        for k in ["config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]  # Not useful to log these

        flat_result = flatten_dict(tmp, delimiter="/")
        path = ["ray", "tune"]
        valid_result = {}

        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            if isinstance(value, tuple(logger.VALID_SUMMARY_TYPES)) and not np.isnan(
                value
            ):
                valid_result[full_attr] = value
                self._file_writer.add_scalar(full_attr, value, global_step=step)
            elif (isinstance(value, list) and len(value) > 0) or (
                isinstance(value, np.ndarray) and value.size > 0
            ):
                valid_result[full_attr] = value

                # Must be a video
                if isinstance(value, np.ndarray) and value.ndim == 5:
                    self._file_writer.add_video(
                        full_attr, value, global_step=step, fps=20
                    )
                    continue

                try:
                    self._file_writer.add_histogram(full_attr, value, global_step=step)
                    # If TensorBoardX still does not think it is a
                    # valid value (e.g. [[]]), warn and move on
                except (ValueError, TypeError):
                    if log_once("invalid_tbx_value"):
                        _LOGGER.warning(
                            f"You are trying to log an invalid value ({full_attr}={value}) "
                            f"through {type(self).__name__}!"
                        )

        self.last_result = valid_result
        self._file_writer.flush()


# Adapted from:
# https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/python/ray/tune/logger.py#L599
class TBXLoggerCallback(logger.TBXLoggerCallback):
    """Modified version of `ray.tune.logger.TBXLoggerCallback` that logs
    values without prepending ``"ray/tune"`` to their tags.

    """

    # override
    def log_trial_result(
        self,
        iteration: "int",
        trial: "Trial",
        result: "Dict[str, Any]",
    ) -> "None":
        if trial not in self._trial_writer:
            self.log_trial_start(trial)

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        tmp = result.copy()
        for k in ["config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]  # Not useful to log these

        flat_result = flatten_dict(tmp, delimiter="/")
        path = []
        valid_result = {}

        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            if isinstance(value, tuple(logger.VALID_SUMMARY_TYPES)) and not np.isnan(
                value
            ):
                valid_result[full_attr] = value
                self._trial_writer[trial].add_scalar(full_attr, value, global_step=step)
            elif (isinstance(value, list) and len(value) > 0) or (
                isinstance(value, np.ndarray) and value.size > 0
            ):
                valid_result[full_attr] = value

                # Must be a video
                if isinstance(value, np.ndarray) and value.ndim == 5:
                    self._trial_writer[trial].add_video(
                        full_attr, value, global_step=step, fps=20
                    )
                    continue

                try:
                    self._file_writer.add_histogram(full_attr, value, global_step=step)
                    # If TensorBoardX still does not think it is a
                    # valid value (e.g. [[]]), warn and move on
                except (ValueError, TypeError):
                    if log_once("invalid_tbx_value"):
                        _LOGGER.warning(
                            f"You are trying to log an invalid value ({full_attr}={value}) "
                            f"through {type(self).__name__}!"
                        )

        self._trial_result[trial] = valid_result
        self._trial_writer[trial].flush()
