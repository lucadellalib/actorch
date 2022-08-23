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

"""Datasets."""

from typing import Dict, Iterator, Tuple, Union

from numpy import ndarray
from torch.utils.data import IterableDataset

from actorch.buffers import Buffer
from actorch.schedules import ConstantSchedule, Schedule
from actorch.utils import CheckpointableMixin


__all__ = [
    "BufferDataset",
]


class BufferDataset(IterableDataset, CheckpointableMixin):
    """Dynamic dataset that generates data on-the-fly by sampling
    from an experience replay buffer.

    """

    _STATE_VARS = [
        "buffer",
        "batch_size",
        "max_trajectory_length",
        "num_iterations",
    ]  # override

    def __init__(
        self,
        buffer: "Buffer",
        batch_size: "Union[int, Schedule]",
        max_trajectory_length: "Union[int, float, Schedule]",
        num_iterations: "int",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        buffer:
            The experience replay buffer to sample data from.
        batch_size:
            The schedule for argument `batch_size' of `buffer.sample`.
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
        max_trajectory_length:
            The schedule for argument `max_trajectory_length' of `buffer.sample`.
            If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
        num_iterations:
            The number of iterations for which `buffer.sample` is called.

        Raises
        ------
        ValueError
            If `num_iterations` is not in the integer interval [1, inf).

        """
        if num_iterations < 1 or not float(num_iterations).is_integer():
            raise ValueError(
                f"`num_iterations` ({num_iterations}) must be in the integer interval [1, inf)"
            )
        self.buffer = buffer
        self.batch_size = (
            batch_size
            if isinstance(batch_size, Schedule)
            else ConstantSchedule(batch_size)
        )
        self.max_trajectory_length = (
            max_trajectory_length
            if isinstance(max_trajectory_length, Schedule)
            else ConstantSchedule(max_trajectory_length)
        )
        self.num_iterations = int(num_iterations)
        self._schedules = {
            "batch_size": self.batch_size,
            "max_trajectory_length": self.max_trajectory_length,
        }

    @property
    def schedules(self) -> "Dict[str, Schedule]":
        """Return the dataset public schedules.

        Returns
        -------
            The dataset public schedules, i.e. a dict that maps
            names of the schedules to the schedules themselves.

        """
        return self._schedules

    # override
    def __iter__(self) -> "Iterator[Tuple[Dict[str, ndarray], ndarray, ndarray]]":
        batch_size = self.batch_size()
        max_trajectory_length = self.max_trajectory_length()
        return (
            self.buffer.sample(batch_size, max_trajectory_length)
            for _ in range(self.num_iterations)
        )

    def __len__(self) -> "int":
        return self.num_iterations

    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(buffer: {self.buffer}, "
            f"batch_size: {self.batch_size}, "
            f"max_trajectory_length: {self.max_trajectory_length}, "
            f"num_iterations: {self.num_iterations})"
        )
