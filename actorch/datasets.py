# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Datasets."""

from typing import Dict, Iterator, Union

from torch.utils.data import IterableDataset

from actorch.buffers import Buffer
from actorch.schedules import ConstantSchedule, Schedule


__all__ = [
    "BufferDataset",
]


class BufferDataset(IterableDataset):
    """Dynamic dataset that generates data on-the-fly by sampling
    from an experience replay buffer.

    """

    def __init__(
        self,
        buffer: "Buffer",
        batch_size: "Union[int, Schedule]",
        max_trajectory_length: "Union[int, float, Schedule]",
        num_iters: "int",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        buffer:
            The experience replay buffer.
        batch_size:
            The schedule for argument `batch_size' of `buffer.sample`.
            If a number, it is wrapped in a `ConstantSchedule`.
        max_trajectory_length:
            The schedule for argument `max_trajectory_length' of `buffer.sample`.
            If a number, it is wrapped in a `ConstantSchedule`.
        num_iters:
            The number of iterations for which method
            `buffer.sample` is called.

        Raises
        ------
        ValueError
            If `num_iters` is not in the integer interval [1, inf).

        """
        if num_iters < 1 or not float(num_iters).is_integer():
            raise ValueError(
                f"`num_iters` ({num_iters}) must be in the integer interval [1, inf)"
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
        self.num_iters = int(num_iters)
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

    def __iter__(self) -> "Iterator":
        batch_size = self.batch_size()
        max_trajectory_length = self.max_trajectory_length()
        return (
            self.buffer.sample(batch_size, max_trajectory_length)
            for _ in range(self.num_iters)
        )

    def __len__(self) -> "int":
        return self.num_iters

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(buffer: {self.buffer}, "
            f"batch_size: {self.batch_size}, "
            f"max_trajectory_length: {self.max_trajectory_length}, "
            f"num_iters: {self.num_iters})"
        )
