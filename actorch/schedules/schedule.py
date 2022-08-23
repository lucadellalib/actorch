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

"""Schedule."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np
from numpy import ndarray

from actorch.utils import CheckpointableMixin


__all__ = [
    "Schedule",
]


class Schedule(ABC, CheckpointableMixin):
    """Time-dependent scheduling schema.

    Useful, for example, to anneal `epsilon` and `beta` parameters in
    epsilon-greedy exploration and prioritized experience replay, respectively.

    """

    _STATE_VARS = ["batch_size"]  # override

    def __init__(
        self,
        batch_size: "Optional[int]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        batch_size:
            The batch size.

        Raises
        ------
        ValueError
            If `batch_size` is not in the integer interval [1, inf).

        """
        if batch_size:
            if batch_size < 1 or not float(batch_size).is_integer():
                raise ValueError(
                    f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
                )
            batch_size = int(batch_size)
        self.batch_size = batch_size
        self.reset()

    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "None":
        """Reset the (possibly batched) schedule state.

        In the following, let `B` denote the batch size.

        Parameters
        ----------
        mask:
            The boolean array indicating which batch elements are
            to reset (True) and which are not (False), shape: ``[B]``.
            If a scalar or a singleton, it is broadcast accordingly.
            Default to ``np.ones(B, dtype=bool)``.

        Raises
        ------
        ValueError
            If length of `mask` is not equal to the batch size.

        """
        batch_size = self.batch_size or 1
        if mask is None:
            mask = np.ones(batch_size, dtype=bool)
        else:
            mask = np.array(mask, copy=False, ndmin=1)
            if len(mask) == 1:
                mask = np.broadcast_to(mask, (batch_size,))
            if len(mask) != batch_size:
                raise ValueError(
                    f"Length of `mask` ({len(mask)}) must be "
                    f"equal to the batch size ({batch_size})"
                )
        return self._reset(mask)

    def step(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "None":
        """Step the (possibly batched) schedule state.

        In the following, let `B` denote the batch size.

        Parameters
        ----------
        mask:
            The boolean array indicating which batch elements are
            to step (True) and which are not (False), shape: ``[B]``.
            If a scalar or a singleton, it is broadcast accordingly.
            Default to ``np.ones(B, dtype=bool)``.

        Raises
        ------
        ValueError
            If length of `mask` is not equal to the batch size.

        """
        batch_size = self.batch_size or 1
        if mask is None:
            mask = np.ones(batch_size, dtype=bool)
        else:
            mask = np.array(mask, copy=False, ndmin=1)
            if len(mask) == 1:
                mask = np.broadcast_to(mask, (batch_size,))
            if len(mask) != batch_size:
                raise ValueError(
                    f"Length of `mask` ({len(mask)}) must be "
                    f"equal to the batch size ({batch_size})"
                )
        return self._step(mask)

    def __repr__(self) -> "str":
        return f"{type(self).__name__}(batch_size: {self.batch_size})"

    def _reset(self, mask: "ndarray") -> "None":
        """See documentation of `reset`."""
        pass

    def _step(self, mask: "ndarray") -> "None":
        """See documentation of `step`."""
        pass

    @abstractmethod
    def __call__(self) -> "Union[int, float, ndarray]":
        """Return the (possibly batched) current value depending
        on the (possibly batched) schedule state.

        Returns
        -------
            The (possibly batched) current value.

        """
        raise NotImplementedError
