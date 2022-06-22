# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Schedule."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from numpy import ndarray


__all__ = [
    "Schedule",
]


class Schedule(ABC):
    """Time-dependent scheduling schema.

    Useful, for example, to anneal `epsilon` and `beta` parameters in
    epsilon-greedy exploration and prioritized experience replay, respectively.

    """

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

    def state_dict(self) -> "Dict[str, Any]":
        """Return the schedule state dict.

        Returns
        -------
            The schedule state dict.

        """
        return {
            k: v.state_dict() if hasattr(v, "state_dict") else v
            for k, v in self.__dict__.items()
        }

    def load_state_dict(self, state_dict: "Dict[str, Any]") -> "None":
        """Load a state dict into the schedule.

        Parameters
        ----------
        state_dict:
            The state dict.

        Raises
        ------
        KeyError
            If `state_dict` contains unknown keys.

        """
        for k, v in state_dict.items():
            if k not in self.__dict__:
                raise KeyError(f"{k}")
            if hasattr(self.__dict__[k], "load_state_dict"):
                self.__dict__[k].load_state_dict(v)
            else:
                self.__dict__[k] = v

    @abstractmethod
    def __call__(self) -> "Union[int, float, ndarray]":
        """Return the (possibly batched) current value depending
        on the (possibly batched) schedule state.

        Returns
        -------
            The (possibly batched) current value.

        """
        raise NotImplementedError

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(batch_size: {self.batch_size})"

    def _reset(self, mask: "ndarray") -> "None":
        """See documentation of `reset`."""
        pass

    def _step(self, mask: "ndarray") -> "None":
        """See documentation of `step`."""
        pass
