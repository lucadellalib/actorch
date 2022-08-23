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

"""Batched environment wrapper."""

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.utils import Nested


__all__ = [
    "BatchedWrapper",
]


class BatchedWrapper(BatchedEnv):
    """Batched environment wrapper."""

    # override
    def __init__(self, env: "BatchedEnv") -> "None":
        """Initialize the object.

        Parameters
        ----------
        env:
            The batched environment to wrap.

        """
        self.env = env

    # override
    @property
    def unwrapped(self) -> "BatchedEnv":
        return self.env.unwrapped

    # override
    def reset(
        self,
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray]":
        return self.env.reset(mask)

    # override
    def step(
        self,
        action: "Nested[ndarray]",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        return self.env.step(action, mask)

    # override
    def seed(
        self,
        seed: "Union[Optional[int], Sequence[Optional[int]], ndarray]" = None,
    ) -> "None":
        self.env.seed(seed)
        if np.isscalar(seed):
            seed = [seed]
        self.single_observation_space.seed(seed[0])
        self.single_action_space.seed(seed[0])
        self.observation_space.seed(seed[0])
        self.action_space.seed(seed[0])

    # override
    def close(self) -> "None":
        return self.env.close()

    # override
    def _reset(self, idx: "Sequence[int]") -> "Tuple[Nested[ndarray], ndarray]":
        raise NotImplementedError

    # override
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        raise NotImplementedError

    # override
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        raise NotImplementedError

    # override
    def _close(self) -> "None":
        raise NotImplementedError

    # override
    def __len__(self) -> "int":
        return self.env.__len__()

    # override
    def __del__(self) -> "None":
        return self.env.__del__()

    # override
    def __repr__(self) -> "str":
        return f"<{type(self).__name__}{self.env}>"

    # Implicitly forward all other attributes to self.env
    def __getattr__(self, name: "str") -> "Any":
        return getattr(self.env, name)
