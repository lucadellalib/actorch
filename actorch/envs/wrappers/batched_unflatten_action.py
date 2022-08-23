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

"""Batched unflatten action environment wrapper."""

from typing import Optional, Sequence, Tuple, Union

from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.spaces import Flat
from actorch.envs.utils import Nested
from actorch.envs.wrappers.batched_wrapper import BatchedWrapper


__all__ = [
    "BatchedUnflattenAction",
]


class BatchedUnflattenAction(BatchedWrapper):
    """Batched environment wrapper that unflattens the action."""

    # override
    def __init__(self, env: "BatchedEnv") -> "None":
        super().__init__(env)
        self._single_action_space = Flat(env.single_action_space)
        self._action_space = Flat(env.action_space, is_batched=True)

    # override
    def step(
        self,
        action: "ndarray",
        mask: "Optional[Union[bool, Sequence[bool], ndarray]]" = None,
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        return self.env.step(self._action(action), mask)

    def _action(self, action: "ndarray") -> "Nested[ndarray]":
        return self._action_space.unflatten(action)
