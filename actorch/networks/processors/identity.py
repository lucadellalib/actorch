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

"""Identity processor."""

from typing import Tuple

from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "Identity",
]


class Identity(Processor):
    """Return a tensor unaltered."""

    # override
    def __init__(self, shape: "Tuple[int, ...]") -> "None":
        """Initialize the object.

        Parameters
        ----------
        shape:
            The input/output event shape.

        """
        self.shape = Size(shape)
        super().__init__()

    # override
    @property
    def in_shape(self) -> "Size":
        return self.shape

    # override
    @property
    def out_shape(self) -> "Size":
        return self.shape

    # override
    @property
    def inv(self) -> "Identity":
        return Identity(self.shape)

    # override
    def _forward(self, input: "Tensor") -> "Tensor":
        return input

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(shape: {self.shape})"
