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

"""Independent processor."""

from typing import Tuple

from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "Independent",
]


class Independent(Processor):
    """Wrap a processor to treat an extra
    rightmost shape as dependent.

    """

    # override
    def __init__(
        self,
        base_processor: "Processor",
        reinterpreted_batch_shape: "Tuple[int, ...]",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        base_processor:
            The base processor.
        reinterpreted_batch_shape:
            The extra rightmost shape
            to treat as dependent.

        """
        self.base_processor = base_processor
        self.reinterpreted_batch_shape = Size(reinterpreted_batch_shape)
        super().__init__()

    # override
    @property
    def in_shape(self) -> "Size":
        return self.reinterpreted_batch_shape + self.base_processor.in_shape

    # override
    @property
    def out_shape(self) -> "Size":
        return self.reinterpreted_batch_shape + self.base_processor.out_shape

    # override
    @property
    def inv(self) -> "Independent":
        return Independent(self.base_processor.inv, self.reinterpreted_batch_shape)

    # override
    def _forward(self, input: "Tensor") -> "Tensor":
        return self.base_processor(input)

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(base_processor: {self.base_processor}, "
            f"reinterpreted_batch_shape: {self.reinterpreted_batch_shape})"
        )
