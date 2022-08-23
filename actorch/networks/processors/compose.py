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

"""Compose processor."""

from typing import Sequence

from torch import Size, Tensor

from actorch.networks.processors.processor import Processor


__all__ = [
    "Compose",
]


class Compose(Processor):
    """Compose multiple base processors in a chain."""

    # override
    def __init__(self, base_processors: "Sequence[Processor]") -> "None":
        """Initialize the object.

        Parameters
        ----------
        base_processors:
            The base processors to compose.

        Raises
        ------
        ValueError
            If input and output shapes of
            `base_processors` are incompatible.

        """
        out_shape = base_processors[0].out_shape
        for i, base_processor in enumerate(base_processors[1:], 1):
            if base_processor.in_shape != out_shape:
                raise ValueError(
                    f"`base_processors[{i}].in_shape` ({base_processor.in_shape}) must "
                    f"be equal to `base_processors[{i - 1}].out_shape` ({out_shape})"
                )
            out_shape = base_processor.out_shape
        self.base_processors = base_processors
        super().__init__()

    # override
    @property
    def in_shape(self) -> "Size":
        return self.base_processors[0].in_shape

    # override
    @property
    def out_shape(self) -> "Size":
        return self.base_processors[-1].out_shape

    # override
    @property
    def inv(self) -> "Compose":
        return Compose([p.inv for p in self.base_processors[::-1]])

    # override
    def _forward(self, input: "Tensor") -> "Tensor":
        output = input
        for base_processor in self.base_processors:
            output = base_processor(output)
        return output

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}" f"(base_processors: {self.base_processors})"
