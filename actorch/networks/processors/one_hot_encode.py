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

"""One-hot encode processor."""

import torch.nn.functional as F
from torch import Size, Tensor

from actorch.networks.processors import one_hot_decode as ohd  # Avoid circular import
from actorch.networks.processors.processor import Processor


__all__ = [
    "OneHotEncode",
]


class OneHotEncode(Processor):
    """One-hot encode a tensor."""

    # override
    def __init__(self, num_classes: "int") -> "None":
        """Initialize the object.

        Parameters
        ----------
        num_classes:
            The number of classes.

        Raises
        ------
        ValueError
            If `num_classes` is not in the integer interval [1, inf).

        """
        if num_classes < 1 or not float(num_classes).is_integer():
            raise ValueError(
                f"`num_classes` ({num_classes}) must be in the integer interval [1, inf)"
            )
        self.num_classes = int(num_classes)
        self._in_shape = Size([])
        self._out_shape = Size([self.num_classes])
        super().__init__()

    # override
    @property
    def in_shape(self) -> "Size":
        return self._in_shape

    # override
    @property
    def out_shape(self) -> "Size":
        return self._out_shape

    # override
    @property
    def inv(self) -> "ohd.OneHotDecode":
        return ohd.OneHotDecode(self.num_classes)

    # override
    def _forward(self, input: "Tensor") -> "Tensor":
        return F.one_hot(input.long(), self.num_classes)

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(num_classes: {self.num_classes})"
