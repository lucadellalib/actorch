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

"""Hydra model."""

from abc import abstractmethod
from typing import Dict, Tuple

from torch import Size, Tensor

from actorch.models.model import Model
from actorch.models.modules import BroadcastCat


__all__ = [
    "Hydra",
]


class Hydra(Model):
    """Hydra model, consisting of multiple input tails,
    a shared torso and multiple output heads.

      input 1   ...  input n
         |              |
    .____|___.      .___|____.
    | tail 1 |  ... | tail n |
    |________|      |________|
         |              |
         |______________|
                 |
             .___|___.
             | torso |
             |_______|
                 |
         ._______|______.
         |              |
    .____|___.      .___|____.
    | head 1 |  ... | head m |
    |________|      |________|
         |              |
         |              |
      output 1  ...  output m

    """

    # override
    def _setup(self) -> "None":
        self._setup_tails()
        example_inputs = self.get_example_inputs((2,))
        torso_example_input, _ = self._forward_tails(*example_inputs)
        torso_in_shape = torso_example_input.shape[1:]
        self._setup_torso(torso_in_shape)
        head_example_input, _ = self._forward_torso(
            torso_example_input, *example_inputs[1:]
        )
        head_in_shape = head_example_input.shape[1:]
        self._setup_heads(head_in_shape)

    # override
    def _forward(
        self,
        inputs: "Dict[str, Tensor]",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor]]":
        torso_input, states = self._forward_tails(inputs, states, mask)
        torso_output, states = self._forward_torso(torso_input, states, mask)
        outputs, states = self._forward_heads(torso_output, states, mask)
        return outputs, states

    def _setup_tails(self) -> "None":
        """Setup the model tails."""
        self.aggregate = BroadcastCat(list(self.in_shapes.values()))

    def _setup_torso(self, in_shape: "Size") -> "None":
        """Setup the model torso.

        Parameters
        ----------
        in_shape:
            The input event shape.

        """
        pass

    def _setup_heads(self, in_shape: "Size") -> "None":
        """Setup the model heads.

        Parameters
        ----------
        in_shape:
            The input event shape.

        """
        pass

    def _forward_tails(
        self,
        inputs: "Dict[str, Tensor]",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Tensor, Dict[str, Tensor]]":
        """Tails forward pass.

        In the following, let `B = {B_1, ..., B_k}` denote the batch shape,
        and `...` an arbitrary event shape.

        Parameters
        ----------
        inputs:
            The inputs, i.e. a dict whose key-value pairs are consistent
            with the given `in_shapes` initialization argument, shape of
            ``inputs[name]``: ``[*B, *in_shapes[name]]``.
        states:
            The states, i.e. a dict with arbitrary key-value pairs.
            Useful, for example, to store hidden states of recurrent models.
        mask:
            The boolean tensor indicating which batch elements are
            valid (True) and which are not (False), shape: ``[*B]``.

        Returns
        -------
            - The output, shape: ``[*B, *...]``;
            - the possibly updated states.

        Warnings
        --------
        Batch dimensions of `states` must be moved to the front.

        """
        return self.aggregate(list(inputs.values())), states

    def _forward_torso(
        self,
        input: "Tensor",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Tensor, Dict[str, Tensor]]":
        """Torso forward pass.

        In the following, let `B = {B_1, ..., B_k}` denote the batch shape,
        and `...` an arbitrary event shape.

        Parameters
        ----------
        input:
            The input, shape: ``[*B, *...]``.
        states:
            The states, i.e. a dict with arbitrary key-value pairs.
            Useful, for example, to store hidden states of recurrent models.
        mask:
            The boolean tensor indicating which batch elements are
            valid (True) and which are not (False), shape: ``[*B]``.

        Returns
        -------
            - The output, shape: ``[*B, *...]``;
            - the possibly updated states.

        Warnings
        --------
        Batch dimensions of `states` must be moved to the front.

        """
        return input, states

    @abstractmethod
    def _forward_heads(
        self,
        input: "Tensor",
        states: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Tensor], Dict[str, Tensor]]":
        """Heads forward pass.

        In the following, let `B = {B_1, ..., B_k}` denote the batch shape,
        and `...` an arbitrary event shape.

        Parameters
        ----------
        input:
            The input, shape: ``[*B, *...]``.
        states:
            The states, i.e. a dict with arbitrary key-value pairs.
            Useful, for example, to store hidden states of recurrent models.
        mask:
            The boolean tensor indicating which batch elements are
            valid (True) and which are not (False), shape: ``[*B]``.

        Returns
        -------
            - The outputs, i.e. a dict whose key-value pairs are consistent
              with the given `out_shapes` initialization argument, shape of
              ``outputs[name]``: ``[*B, *out_shapes[name]]``;
            - the possibly updated states.

        Warnings
        --------
        Batch dimensions of `states` must be moved to the front.

        """
        raise NotImplementedError
