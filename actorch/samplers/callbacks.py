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

"""Experience sampler callbacks."""

from typing import Any, Dict


__all__ = [
    "Callback",
]


class Callback:
    """Experience sampler callback."""

    def on_episode_start(
        self,
        stats: "Dict[str, Any]",
        info: "Dict[str, Any]",
    ) -> "None":
        """The function called directly after an episode has started.

        Parameters
        ----------
        stats:
            The current sampling statistics.
        info:
            The auxiliary diagnostic information
            returned by the environment.

        See Also
        --------
        actorch.samplers.sampler.Sampler.stats

        """
        pass

    def on_episode_step(
        self,
        stats: "Dict[str, Any]",
        info: "Dict[str, Any]",
    ) -> "None":
        """The function called directly after an episode has stepped.

        Parameters
        ----------
        stats:
            The current sampling statistics.
        info:
            The auxiliary diagnostic information
            returned by the environment.

        See Also
        --------
        actorch.samplers.sampler.Sampler.stats

        """
        pass

    def on_episode_end(
        self,
        stats: "Dict[str, Any]",
    ) -> "None":
        """The function called directly after an episode has ended.

        Parameters
        ----------
        stats:
            The current sampling statistics.

        See Also
        --------
        actorch.samplers.sampler.Sampler.stats

        """
        pass

    def __repr__(self) -> "str":
        return f"{type(self).__name__}()"
