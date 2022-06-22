# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
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
    ) -> "None":
        """The function called directly after an episode has started.

        Parameters
        ----------
        stats:
            The current sampling statistics.

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
        return f"{self.__class__.__name__}()"
