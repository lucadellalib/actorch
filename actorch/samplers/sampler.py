# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Experience sampler."""

import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

from gym import Env

from actorch.agents import Agent
from actorch.envs import BatchedEnv
from actorch.samplers.callbacks import Callback


__all__ = [
    "Sampler",
]


class Sampler:
    """Sampler that enables an agent to interact with
    an environment and sample experiences from it.

    """

    def __init__(
        self,
        env: "Env",
        agent: "Agent",
        callbacks: "Optional[Sequence[Callback]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        env:
            The environment to sample experiences from.
        agent:
            The agent that interacts with the environment.
        callbacks:
            The experience sampler callbacks.
            Default to ``[]``.

        Raises
        ------
        TypeError
            If `env` is instance of `actorch.envs.BatchedEnv`.
        ValueError
            If observation or action spaces of `agent` and `env` are not equal.

        """
        if agent.observation_space != env.observation_space:
            raise ValueError(
                f"Observation spaces of `agent` ({agent.observation_space}) "
                f"and `env` ({env.observation_space}) must be equal"
            )
        if agent.action_space != env.action_space:
            raise ValueError(
                f"Action spaces of `agent` ({agent.action_space}) "
                f"and `env` ({env.action_space}) must be equal"
            )
        self.env = env
        self.agent = agent
        self.callbacks = callbacks or []
        self.reset()
        if isinstance(env, BatchedEnv):
            raise TypeError(
                f"`env` ({env}) must be instance of "
                f"`{BatchedEnv.__module__}.{BatchedEnv.__name__}`"
            )

    @property
    def stats(self) -> "Optional[Dict[str, Any]]":
        """Return the sampling statistics.

        Returns
        -------
            The sampling statistics, i.e. a dict with the following key-value pairs:
            - "num_timesteps":      the number of sampled timesteps;
            - "num_episodes":       the number of sampled episodes;
            - "episode_lengths":    the lengths of each sampled episode;
            - "episode_cumrewards": the cumulative rewards of each sampled episode;
            - "inference_time_ms":  the agent inference times in milliseconds;
            - "step_time_ms":       the environment step times in milliseconds;
            - "render_time_ms":     the environment render times in milliseconds
                                    (if rendering is enabled).

        Notes
        -----
        Callbacks can be used to add custom metrics to the sampling statistics.

        """
        return self._stats

    def reset(self) -> "None":
        """Reset the sampler state."""
        self._length = 0
        self._cumreward = 0.0
        self._stats = None
        self._next_observation = None

    def sample(
        self,
        num_timesteps: "Optional[Union[int, float]]" = None,
        num_episodes: "Optional[Union[int, float]]" = None,
        render: "bool" = False,
    ) -> "Iterator[Tuple[Dict[str, Any], bool]]":
        """Sample experiences from the environment for a given
        number of timesteps or episodes.

        Parameters
        ----------
        num_timesteps:
            The number of timesteps to sample.
            Must be None if `num_episodes` is given.
        num_episodes:
            The number of episodes to sample.
            Must be None if `num_timesteps` is given.
        render:
            True to render the environment, False otherwise.

        Yields
        ------
            - The sampled experience, i.e. a dict with the
              following key-value pairs:
              - "observation":      the observation;
              - "action":           the action;
              - "log_prob":         the action log probability;
              - "reward":           the reward;
              - "next_observation": the next observation;
              - "done":             the end-of-episode flag;
            - the terminal flag (might differ from `done` if
              the environment has a time limit).

        Raises
        ------
        ValueError
            If both or none of `num_timesteps` and `num_episodes`
            are given or if `num_timesteps` or `num_episodes`
            are not in the integer interval [1, inf].

        """
        if not (bool(num_timesteps) ^ bool(num_episodes)):
            raise ValueError(
                f"Either `num_timesteps` ({num_timesteps}) "
                f"or `num_episodes` ({num_episodes}) must be "
                f"given, but not both"
            )
        if num_timesteps is not None and num_timesteps != float("inf"):
            if num_timesteps < 0 or not float(num_timesteps).is_integer():
                raise ValueError(
                    f"`num_timesteps` ({num_timesteps}) must be in the integer interval [0, inf]"
                )
            num_timesteps = int(num_timesteps)
        if num_episodes is not None and num_episodes != float("inf"):
            if num_episodes < 0 or not float(num_episodes).is_integer():
                raise ValueError(
                    f"`num_episodes` ({num_episodes}) must be in the integer interval [0, inf]"
                )
            num_episodes = int(num_episodes)
        num_timesteps = num_timesteps or float("inf")
        num_episodes = num_episodes or float("inf")
        self._stats = {
            "num_timesteps": 0,
            "num_episodes": 0,
            "episode_lengths": [],
            "episode_cumrewards": [],
            "inference_time_ms": [],
            "step_time_ms": [],
        }
        if render:
            self.stats["render_time_ms"] = []
        yield from self._sample(num_timesteps, num_episodes, render)

    def _sample(
        self,
        num_timesteps: "Union[int, float]",
        num_episodes: "Union[int, float]",
        render: "bool",
    ) -> "Iterator[Tuple[Dict[str, Any], bool]]":
        num_sampled_timesteps, num_sampled_episodes = 0, 0
        while (
            num_sampled_timesteps < num_timesteps
            and num_sampled_episodes < num_episodes
        ):
            if self._next_observation is None:
                # First call
                self._next_observation = self.env.reset()
                if render:
                    render_time = time.time()
                    self.env.render()
                    self._stats["render_time_ms"] += (time.time() - render_time) * 1000
                # Callbacks
                for callback in self.callbacks:
                    callback.on_episode_start(self._stats)
            observation = self._next_observation
            inference_time = time.time()
            action, log_prob = self.agent(observation)
            self._stats["inference_time_ms"] += (time.time() - inference_time) * 1000
            step_time = time.time()
            next_observation, reward, terminal, info = self.env.step(action)
            self._stats["step_time_ms"] += (time.time() - step_time) * 1000
            if render:
                render_time = time.time()
                self.env.render()
                self._stats["render_time_ms"] += (time.time() - render_time) * 1000
            self._next_observation = next_observation
            done = terminal
            self._length += 1
            self._cumreward += reward
            num_sampled_timesteps += 1
            self._stats["num_timesteps"] = num_sampled_timesteps
            # Callbacks
            for callback in self.callbacks:
                callback.on_episode_step(self._stats, info)

            if terminal:
                self._next_observation = self.env.reset()
                if render:
                    render_time = time.time()
                    self.env.render()
                    self._stats["render_time_ms"] += (time.time() - render_time) * 1000
                self._stats["episode_lengths"].append(self._length)
                self._stats["episode_cumrewards"].append(self._cumreward)
                self._length = 0
                self._cumreward = 0.0
                num_sampled_episodes += 1
                self._stats["num_episodes"] = num_sampled_episodes
                self.agent.reset()
                done = not info.get("TimeLimit.truncated", not terminal)
                # Callbacks
                for callback in self.callbacks:
                    callback.on_episode_end(self._stats)

            experience = {
                "observation": observation,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
            }
            yield experience, terminal

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(env: {self.env}, "
            f"agent: {self.agent}, "
            f"callbacks: {self.callbacks})"
        )
