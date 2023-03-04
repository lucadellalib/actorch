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

"""Experience sampler."""

import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from gymnasium import Env
from numpy import ndarray

from actorch.agents import Agent
from actorch.envs import BatchedEnv
from actorch.samplers.callbacks import Callback


__all__ = [
    "Sampler",
]


class Sampler:
    """Sampler that enables an agent to interact with a (possibly batched)
    environment and sample (possibly batched) experiences from it.

    """

    def __init__(
        self,
        env: "Union[Env, BatchedEnv]",
        agent: "Agent",
        callbacks: "Optional[Sequence[Callback]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        env:
            The (possibly batched) environment to sample
            (possibly batched) experiences from.
        agent:
            The agent that interacts with the (possibly
            batched) environment.
        callbacks:
            The experience sampler callbacks.
            Default to ``[]``.

        Raises
        ------
        ValueError
            If observation or action spaces of `env`
            and `agent` are not equal.

        """
        if env.observation_space != agent.observation_space:
            raise ValueError(
                f"Observation spaces of `env` ({env.observation_space}) and "
                f"`agent` ({agent.observation_space}) must be identical"
            )
        if env.action_space != agent.action_space:
            raise ValueError(
                f"Action spaces of `env` ({env.action_space}) and "
                f"`agent` ({agent.action_space}) must be identical"
            )
        self.env = env
        self.agent = agent
        self.callbacks = callbacks or []
        self._batch_size = None
        if isinstance(env, BatchedEnv):
            self._batch_size = len(env)
        self.reset()

    @property
    def stats(self) -> "Optional[Dict[str, Any]]":
        """Return the sampling statistics.

        Returns
        -------
            The sampling statistics, i.e. a dict with the following key-value pairs:
            - "num_timesteps":     the number of sampled timesteps;
            - "num_episodes":      the number of sampled episodes;
            - "episode_length":    the lengths of each sampled episode;
            - "episode_cumreward": the cumulative rewards of each sampled episode;
            - "inference_time_ms": the agent inference times in milliseconds;
            - "step_time_ms":      the (possibly batched) environment step times
                                   in milliseconds.

        Notes
        -----
        Callbacks can be used to add custom metrics to the sampling statistics.

        """
        return self._stats

    def reset(self) -> "None":
        """Reset the sampler state."""
        self._length = 0
        self._cumreward = 0.0
        if self._batch_size:
            self._length = np.zeros(self._batch_size, dtype=np.int64)
            self._cumreward = np.zeros(self._batch_size)
        self._stats = None
        self._next_observation = None

    def sample(
        self,
        num_timesteps: "Optional[Union[int, float]]" = None,
        num_episodes: "Optional[Union[int, float]]" = None,
    ) -> "Iterator[Tuple[Dict[str, Any], Union[bool, ndarray]]]":
        """Sample (possibly batched) experiences from the (possibly batched)
        environment for a given number of timesteps or episodes.

        Parameters
        ----------
        num_timesteps:
            The number of timesteps to sample.
            Must be None if `num_episodes` is given.
        num_episodes:
            The number of episodes to sample.
            Must be None if `num_timesteps` is given.

        Yields
        ------
            - The sampled (possibly batched) experience, i.e.
              a dict with the following key-value pairs:
              - "observation":      the (possibly batched) observation;
              - "next_observation": the (possibly batched) next observation;
              - "action":           the (possibly batched) action;
              - "log_prob":         the (possibly batched) action log probability;
              - "reward":           the (possibly batched) reward;
              - "terminal":         the (possibly batched) terminal flag;
            - the (possibly batched) end-of-episode flag (might differ from
              ``experience["terminal"]`` in case of episode truncation).

        Raises
        ------
        ValueError
            If both or none of `num_timesteps` and `num_episodes`
            are given or if `num_timesteps` or `num_episodes`
            are not in the integer interval [1, inf].

        Warnings
        --------
        If the environment is batched, since experiences are sampled in batches
        of size `B`, up to `B` - 1 extra timesteps or episodes might be sampled.

        """
        if not (bool(num_timesteps) ^ bool(num_episodes)):
            raise ValueError(
                f"Either `num_timesteps` ({num_timesteps}) or `num_episodes` "
                f"({num_episodes}) must be given, but not both"
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
            "episode_length": [],
            "episode_cumreward": [],
            "inference_time_ms": [],
            "step_time_ms": [],
        }
        args = [num_timesteps, num_episodes]
        yield from (
            self._batched_sample(*args) if self._batch_size else self._sample(*args)
        )

    def _sample(
        self,
        num_timesteps: "Union[int, float]",
        num_episodes: "Union[int, float]",
    ) -> "Iterator[Tuple[Dict[str, Any], bool]]":
        num_sampled_timesteps, num_sampled_episodes = 0, 0
        while (
            num_sampled_timesteps < num_timesteps
            and num_sampled_episodes < num_episodes
        ):
            if self._next_observation is None:
                # First call
                self._next_observation, info = self.env.reset()
                # Callbacks
                for callback in self.callbacks:
                    callback.on_episode_start(self._stats, info)
            observation = self._next_observation
            inference_time = time.time()
            action, log_prob = self.agent(observation)
            self._stats["inference_time_ms"].append(
                (time.time() - inference_time) * 1000
            )
            step_time = time.time()
            next_observation, reward, terminal, truncated, info = self.env.step(action)
            self._stats["step_time_ms"].append((time.time() - step_time) * 1000)
            self._next_observation = next_observation
            done = terminal or truncated
            self._length += 1
            self._cumreward += reward
            num_sampled_timesteps += 1
            self._stats["num_timesteps"] = num_sampled_timesteps
            # Callbacks
            for callback in self.callbacks:
                callback.on_episode_step(self._stats, info)

            if done:
                self._stats["episode_length"].append(self._length)
                self._stats["episode_cumreward"].append(self._cumreward)
                self._length = 0
                self._cumreward = 0.0
                num_sampled_episodes += 1
                self._stats["num_episodes"] = num_sampled_episodes
                self.agent.reset()
                # Callbacks
                for callback in self.callbacks:
                    callback.on_episode_end(self._stats)
                self._next_observation, info = self.env.reset()
                # Callbacks
                for callback in self.callbacks:
                    callback.on_episode_start(self._stats, info)

            experience = {
                "observation": observation,
                "next_observation": next_observation,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "terminal": terminal,
            }
            yield experience, done

    def _batched_sample(
        self,
        num_timesteps: "Union[int, float]",
        num_episodes: "Union[int, float]",
    ) -> "Iterator[Tuple[Dict[str, Any], ndarray]]":
        num_sampled_timesteps, num_sampled_episodes = 0, 0
        while (
            num_sampled_timesteps < num_timesteps
            and num_sampled_episodes < num_episodes
        ):
            if self._next_observation is None:
                # First call
                self._next_observation, info = self.env.reset()
                # Callbacks
                for callback in self.callbacks:
                    for i in range(self._batch_size):
                        callback.on_episode_start(self._stats, info[i])
            observation = self._next_observation
            inference_time = time.time()
            action, log_prob = self.agent(observation)
            self._stats["inference_time_ms"].append(
                (time.time() - inference_time) * 1000
            )
            step_time = time.time()
            next_observation, reward, terminal, truncated, info = self.env.step(action)
            self._stats["step_time_ms"].append((time.time() - step_time) * 1000)
            self._next_observation = next_observation
            done = terminal | truncated
            self._length += 1
            self._cumreward += reward
            num_sampled_timesteps += self._batch_size
            self._stats["num_timesteps"] = num_sampled_timesteps
            # Callbacks
            for callback in self.callbacks:
                for i in range(self._batch_size):
                    callback.on_episode_step(self._stats, info[i])

            if done.any():
                self._stats["episode_length"] += self._length[done].tolist()
                self._stats["episode_cumreward"] += self._cumreward[done].tolist()
                self._length[done] = 0
                self._cumreward[done] = 0.0
                num_sampled_episodes += done.sum()
                self._stats["num_episodes"] = num_sampled_episodes
                self.agent.reset(done)
                # Callbacks
                if self.callbacks:
                    done_idxes = np.where(done)[0]
                    for callback in self.callbacks:
                        for _ in done_idxes:
                            callback.on_episode_end(self._stats)
                self._next_observation, info = self.env.reset(done)
                # Callbacks
                if self.callbacks:
                    for callback in self.callbacks:
                        for i in done_idxes:
                            callback.on_episode_start(self._stats, info[i])

            experience = {
                "observation": observation,
                "next_observation": next_observation,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "terminal": terminal,
            }
            yield experience, done

    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(env: {self.env}, "
            f"agent: {self.agent}, "
            f"callbacks: {self.callbacks})"
        )
