# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Batched experience sampler."""

import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from actorch.agents import Agent
from actorch.envs import BatchedEnv
from actorch.samplers.callbacks import Callback
from actorch.samplers.sampler import Sampler


__all__ = [
    "BatchedSampler",
]


class BatchedSampler(Sampler):
    """Sampler that samples batched experiences from a batched environment."""

    def __init__(
        self,
        env: "BatchedEnv",
        agent: "Agent",
        callbacks: "Optional[Sequence[Callback]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        env:
            The batched environment to sample batched experiences from.
        agent:
            The agent that interacts with the batched environment.
        callbacks:
            The experience sampler callbacks.
            Default to ``[]``.

        Raises
        ------
        TypeError
            If `env` is not instance of `actorch.envs.BatchedEnv`.

        """
        if not isinstance(env, BatchedEnv):
            raise TypeError(
                f"`env` ({env}) must be instance of "
                f"`{BatchedEnv.__module__}.{BatchedEnv.__name__}`"
            )
        self._batch_size = len(env)
        try:
            super().__init__(env, agent, callbacks)
        except TypeError:
            pass

    # override
    def reset(self) -> "None":
        super().reset()
        self._length = np.zeros(self._batch_size, dtype=int)
        self._cumreward = np.zeros(self._batch_size)

    # override
    def sample(
        self,
        num_timesteps: "Optional[Union[int, float]]" = None,
        num_episodes: "Optional[Union[int, float]]" = None,
        render: "bool" = False,
    ) -> "Iterator[Tuple[Dict[str, Any], ndarray]]":
        """Sample batched experiences from the batched environment
        for a given number of timesteps or episodes.

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
            - The sampled batched experience, i.e. a dict with the
              following key-value pairs:
              - "observation":      the batched observation;
              - "action":           the batched action;
              - "log_prob":         the batched action log probability;
              - "reward":           the batched reward;
              - "next_observation": the batched next observation;
              - "done":             the batched end-of-episode flag;
            - the batched terminal flag (might differ from `done` if
              the environment has a time limit).

        Warnings
        --------
        Since experiences are sampled in batches of size `B`,
        up to ``B - 1`` extra timesteps or episodes might be sampled.

        """
        yield from super().sample(num_timesteps, num_episodes, render)

    # override
    def _sample(
        self,
        num_timesteps: "Union[int, float]",
        num_episodes: "Union[int, float]",
        render: "bool",
    ) -> "Iterator[Tuple[Dict[str, Any], ndarray]]":
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
                    for _ in range(self._batch_size):
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
            num_sampled_timesteps += self._batch_size
            self._stats["num_timesteps"] = num_sampled_timesteps
            # Callbacks
            for callback in self.callbacks:
                for i in range(self._batch_size):
                    callback.on_episode_step(self._stats, info[i])

            if terminal.any():
                self._next_observation = self.env.reset(terminal)
                if render:
                    render_time = time.time()
                    self.env.render(terminal)
                    self._stats["render_time_ms"] += (time.time() - render_time) * 1000
                self._stats["episode_lengths"] += self._length[terminal].tolist()
                self._stats["episode_cumrewards"] += self._cumreward[terminal].tolist()
                self._length[terminal] = 0
                self._cumreward[terminal] = 0.0
                num_sampled_episodes += terminal.sum()
                self._stats["num_episodes"] = num_sampled_episodes
                self.agent.reset(terminal)
                done = ~np.array(
                    info[i].get("TimeLimit.truncated", not terminal[i])
                    for i in range(self._batch_size)
                )
                # Callbacks
                if self.callbacks:
                    terminal_idxes = np.where(terminal)[0]
                    for callback in self.callbacks:
                        for _ in terminal_idxes:
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
