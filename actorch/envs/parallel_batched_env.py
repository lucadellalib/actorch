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

"""Parallel batched environment."""

import atexit
import multiprocessing as mp
import signal
import time
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cloudpickle
import numpy as np
from gymnasium import Env
from gymnasium.utils import seeding
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.utils import Nested, flatten, unflatten


__all__ = [
    "ParallelBatchedEnv",
]


class EnvWorkerSharedMemory(mp.Process):
    """Worker process that runs an environment.

    Observations are sent back to the main process through shared memory.

    """

    # override
    def __init__(
        self,
        connection: "mp.connection.Connection",
        shared_memory: "mp.RawArray",
        idx: "int",
        env_builder: "Callable[..., Env]",
        env_config: "Dict[str, Any]",
        daemon: "Optional[bool]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        connection:
            The connection for communicating with the main process.
        shared_memory:
            The shared memory for sending observations back to the
            main process.
        idx:
            The environment worker index, used to locate the destination
            in the shared memory to write observations to.
        env_builder:
            The environment builder, i.e. a callable that
            receives keyword arguments from an environment
            configuration and returns an environment.
        env_config:
            The environment configuration.
        daemon:
            True to run as a daemonic process, False otherwise.
            If None, its value is inherited from the creating process.

        """
        super().__init__(daemon=daemon)
        self._connection = connection
        self._shared_memory = shared_memory
        self._idx = idx
        self._pickled_env_builder = cloudpickle.dumps(env_builder)
        self._pickled_env_config = cloudpickle.dumps(env_config)

    # override
    def run(self) -> "None":
        env = cloudpickle.loads(self._pickled_env_builder)(
            **cloudpickle.loads(self._pickled_env_config),
        )
        del self._pickled_env_builder
        del self._pickled_env_config

        # Locate destination in shared memory to write observations to
        dummy_flat_observation = flatten(
            env.observation_space, env.observation_space.sample(), copy=False
        )
        is_atleast_1d = dummy_flat_observation.ndim > 0
        size = len(dummy_flat_observation) if is_atleast_1d else 1
        dtype = dummy_flat_observation.dtype
        destination = np.frombuffer(
            self._shared_memory, dtype, size, (self._idx * size) * dtype.itemsize
        )
        del dummy_flat_observation
        del is_atleast_1d
        del size
        del dtype
        del self._idx

        try:
            while True:
                command, data = self._connection.recv()
                if command == "reset":
                    observation, info = env.reset()
                    flat_observation = flatten(
                        env.observation_space, observation, copy=False
                    )
                    np.copyto(destination, flat_observation)
                    # Send (result, has_errored) to the main process
                    self._connection.send((info, False))
                elif command == "step":
                    observation, reward, terminal, truncated, info = env.step(data)
                    flat_observation = flatten(
                        env.observation_space, observation, copy=False
                    )
                    np.copyto(destination, flat_observation)
                    self._connection.send(((reward, terminal, truncated, info), False))
                elif command == "seed":
                    # Bypass deprecation warning
                    env.np_random, _ = seeding.np_random(data)
                    self._connection.send((None, False))
                elif command == "close":
                    self._connection.send((None, False))
                    break
                else:
                    raise NotImplementedError(f"Unknown command: {command}")
        except (Exception, KeyboardInterrupt):
            import traceback

            self._connection.send((traceback.format_exc(), True))
        finally:
            env.close()
            self._connection.close()


class ParallelBatchedEnv(BatchedEnv):
    """Batched environment based on subprocesses."""

    _EnvWorker = EnvWorkerSharedMemory

    # override
    def __init__(
        self,
        base_env_builder: "Callable[..., Env]",
        base_env_config: "Optional[Dict[str, Any]]" = None,
        num_workers: "int" = 1,
        timeout_s: "float" = 60.0,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        base_env_builder:
            The base environment builder, i.e. a callable that
            receives keyword arguments from a base environment
            configuration and returns a base environment.
        base_env_config:
            The base environment configuration.
            Default to ``{}``.
        num_workers:
            The number of copies of the base environment.
        timeout_s:
            The timeout in seconds for worker operations.

        Raises
        ------
        ValueError
            If `timeout_s` is not in the interval (0, inf).

        """
        super().__init__(base_env_builder, base_env_config, num_workers)
        if timeout_s <= 0.0:
            raise ValueError(
                f"`timeout_s` ({timeout_s}) must be in the interval (0, inf)"
            )
        self.timeout_s = timeout_s
        self._repr = f"{self._repr[:-2]}, timeout_s: {timeout_s})>"
        dummy_flat_observation = flatten(
            self.single_observation_space,
            self.single_observation_space.sample(),
            copy=False,
        )
        is_atleast_1d = dummy_flat_observation.ndim > 0
        size = len(dummy_flat_observation) if is_atleast_1d else 1
        dtype = dummy_flat_observation.dtype
        typecode = dtype.char
        if typecode == "?":
            from ctypes import c_bool

            typecode = c_bool
        # No locking required, as each environment worker writes
        # to a non-overlapping portion of the shared memory
        shared_memory = mp.RawArray(typecode, num_workers * size)
        self._connections, self._env_workers = [], []
        for i in range(num_workers):
            connection, child = mp.Pipe()
            self._connections.append(connection)
            env_worker = self._EnvWorker(
                child,
                shared_memory,
                i,
                self.base_env_builder,
                self.base_env_config,
                daemon=True,
            )
            self._env_workers.append(env_worker)
            env_worker.start()
            child.close()  # Close after starting the environment worker
        self._observation_buffer = np.frombuffer(shared_memory, dtype).reshape(
            (num_workers,) + ((-1,) if is_atleast_1d else ())
        )
        self._reward = np.zeros(num_workers)
        self._terminal = np.zeros(num_workers, dtype=bool)
        self._truncated = np.zeros(num_workers, dtype=bool)
        self._info = np.array([{} for _ in range(num_workers)])
        # True to unflatten the observation, False otherwise
        # Used by FlattenObservation wrapper to improve performance
        self.unflatten_observation = True
        # Fix AttributeError: 'NoneType' object has no attribute 'dumps'
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, self.close)
        signal.signal(signal.SIGINT, self.close)

    # override
    def _reset(self, idx: "Sequence[int]") -> "Tuple[Nested[ndarray], ndarray]":
        for i in idx:
            self._connections[i].send(["reset", None])
        self._poll(idx)
        errors = []
        for i in idx:
            result, has_errored = self._connections[i].recv()
            if has_errored:
                errors.append((i, result))
            else:
                self._info[i] = result
        self._raise(errors)
        observation = (
            unflatten(self.observation_space, self._observation_buffer, is_batched=True)
            if self.unflatten_observation
            else np.array(self._observation_buffer)
        )  # Copy
        return observation, deepcopy(self._info)

    # override
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray, ndarray]":
        for i in idx:
            self._connections[i].send(["step", action[i]])
        self._poll(idx)
        errors = []
        for i in idx:
            result, has_errored = self._connections[i].recv()
            if has_errored:
                errors.append((i, result))
            else:
                (
                    self._reward[i],
                    self._terminal[i],
                    self._truncated[i],
                    self._info[i],
                ) = result
        self._raise(errors)
        observation = (
            unflatten(self.observation_space, self._observation_buffer, is_batched=True)
            if self.unflatten_observation
            else np.array(self._observation_buffer)
        )  # Copy
        return (
            observation,
            np.array(self._reward),  # Copy
            np.array(self._terminal),  # Copy
            np.array(self._truncated),  # Copy
            deepcopy(self._info),
        )

    # override
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        for i, connection in enumerate(self._connections):
            connection.send(["seed", seed[i]])
        self._poll()
        errors = []
        for i, connection in enumerate(self._connections):
            result, has_errored = connection.recv()
            if has_errored:
                errors.append((i, result))
        self._raise(errors)

    # override
    def _close(self) -> "None":
        try:
            for connection in self._connections:
                connection.send(["close", None])
            self._poll()
            errors = []
            for i, connection in enumerate(self._connections):
                result, has_errored = connection.recv()
                if has_errored:
                    errors.append((i, result))
            self._raise(errors)
        except OSError:
            pass
        finally:
            self._cleanup()

    def _poll(self, idx: "Optional[Sequence[int]]" = None) -> "None":
        if idx is None:
            idx = range(self.num_workers)
        error_msgs = []
        end_time = time.perf_counter() + self.timeout_s
        for i in idx:
            timeout = max(end_time - time.perf_counter(), 0)
            if not self._connections[i].poll(timeout):
                error_msgs.append(
                    f"[Environment worker {i}] Operation has timed out after {self.timeout_s} second(s)"
                )
        if error_msgs:
            self._cleanup()
            raise mp.TimeoutError("\n" + "\n".join(error_msgs))

    def _raise(self, errors: "Sequence[Tuple[int, str]]") -> "None":
        if errors:
            error_msgs = [
                f"[Environment worker {i}] {traceback}" for i, traceback in errors
            ]
            self._cleanup()
            raise RuntimeError(f"\n{''.join(error_msgs)}")

    def _cleanup(self) -> "None":
        for connection in self._connections:
            connection.close()
        for env_worker in self._env_workers:
            env_worker.join()
