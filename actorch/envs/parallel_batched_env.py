# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Parallel batched environment."""

import atexit
import multiprocessing as mp
import time
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cloudpickle
import numpy as np
from gym import Env
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.utils import Nested, flatten, unflatten


__all__ = [
    "ParallelBatchedEnv",
]


class WorkerSharedMemory(mp.Process):
    """Worker process that runs an environment.

    Observations are sent back to the main process through shared memory.

    """

    def __init__(
        self,
        env_builder: "Callable[..., Env]",
        connection: "mp.connection.Connection",
        shared_memory: "mp.RawArray",
        idx: "int",
        env_config: "Optional[Dict[str, Any]]" = None,
        daemon: "Optional[bool]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        env_builder:
            The environment builder, i.e. a callable that
            receives keyword arguments from a configuration
            and returns a base environment.
        connection:
            The connection for communicating with the main process.
        shared_memory:
            The shared memory for sending observations back to the
            main process.
        idx:
            The worker index, used to locate the destination in the
            shared memory to write observations to.
        env_config:
            The environment configuration.
            Default to ``{}``.
        daemon:
            True to run as a daemonic process, False otherwise.

        """
        super().__init__(daemon=daemon)
        self._pickled_env_builder = cloudpickle.dumps(env_builder)
        self._pickled_env_config = cloudpickle.dumps(env_config or {})
        self._connection = connection
        self._shared_memory = shared_memory
        self._idx = idx

    # override
    def run(self) -> "None":
        env = cloudpickle.loads(self._pickled_env_builder)(
            **cloudpickle.loads(self._pickled_env_config)
        )
        del self._pickled_env_builder
        del self._pickled_env_config

        # Locate destination in shared memory to write observations to
        dummy_flat_observation = flatten(
            env.observation_space, env.observation_space.sample(), copy=False
        )
        size = dummy_flat_observation.shape[0]
        dtype = dummy_flat_observation.dtype
        del dummy_flat_observation
        destination = np.frombuffer(
            self._shared_memory, dtype, size, (self._idx * size) * dtype.itemsize
        )
        del self._idx

        try:
            while True:
                command, data = self._connection.recv()
                if command == "reset":
                    observation = env.reset()
                    flat_observation = flatten(
                        env.observation_space, observation, copy=False
                    )
                    np.copyto(destination, flat_observation)
                    # Send (result, has_errored) to the main process
                    self._connection.send((None, False))
                elif command == "step":
                    observation, reward, done, info = env.step(data)
                    flat_observation = flatten(
                        env.observation_space, observation, copy=False
                    )
                    np.copyto(destination, flat_observation)
                    self._connection.send(((reward, done, info), False))
                elif command == "seed":
                    self._connection.send((env.seed(data), False))
                elif command == "render":
                    self._connection.send((env.render(**data), False))
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

    def __init__(
        self,
        env_builder: "Callable[..., Env]",
        env_config: "Optional[Dict[str, Any]]" = None,
        num_workers: "int" = 1,
        timeout_s: "float" = 60.0,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        env_builder:
            The base environment builder, i.e. a callable that
            receives keyword arguments from a configuration
            and returns a base environment.
        env_config:
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
        super().__init__(env_builder, env_config, num_workers)
        if timeout_s <= 0.0:
            raise ValueError(
                f"`timeout_s` ({timeout_s}) must be in the interval (0, inf)"
            )
        self.timeout_s = timeout_s
        dummy_flat_observation = flatten(
            self.single_observation_space,
            self.single_observation_space.sample(),
            copy=False,
        )
        size = len(dummy_flat_observation)
        dtype = dummy_flat_observation.dtype
        typecode = dtype.char
        if typecode == "?":
            from ctypes import c_bool

            typecode = c_bool
        # No locking required, as each child process writes
        # to a non-overlapping portion of the shared memory
        shared_memory = mp.RawArray(typecode, num_workers * size)
        self._parents, self._processes = [], []
        for i in range(num_workers):
            parent, child = mp.Pipe()
            self._parents.append(parent)
            process = WorkerSharedMemory(
                self.env_builder,
                child,
                shared_memory,
                i,
                self.env_config,
                daemon=True,
            )
            self._processes.append(process)
            process.start()
            child.close()  # Close after starting the process
        self._observation_buffer = np.frombuffer(shared_memory, dtype).reshape(
            num_workers, -1
        )
        self._reward = np.zeros(num_workers)
        self._done = np.zeros(num_workers, dtype=bool)
        self._info = np.array([{} for _ in range(num_workers)])
        # True to unflatten the observation, False otherwise
        # Used by FlattenObservation wrapper to improve performance
        self.unflatten_observation = True
        # Fix AttributeError: 'NoneType' object has no attribute 'dumps'
        atexit.register(self.close)

    # override
    def _reset(self, idx: "Sequence[int]") -> "Nested[ndarray]":
        for i in idx:
            self._parents[i].send(["reset", None])
        self._poll(idx)
        errors = []
        for i in idx:
            result, has_errored = self._parents[i].recv()
            if has_errored:
                errors.append((i, result))
        self._raise(errors)
        observation = (
            unflatten(self.observation_space, self._observation_buffer)
            if self.unflatten_observation
            else np.array(self._observation_buffer)
        )  # Copy
        return observation

    # override
    def _step(
        self, action: "Sequence[Nested]", idx: "Sequence[int]"
    ) -> "Tuple[Nested[ndarray], ndarray, ndarray, ndarray]":
        for i in idx:
            self._parents[i].send(["step", action[i]])
        self._poll(idx)
        errors = []
        for i in idx:
            result, has_errored = self._parents[i].recv()
            if has_errored:
                errors.append((i, result))
            else:
                self._reward[i], self._done[i], self._info[i] = result
        self._raise(errors)
        observation = (
            unflatten(self.observation_space, self._observation_buffer)
            if self.unflatten_observation
            else np.array(self._observation_buffer)
        )  # Copy
        return (
            observation,
            np.array(self._reward),  # Copy
            np.array(self._done),  # Copy
            deepcopy(self._info),
        )

    # override
    def _seed(self, seed: "Sequence[Optional[int]]") -> "None":
        for i, parent in enumerate(self._parents):
            parent.send(["seed", seed[i]])
        self._poll()
        errors = []
        for i, parent in enumerate(self._parents):
            result, has_errored = parent.recv()
            if has_errored:
                errors.append((i, result))
        self._raise(errors)

    # override
    def _render(self, idx: "Sequence[int]", **kwargs: "Any") -> "None":
        for i in idx:
            self._parents[i].send(["render", kwargs])
        self._poll(idx)
        errors = []
        for i in idx:
            result, has_errored = self._parents[i].recv()
            if has_errored:
                errors.append((i, result))
        self._raise(errors)

    # override
    def _close(self) -> "None":
        try:
            for parent in self._parents:
                parent.send(["close", None])
            self._poll()
            errors = []
            for i, parent in enumerate(self._parents):
                result, has_errored = parent.recv()
                if has_errored:
                    errors.append((i, result))
            self._raise(errors)
        except (BrokenPipeError, OSError):
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
            if not self._parents[i].poll(timeout):
                error_msgs.append(
                    f"[Worker {i}] Operation has timed out after {self.timeout_s} second(s)"
                )
        if error_msgs:
            self._cleanup()
            raise mp.TimeoutError("\n" + "\n".join(error_msgs))

    def _raise(self, errors: "Sequence[Tuple[int, str]]") -> "None":
        if errors:
            error_msgs = [f"[Worker {i}] {traceback}" for i, traceback in errors]
            self._cleanup()
            raise RuntimeError(f"\n{''.join(error_msgs)}")

    def _cleanup(self) -> "None":
        for parent in self._parents:
            parent.close()
        for process in self._processes:
            if process.is_alive():
                process.terminate()
            process.join()
