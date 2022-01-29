# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Parallel batched environment."""

import multiprocessing as mp
from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Tuple

import cloudpickle
import numpy as np
from gym import Env
from numpy import ndarray

from actorch.envs.batched_env import BatchedEnv
from actorch.envs.utils import Nested, flatten, unflatten


__all__ = [
    "ParallelBatchedEnv",
]


class _WorkerSharedMemory(mp.Process):
    """Worker process that runs an environment.

    Observations are sent back to the main process via shared memory.

    """

    def __init__(
        self,
        env_builder: "Callable[[], Env]",
        connection: "mp.connection.Connection",
        shared_memory: "mp.RawArray",
        idx: "int",
        daemon: "Optional[bool]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        env_builder:
            The environment builder.
        connection:
            The connection for communicating with the main process.
        shared_memory:
            The shared memory for sending observations back to the
            main process.
        idx:
            The worker index, used to locate the destination in the
            shared memory to write observations to.
        daemon:
            True to run as a daemonic process, False otherwise.

        """
        super().__init__(daemon=daemon)
        self._pickled_env_builder = cloudpickle.dumps(env_builder)
        self._connection = connection
        self._shared_memory = shared_memory
        self._idx = idx

    # override
    def run(self) -> "None":
        env = cloudpickle.loads(self._pickled_env_builder)()
        del self._pickled_env_builder

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

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__[1:]}()"


class ParallelBatchedEnv(BatchedEnv):
    """Batched environment based on subprocesses."""

    def __init__(
        self, env_builder: "Callable[[], Env]", num_workers: "int" = 1
    ) -> "None":
        super().__init__(env_builder, num_workers)
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
            process = _WorkerSharedMemory(
                self.env_builder,
                child,
                shared_memory,
                i,
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

    # override
    def _reset(self, idx: "Sequence[int]") -> "Nested[ndarray]":
        for i in idx:
            self._parents[i].send(["reset", None])
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
