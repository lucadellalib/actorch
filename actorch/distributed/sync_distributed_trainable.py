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

"""Synchronous distributed trainable."""

import os
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import ray
import ray.train.torch  # Fix missing train.torch attribute
import torch
from ray import train
from ray.actor import ActorHandle
from ray.tune import PlacementGroupFactory
from ray.tune.integration.torch import logger_creator
from ray.tune.resources import Resources
from ray.tune.result import AUTO_RESULT_KEYS, DONE, TRIAL_ID
from ray.tune.syncer import NodeSyncer, get_node_syncer
from ray.tune.trainable import Trainable
from ray.tune.utils.placement_groups import resource_dict_to_pg_factory
from ray.tune.utils.trainable import TrainableUtil
from torch import distributed as dist

from actorch.distributed.distributed_trainable import DistributedTrainable, Tunable


__all__ = [
    "SyncDistributedTrainable",
]


_T = TypeVar("_T")


# Adapted from:
# https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/python/ray/tune/integration/torch.py#L47
class SyncDistributedTrainable(DistributedTrainable):
    """Distributed Ray Tune trainable that sets up a PyTorch process group
    and the corresponding workers for synchronous execution.

    Derived classes must implement `get_worker_cls`.

    Warnings
    --------
    Models exported through `export_model` are not automatically synchronized to the
    driver node. You can manually retrieve them by accessing (e.g. through SSH) the
    cluster node where they are stored.

    """

    _REDUCTION_MODES = ["mean", "sum", "single"]

    # override
    class Config(dict):
        """Keyword arguments expected in the configuration received by `setup`."""

        def __init__(
            self,
            num_workers: "Tunable[int]" = 1,
            num_cpus_per_worker: "Tunable[float]" = 1.0,
            num_gpus_per_worker: "Tunable[float]" = 0.0,
            extra_resources_per_worker: "Tunable[Optional[Dict[str, Any]]]" = None,
            placement_strategy: "Tunable[str]" = "PACK",
            reduction_mode: "Tunable[str]" = "mean",
            backend: "Tunable[Optional[str]]" = None,
            timeout_s: "Tunable[float]" = 1800.0,
            node_syncer_builder: "Tunable[Optional[Callable[..., NodeSyncer]]]" = None,
            node_syncer_config: "Tunable[Optional[Dict[str, Any]]]" = None,
            worker_init_fn: "Tunable[Optional[Callable[[], None]]]" = None,
            **worker_config: "Any",
        ) -> "None":
            """Initialize the object.

            Parameters
            ----------
            num_workers:
                The number of workers.
            num_cpus_per_worker:
                The number of CPUs to assign to each worker.
            num_gpus_per_worker:
                The number of GPUs to assign to each worker.
            extra_resources_per_worker:
                The extra resources to assign to each worker.
                Default to ``{}``.
            placement_strategy:
                The placement strategy
                (see https://docs.ray.io/en/latest/ray-core/placement-group.html).
            reduction_mode:
                The reduction mode for worker results.
                Must be one of the following:
                - "mean":   compute the key-wise mean of the result dicts returned by each worker
                            (numerical values only, non-numerical ones are copied from the worker
                            with rank 0);
                - "sum":    compute the key-wise sum of the result dicts returned by each worker
                            (numerical values only, non-numerical ones are copied from the worker
                            with rank 0);
                - "single": return the result dict of the worker with rank 0.
            backend:
                The backend for distributed execution
                (see https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group).
                Default to "nccl" if at least 1 GPU is available for each process, "gloo" otherwise.
            timeout_s:
                The timeout in seconds for distributed operations
                (see https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group).
            node_syncer_builder:
                The node syncer builder, i.e. a callable that receives keyword arguments
                from a configuration and returns a node syncer that synchronizes log
                directories of workers running on remote nodes.
                If all workers run on the same node, synchronization is not performed.
                Default to ``ray.tune.syncer.get_node_syncer``.
            node_syncer_config:
                The node syncer configuration.
                Arguments `local_dir` and `remote_dir` are set internally.
                Default to ``{"sync_function": True}`` if `node_syncer_builder`
                is None, ``{}`` otherwise.
            worker_init_fn:
                The worker initialization function, i.e. a callable
                that receives no arguments and returns None.
                Useful, for example, to set environment variables.
            worker_config:
                The worker configuration.
                Default to ``{}``.

            """
            super().__init__(
                num_workers=num_workers,
                num_cpus_per_worker=num_cpus_per_worker,
                num_gpus_per_worker=num_gpus_per_worker,
                extra_resources_per_worker=extra_resources_per_worker,
                placement_strategy=placement_strategy,
                reduction_mode=reduction_mode,
                backend=backend,
                timeout_s=timeout_s,
                node_syncer_builder=node_syncer_builder,
                node_syncer_config=node_syncer_config,
                worker_init_fn=worker_init_fn,
                worker_config=worker_config,
            )

    # override
    @classmethod
    def default_resource_request(
        cls,
        config: "Dict[str, Any]",
    ) -> "PlacementGroupFactory":
        config = SyncDistributedTrainable.Config(**config)
        worker_cls = cls.get_worker_cls()
        default_resource_request = worker_cls.default_resource_request(
            config["worker_config"]
        )
        if isinstance(default_resource_request, Resources):
            default_resource_request = resource_dict_to_pg_factory(
                default_resource_request
            )
        default_resources: "Dict[str, float]" = (
            default_resource_request.required_resources
            if default_resource_request
            else {}
        )
        # Define resources per worker
        num_workers = config["num_workers"]
        if num_workers < 1 or not float(num_workers).is_integer():
            raise ValueError(
                f"`num_workers` ({num_workers}) must be in the integer interval [1, inf)"
            )
        num_workers = int(num_workers)
        num_cpus_per_worker = config["num_cpus_per_worker"]
        if num_cpus_per_worker < 1.0:
            raise ValueError(
                f"`num_cpus_per_worker` ({num_cpus_per_worker}) must be in the interval [1, inf)"
            )
        num_cpus_per_worker = max(num_cpus_per_worker, default_resources.pop("CPU", 1))
        num_gpus_per_worker = config["num_gpus_per_worker"]
        if num_gpus_per_worker < 0.0:
            raise ValueError(
                f"`num_gpus_per_worker` ({num_gpus_per_worker}) must be in the interval [0, inf)"
            )
        num_gpus_per_worker = max(num_gpus_per_worker, default_resources.pop("GPU", 0))
        extra_resources_per_worker = config["extra_resources_per_worker"] or {}
        for k, v in extra_resources_per_worker:
            extra_resources_per_worker[k] = max(v, default_resources.pop(k, 0))
        placement_strategy = config["placement_strategy"]
        bundles = [{"CPU": 1}]
        bundles += [
            {
                "CPU": num_cpus_per_worker,
                "GPU": num_gpus_per_worker,
                **extra_resources_per_worker,
            }
        ] * num_workers
        return super().default_resource_request(
            super().Config(bundles, placement_strategy)
        )

    # override
    @classmethod
    def resource_help(cls, config: "Dict[str, Any]") -> "str":
        return (
            "The following keyword arguments should be given to configure the resources:"
            + "\n"
            "    num_workers:" + "\n"
            "        The number of workers." + "\n"
            "    num_cpus_per_worker:" + "\n"
            "        The number of CPUs to assign to each worker." + "\n"
            "    num_gpus_per_worker:" + "\n"
            "        The number of GPUs to assign to each worker." + "\n"
            "    extra_resources_per_worker:" + "\n"
            "        The extra resources to assign to each worker." + "\n"
            "        Default to ``{}``." + "\n"
            "    placement_strategy:" + "\n"
            "        The placement strategy" + "\n"
            "        (see https://docs.ray.io/en/latest/ray-core/placement-group.html)."
        )

    # override
    def setup(self, config: "Dict[str, Any]") -> "None":
        config = SyncDistributedTrainable.Config(**config)  # Copy of self.config
        bundles = self.default_resource_request(config).bundles
        super().setup(super().Config(bundles, config["placement_strategy"]))
        self.config = deepcopy(config)
        self.num_workers = config["num_workers"]
        self.num_cpus_per_worker = config["num_cpus_per_worker"]
        self.num_gpus_per_worker = config["num_gpus_per_worker"]
        self.extra_resources_per_worker = config["extra_resources_per_worker"] or {}
        self.reduction_mode = config["reduction_mode"]
        if self.reduction_mode not in self._REDUCTION_MODES:
            raise ValueError(
                f"`reduction_mode` ({self.reduction_mode}) must be one of {self._REDUCTION_MODES}"
            )
        self.backend = config["backend"] or (
            "nccl" if self.num_gpus_per_worker > 0 else "gloo"
        )
        self.timeout_s = config["timeout_s"]
        if self.timeout_s <= 0:
            raise ValueError(
                f"`timeout_s` ({self.timeout_s}) must be in the interval (0, inf)"
            )
        self.node_syncer_builder = config["node_syncer_builder"] or get_node_syncer
        self.node_syncer_config = (
            config["node_syncer_config"]
            if config["node_syncer_config"] is not None
            else (
                {"sync_function": True} if config["node_syncer_builder"] is None else {}
            )
        )
        self.worker_init_fn = config["worker_init_fn"]
        self.worker_config = config["worker_config"]

        # Setup options
        options = {
            "num_cpus": bundles[1].pop("CPU", 1),
            "num_gpus": bundles[1].pop("GPU", 0),
            "resources": bundles[1],
        }

        # Set placement group if created manually
        if self._placement_group:
            options["placement_group"] = self._placement_group

        self._workers = self._setup_workers(options)

        # If multi-node, set up node syncer
        ip = self.get_current_ip()
        self._remote_node_ips = list(
            filter(
                lambda worker_ip: worker_ip != ip,
                ray.get([worker.get_current_ip.remote() for worker in self._workers]),
            )
        )
        self._node_syncer = None
        if self._remote_node_ips:
            self._node_syncer = self.node_syncer_builder(
                self.logdir, self.logdir, **self.node_syncer_config
            )

    # override
    def reset_config(self, new_config: "Dict[str, Any]") -> "bool":
        new_config = SyncDistributedTrainable.Config(**new_config)
        if any(
            [
                self.num_workers != new_config["num_workers"],
                self.num_cpus_per_worker != new_config["num_cpus_per_worker"],
                self.num_gpus_per_worker != new_config["num_gpus_per_worker"],
                self.extra_resources_per_worker
                != new_config["extra_resources_per_worker"],
                self.placement_strategy != new_config["placement_strategy"],
                self.backend != new_config["backend"],
                self.timeout_s != new_config["timeout_s"],
                self.worker_init_fn.__code__ != new_config["worker_init_fn"].__code__,
            ]
        ):
            return False

        self.config = deepcopy(new_config)
        self.reduction_mode = new_config["reduction_mode"]
        if self.reduction_mode not in self._REDUCTION_MODES:
            raise ValueError(
                f"`reduction_mode` ({self.reduction_mode}) must be one of {self._REDUCTION_MODES}"
            )
        self.node_syncer_builder = new_config["node_syncer_builder"] or get_node_syncer
        self.node_syncer_config = (
            new_config["node_syncer_config"]
            if new_config["node_syncer_config"] is not None
            else (
                {"sync_function": True}
                if new_config["node_syncer_builder"] is None
                else {}
            )
        )
        self.worker_config = new_config["worker_config"]

        if self._node_syncer:
            self._node_syncer = self.node_syncer_builder(
                self.logdir, self.logdir, **self.node_syncer_config
            )

        new_worker_config = self.build_config(self.worker_config)
        return all(
            ray.get(
                [
                    worker.reset_config.remote(new_worker_config)
                    for worker in self._workers
                ]
            )
        )

    # override
    def step(self) -> "Dict[str, Any]":
        results = ray.get([worker.train.remote() for worker in self._workers])
        for result in results:
            for key in [*AUTO_RESULT_KEYS, TRIAL_ID, DONE, "perf"]:
                result.pop(key, None)
        result = self._reduce(results) if len(results) > 1 else results[0]
        return result

    # override
    def save_checkpoint(self, tmp_checkpoint_dir: "str") -> "str":
        checkpoint_obj = ray.get(self._workers[0].save_to_object.remote())
        checkpoint_file = TrainableUtil.create_from_pickle(
            checkpoint_obj, tmp_checkpoint_dir
        )
        for ip in self._remote_node_ips:
            self._node_syncer.set_worker_ip(ip)
            self._node_syncer.sync_down()
        return checkpoint_file

    # override
    def load_checkpoint(self, checkpoint: "str") -> "None":
        checkpoint_obj = TrainableUtil.checkpoint_to_object(checkpoint)
        ray.get(
            [
                worker.restore_from_object.remote(checkpoint_obj)
                for worker in self._workers
            ]
        )

    # override
    def cleanup(self) -> "None":
        ray.get([worker.stop.remote() for worker in self._workers])
        if self._node_syncer:
            self._node_syncer.close()
        super().cleanup()

    # override
    def export_model(
        self,
        export_formats: "Union[str, Sequence[str]]",
        export_dir: "Optional[str]" = None,
    ) -> "Dict[str, str]":
        return ray.get(self._workers[0].export_model.remote(export_formats, export_dir))

    def _setup_workers(self, options: "Dict[str, Any]") -> "List[ActorHandle]":
        @ray.remote
        class Worker(self.get_worker_cls()):
            def __init__(self) -> "None":
                # Delayed initialization
                pass

            def init(self, *args: "Any", **kwargs: "Any") -> "None":
                super().__init__(*args, **kwargs)

            def execute(
                self, fn: "Callable[..., _T]", *fn_args: "Any", **fn_kwargs: "Any"
            ) -> "_T":
                return fn(*fn_args, **fn_kwargs)

        workers = [
            Worker.options(**options).remote() for rank in range(self.num_workers)
        ]

        # Run worker initialization function
        if self.worker_init_fn:
            ray.get([worker.execute.remote(self.worker_init_fn) for worker in workers])

        if self.num_workers > 1:
            if not dist.is_available():
                raise RuntimeError(
                    "`torch.distributed` is not available, "
                    "see https://pytorch.org/docs/stable/distributed.html"
                )

            # Retrieve master address
            ip, port = ray.get(
                workers[0].execute.remote(train.utils.get_address_and_port)
            )

            # Setup process group
            ray.get(
                [
                    worker.execute.remote(
                        train.torch.setup_torch_process_group,
                        backend=self.backend,
                        world_rank=rank,
                        world_size=self.num_workers,
                        init_method=f"tcp://{ip}:{port}",
                        timeout_s=self.timeout_s,
                    )
                    for rank, worker in enumerate(workers)
                ]
            )

        # Setup workers
        worker_config = self.build_config(self.worker_config)
        workers_dir = os.path.join(self.logdir, "workers")
        ray.get(
            [
                worker.init.remote(
                    config=worker_config,
                    logger_creator=lambda config: logger_creator(
                        worker_config, workers_dir, rank
                    ),
                )
                for rank, worker in enumerate(workers)
            ]
        )
        return workers

    def _reduce(self, results: "Sequence[Dict[str, Any]]") -> "Dict[str, Any]":
        reduced = results[0]
        if self.reduction_mode == "single":
            return reduced
        for k, v in reduced.items():
            if isinstance(v, dict):
                reduced[k] = self._reduce(
                    [result[k] for result in results if k in result]
                )
                continue
            if isinstance(v, np.ndarray):
                value = np.concatenate([result[k] for result in results if k in result])
            elif isinstance(v, torch.Tensor):
                value = torch.cat(
                    [result[k] for result in results if k in result]
                ).numpy()
            else:
                value = np.asarray([result[k] for result in results if k in result])
            try:
                reduced[k] = (
                    value.all()
                    if value.dtype == bool
                    else (
                        value.mean(axis=0)
                        if self.reduction_mode == "mean"
                        else value.sum(axis=0)
                    )
                )
            except Exception:
                pass
        return reduced

    @classmethod
    @abstractmethod
    def get_worker_cls(cls) -> "Type[Trainable]":
        """Return the worker class (i.e. the trainable class
        to run in a synchronous distributed fashion).

        Returns
        -------
            The worker class.

        """
        raise NotImplementedError
