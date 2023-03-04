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

"""Deep reinforcement learning algorithm."""

import contextlib
import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import ray
import torch
import torch.nn.functional as F
from gymnasium import Env, spaces
from numpy import ndarray
from ray.train import world_rank
from ray.train.torch import accelerate, prepare_data_loader
from ray.tune import Trainable
from ray.tune import result as tune_result
from ray.tune.sample import Domain
from ray.tune.syncer import NodeSyncer
from ray.tune.trial import ExportFormat
from torch import Tensor
from torch.cuda.amp import autocast
from torch.distributions import Bernoulli, Categorical, Distribution, Normal
from torch.profiler import profile, record_function, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from actorch.agents import Agent, StochasticAgent
from actorch.algorithms.utils import (
    count_params,
    init_mock_train_session,
    prepare_model,
)
from actorch.buffers import Buffer, UniformBuffer
from actorch.datasets import BufferDataset
from actorch.distributed import SyncDistributedTrainable
from actorch.envs import (
    BatchedEnv,
    BatchedFlattenObservation,
    BatchedUnflattenAction,
    SerialBatchedEnv,
)
from actorch.models import FCNet, Model
from actorch.networks import (
    DistributionParametrization,
    Identity,
    Independent,
    NormalizingFlow,
    OneHotEncode,
    PolicyNetwork,
    Processor,
)
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, Schedule
from actorch.utils import FutureRef, singledispatchmethod


__all__ = [
    "Algorithm",
    "DistributedDataParallelAlgorithm",
    "RefOrFutureRef",
    "Tunable",
]


_LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")

_GridSearch = Dict[str, List[_T]]

Tunable = Union[_T, Domain, _GridSearch[_T]]
"""Ray Tune tunable argument."""

RefOrFutureRef = Union[_T, FutureRef[_T]]
"""Reference or future reference."""


class Algorithm(ABC, Trainable):
    """Deep reinforcement learning algorithm."""

    _EXPORT_FORMATS = [ExportFormat.CHECKPOINT, ExportFormat.MODEL]

    _UPDATE_BUFFER_DATASET_SCHEDULES_AFTER_TRAIN_EPOCH = True

    class Config(dict):
        """Keyword arguments expected in the configuration received by `setup`."""

        def __init__(
            self,
            train_env_builder: "Tunable[RefOrFutureRef[Callable[..., Union[Env, BatchedEnv]]]]",
            train_env_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            train_agent_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Agent]]]]" = None,
            train_agent_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            train_sampler_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Sampler]]]]" = None,
            train_sampler_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            train_num_timesteps_per_iter: "Tunable[RefOrFutureRef[Optional[Union[int, float, Schedule]]]]" = None,
            train_num_episodes_per_iter: "Tunable[RefOrFutureRef[Optional[Union[int, float, Schedule]]]]" = None,
            eval_freq: "Tunable[RefOrFutureRef[Optional[int]]]" = 1,
            eval_env_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Union[Env, BatchedEnv]]]]]" = None,
            eval_env_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            eval_agent_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Agent]]]]" = None,
            eval_agent_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            eval_sampler_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Sampler]]]]" = None,
            eval_sampler_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            eval_num_timesteps_per_iter: "Tunable[RefOrFutureRef[Optional[Union[int, float, Schedule]]]]" = None,
            eval_num_episodes_per_iter: "Tunable[RefOrFutureRef[Optional[Union[int, float, Schedule]]]]" = None,
            policy_network_preprocessors: "Tunable[RefOrFutureRef[Optional[Dict[str, Processor]]]]" = None,
            policy_network_model_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Model]]]]" = None,
            policy_network_model_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            policy_network_distribution_builders: "Tunable[RefOrFutureRef[Optional[Dict[str, Callable[..., Distribution]]]]]" = None,
            policy_network_distribution_parametrizations: "Tunable[RefOrFutureRef[Optional[Dict[str, DistributionParametrization]]]]" = None,
            policy_network_distribution_configs: "Tunable[RefOrFutureRef[Optional[Dict[str, Dict[str, Any]]]]]" = None,
            policy_network_normalizing_flows: "Tunable[RefOrFutureRef[Optional[Dict[str, NormalizingFlow]]]]" = None,
            policy_network_sample_fn: "Tunable[RefOrFutureRef[Optional[Callable[[Distribution], Tensor]]]]" = None,
            policy_network_prediction_fn: "Tunable[RefOrFutureRef[Optional[Callable[[Tensor], Tensor]]]]" = None,
            policy_network_postprocessors: "Tunable[RefOrFutureRef[Optional[Dict[str, Processor]]]]" = None,
            buffer_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Buffer]]]]" = None,
            buffer_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            buffer_checkpoint: "Tunable[RefOrFutureRef[bool]]" = False,
            buffer_dataset_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., BufferDataset]]]]" = None,
            buffer_dataset_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            dataloader_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., DataLoader]]]]" = None,
            dataloader_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            cumreward_window_size: "Tunable[RefOrFutureRef[int]]" = 100,
            seed: "Tunable[RefOrFutureRef[int]]" = 0,
            enable_amp: "Tunable[RefOrFutureRef[Union[bool, Dict[str, Any]]]]" = False,
            enable_reproducibility: "Tunable[RefOrFutureRef[bool]]" = False,
            enable_anomaly_detection: "Tunable[RefOrFutureRef[bool]]" = False,
            enable_profiling: "Tunable[RefOrFutureRef[Union[bool, Dict[str, Any]]]]" = False,
            log_sys_usage: "Tunable[RefOrFutureRef[bool]]" = False,
            suppress_warnings: "Tunable[RefOrFutureRef[bool]]" = False,
            _accept_kwargs: "bool" = False,
            **kwargs: "Any",
        ) -> "None":
            """Initialize the object.

            Parameters
            ----------
            train_env_builder:
                The (possibly batched) training environment builder, i.e.
                a callable that receives keyword arguments from a (possibly
                batched) training environment configuration and returns a
                (possibly batched) training environment.
            train_env_config:
                The (possibly batched) training environment configuration.
                Default to ``{}``.
            train_agent_builder:
                The training agent builder, i.e. a callable that receives
                keyword arguments from a training agent configuration and
                returns a training agent.
                Default to ``actorch.agents.StochasticAgent``.
            train_agent_config:
                The training agent configuration.
                Arguments `policy_network`, `observation_space`
                and `action_space` are set internally.
                Default to ``{
                    "clip_action": True,
                    "device": "cpu",
                    "num_random_timesteps": 0,
                }`` if `train_agent_builder` is None, ``{}`` otherwise.
            train_sampler_builder:
                The training sampler builder, i.e. a callable that receives
                keyword arguments from a training sampler configuration and
                returns a training sampler.
                Default to ``actorch.samplers.Sampler``.
            train_sampler_config:
                The training sampler configuration.
                Arguments `env` and `agent` are set internally.
                Default to ``{"callbacks": []}`` if
                `train_sampler_builder` is None, ``{}`` otherwise.
            train_num_timesteps_per_iter:
                The schedule for the number of timesteps to sample from
                the training environment at each training iteration.
                If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
                Must be None if `train_num_episodes_per_iter` is given.
            train_num_episodes_per_iter:
                The schedule for the number of episodes to sample from
                the training environment at each training iteration.
                If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
                Must be None if `train_num_timesteps_per_iter` is given.
                Default to 1 if `train_num_timesteps_per_iter` is None.
            eval_freq:
                Run evaluation every `eval_freq` training iterations.
                Set to None to skip evaluation.
            eval_env_builder:
                The (possibly batched) evaluation environment builder, i.e.
                a callable that receives keyword arguments from a (possibly
                batched) evaluation environment configuration and returns a
                (possibly batched) evaluation environment.
                Default to `train_env_builder`.
            eval_env_config:
                The (possibly batched) evaluation environment configuration.
                Default to `train_env_config` if `eval_env_builder`
                is None, ``{}`` otherwise.
            eval_agent_builder:
                The evaluation agent builder, i.e. a callable that receives
                keyword arguments from an evaluation agent configuration and
                returns an evaluation agent.
                Default to ``actorch.agents.Agent``.
            eval_agent_config:
                The evaluation agent configuration.
                Arguments `policy_network`, `observation_space`
                and `action_space` are set internally.
                Default to ``{"clip_action": True, "device": "cpu"}`` if
                `eval_agent_builder` is None, ``{}`` otherwise.
            eval_sampler_builder:
                The evaluation sampler builder, i.e. a callable that receives
                keyword arguments from an evaluation sampler configuration and
                returns an evaluation sampler.
                Default to ``actorch.samplers.Sampler``.
            eval_sampler_config:
                The evaluation sampler configuration.
                Arguments `env` and `agent` are set internally.
                Default to ``{"callbacks": []}`` if
                `eval_sampler_builder` is None, ``{}`` otherwise.
            eval_num_timesteps_per_iter:
                The schedule for the number of timesteps to sample from
                the evaluation environment at each training iteration.
                If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
                Must be None if `eval_num_episodes_per_iter` is given.
            eval_num_episodes_per_iter:
                The schedule for the number of episodes to sample from
                the evaluation environment at each training iteration.
                If a number, it is wrapped in an `actorch.schedules.ConstantSchedule`.
                Must be None if `eval_num_timesteps_per_iter` is given.
                Default to 1 if `eval_num_timesteps_per_iter` is None.
            policy_network_preprocessors:
                The policy network preprocessors, i.e. a dict that maps
                names of the policy network input modalities to their
                corresponding policy network preprocessors.
                Default to TODO
            policy_network_model_builder:
                The policy network model builder, i.e. a callable that
                receives keyword arguments from a policy network model
                configuration and returns a policy network model.
                Default to ``actorch.models.FCNet``.
            policy_network_model_config:
                The policy network model configuration.
                Arguments `in_shapes` and `out_shapes` are set internally.
                Default to ``{
                    "torso_fc_configs": [
                        {"out_features": 64, "bias": True},
                        {"out_features": 32, "bias": True},
                    ],
                    "torso_activation_builder": torch.nn.ReLU,
                    "torso_activation_config": {"inplace": False},
                    "head_fc_bias": True,
                    "head_activation_builder": torch.nn.Identity,
                    "head_activation_config": {},
                    "independent_heads": [],
                }`` if `policy_network_model_builder` is None, ``{}`` otherwise.
            policy_network_distribution_builders:
                The policy network distribution builders, i.e. a dict that
                maps names of the policy network output modalities to their
                corresponding policy network distribution builders.
                A policy network distribution builder is a callable that
                receives keyword arguments from a policy network distribution
                parametrization and a policy network distribution configuration
                and returns a policy network distribution.
                Default to TODO
            policy_network_distribution_parametrizations:
                The policy network distribution parametrizations, i.e. a dict
                that maps names of the policy network output modalities to their
                corresponding policy network distribution parametrizations.
                Default to TODO
            policy_network_distribution_configs:
                The policy network distribution configurations, i.e. a dict that
                maps names of the policy network output modalities to their
                corresponding policy network distribution configurations.
                Arguments defined in `policy_network_distribution_parametrizations`
                are set internally.
                Default to ``{}``. TODO
            policy_network_normalizing_flows:
                The policy network normalizing flows, i.e. a dict that maps names
                of the policy network output modalities to their corresponding
                policy network normalizing flows.
                Default to ``{}``. TODO
            policy_network_sample_fn:
                The policy network sample function. It receives as an argument
                the policy network predictive distribution and returns the
                corresponding sample.
                Default to ``lambda distribution: distribution.mode``.
            policy_network_prediction_fn:
                The policy network prediction function. It receives as an
                argument a sample drawn from the policy network predictive
                distribution and returns the corresponding prediction.
                Default to ``lambda sample: sample``.
            policy_network_postprocessors:
                The policy network postprocessors, i.e. a dict that maps
                names of the policy network output modalities to their
                corresponding policy network postprocessors.
                Default to ``{}``. TODO
            buffer_builder:
                The experience replay buffer builder, i.e. a callable that
                receives keyword arguments from an experience replay buffer
                configuration and returns an experience replay buffer.
                Default to ``actorch.buffers.UniformBuffer``.
            buffer_config:
                The experience replay buffer configuration.
                Argument `spec` is set internally.
                Default to ``{"capacity": max(1, len(self._train_env))}``.
            buffer_checkpoint:
                True to include the buffer state dict in the checkpoints,
                False otherwise.
            buffer_dataset_builder:
                The buffer dataset builder, i.e. a callable that receives
                keyword arguments from a buffer dataset configuration and
                returns a buffer dataset.
                Default to ``actorch.datasets.BufferDataset``.
            buffer_dataset_config:
                The buffer dataset configuration.
                Argument `buffer` is set internally.
                Default to ``{
                    "batch_size": 1,
                    "max_trajectory_length": 1,
                    "num_iters": 1,
                }`` if `buffer_dataset_builder` is None, ``{}`` otherwise.
            dataloader_builder:
                The dataloader builder, i.e. a callable that receives
                keyword arguments from a dataloader configuration and
                returns a dataloader.
                Default to ``torch.utils.data.DataLoader``.
            dataloader_config:
                The dataloader configuration.
                Arguments `dataset`, `batch_size`, `shuffle`, `sampler`,
                `batch_sampler`, `collate_fn`, `drop_last` and
                `persistent_workers` are set internally.
                Default to ``{
                    "num_workers": 1 if torch.multiprocessing.get_start_method() == "fork" else 0,
                    "pin_memory": True,
                    "timeout": 0,
                    "worker_init_fn": None,
                    "generator": None,
                    "prefetch_factor": 1 if torch.multiprocessing.get_start_method() == "fork" else 2,
                    "pin_memory_device": "",
                }`` if `dataloader_builder` is None, ``{}`` otherwise.
            cumreward_window_size:
                The window size for the episode cumulative reward moving average.
            seed:
                The seed for generating random numbers.
            enable_amp:
                True to enable automatic mixed precision, False otherwise.
                If dict, it is interpreted as an autocast configuration.
                If True, the following configuration is used: ``{
                    "enabled": True,
                    "dtype": torch.float16,
                    "cache_enabled": True,
                }``.
            enable_reproducibility:
                True to enable full reproducibility, False otherwise.
                Note that it might hamper performance and raise unexpected errors
                (see https://pytorch.org/docs/stable/notes/randomness.html).
            enable_anomaly_detection:
                True to enable anomaly detection, False otherwise.
                Useful, for example, to investigate NaN issues.
            enable_profiling:
                True to enable performance profiling, False otherwise.
                If dict, it is interpreted as a profiler configuration.
                If True, the following configuration is used: ``{
                    "activities": None,
                    "schedule": None,
                    "on_trace_ready": torch.profiler.tensorboard_trace_handler(
                        os.path.join(self.logdir, "profiler")
                    ),
                    "record_shapes": True,
                    "profile_memory": True,
                    "with_stack": False,
                    "with_flops": False,
                    "with_modules": False,
                }``.
            log_sys_usage:
                True to log the system usage statistics, False otherwise.
            suppress_warnings:
                True to suppress warnings, False otherwise.

            """
            if not _accept_kwargs and kwargs:
                raise ValueError(f"Unexpected configuration arguments: {list(kwargs)}")
            super().__init__(
                train_env_builder=train_env_builder,
                train_env_config=train_env_config,
                train_agent_builder=train_agent_builder,
                train_agent_config=train_agent_config,
                train_sampler_builder=train_sampler_builder,
                train_sampler_config=train_sampler_config,
                train_num_timesteps_per_iter=train_num_timesteps_per_iter,
                train_num_episodes_per_iter=train_num_episodes_per_iter,
                eval_freq=eval_freq,
                eval_env_builder=eval_env_builder,
                eval_env_config=eval_env_config,
                eval_agent_builder=eval_agent_builder,
                eval_agent_config=eval_agent_config,
                eval_sampler_builder=eval_sampler_builder,
                eval_sampler_config=eval_sampler_config,
                eval_num_timesteps_per_iter=eval_num_timesteps_per_iter,
                eval_num_episodes_per_iter=eval_num_episodes_per_iter,
                policy_network_preprocessors=policy_network_preprocessors,
                policy_network_model_builder=policy_network_model_builder,
                policy_network_model_config=policy_network_model_config,
                policy_network_distribution_builders=policy_network_distribution_builders,
                policy_network_distribution_parametrizations=policy_network_distribution_parametrizations,
                policy_network_distribution_configs=policy_network_distribution_configs,
                policy_network_normalizing_flows=policy_network_normalizing_flows,
                policy_network_sample_fn=policy_network_sample_fn,
                policy_network_prediction_fn=policy_network_prediction_fn,
                policy_network_postprocessors=policy_network_postprocessors,
                buffer_builder=buffer_builder,
                buffer_config=buffer_config,
                buffer_checkpoint=buffer_checkpoint,
                buffer_dataset_builder=buffer_dataset_builder,
                buffer_dataset_config=buffer_dataset_config,
                dataloader_builder=dataloader_builder,
                dataloader_config=dataloader_config,
                cumreward_window_size=cumreward_window_size,
                seed=seed,
                enable_amp=enable_amp,
                enable_reproducibility=enable_reproducibility,
                enable_anomaly_detection=enable_anomaly_detection,
                enable_profiling=enable_profiling,
                log_sys_usage=log_sys_usage,
                suppress_warnings=suppress_warnings,
                **kwargs,
            )

    @classmethod
    def rename(cls, name: "str") -> "Type[Algorithm]":
        """Return a copy of this class with
        name set to `name`.

        Parameters
        ----------
        name:
            The new name.

        Returns
        -------
            The renamed class.

        """
        return type(name, (cls,), {})

    # override
    def setup(self, config: "Dict[str, Any]") -> "None":
        self.config = Algorithm.Config(**self.config)

        if self.suppress_warnings:
            warnings.filterwarnings("ignore")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._seed()

        if self.enable_reproducibility:
            self._enable_reproducibility()

        self._train_env = self._build_train_env()
        self._eval_env = None
        if self.eval_freq is not None:
            if self.eval_freq < 1 or not float(self.eval_freq).is_integer():
                raise ValueError(
                    f"`eval_freq` ({self.eval_freq}) "
                    f"must be in the integer interval [1, inf)"
                )
            self.eval_freq = int(self.eval_freq)
            self._eval_env = self._build_eval_env()

        self._policy_network = self._build_policy_network()

        self._train_agent = self._build_train_agent()
        self._eval_agent = None
        if self.eval_freq is not None:
            self._eval_agent = self._build_eval_agent()

        self._train_sampler = self._build_train_sampler()
        self._eval_sampler = None
        if self.eval_freq is not None:
            self._eval_sampler = self._build_eval_sampler()

        self._buffer = self._build_buffer()

        self._buffer_dataset = self._build_buffer_dataset()

        self._dataloader = self._build_dataloader()

        if (
            self.cumreward_window_size < 1
            or not float(self.cumreward_window_size).is_integer()
        ):
            raise ValueError(
                f"`cumreward_window_size` ({self.cumreward_window_size}) "
                f"must be in the integer interval [1, inf)"
            )
        self.cumreward_window_size = int(self.cumreward_window_size)
        self._cumrewards = deque(maxlen=self.cumreward_window_size)

        torch.autograd.set_detect_anomaly(self.enable_anomaly_detection)

        self._profiler = None
        if isinstance(self.enable_profiling, dict) or self.enable_profiling:
            self._profiler = self._build_profiler()

        if not isinstance(self.enable_amp, dict):
            self.enable_amp = (
                {
                    "enabled": True,
                    "dtype": torch.float16,
                    "cache_enabled": True,
                }
                if self.enable_amp
                else {"enabled": False}
            )

    # override
    def reset_config(self, new_config: "Dict[str, Any]") -> "bool":
        self.stop()
        for key in self.config:
            self.__dict__.pop(key, None)
        self.config = new_config
        self.setup(deepcopy(new_config))
        return True

    # override
    def step(self) -> "Dict[str, Any]":
        result = {}

        result["train"] = self._train_step()

        if self.eval_freq is not None:
            if self.iteration % self.eval_freq == 0:
                result["eval"] = self._eval_step()

        if self._cumrewards:
            result[f"cumreward_{self._cumrewards.maxlen}"] = np.mean(self._cumrewards)
        if "num_timesteps" in result["train"]:
            result[tune_result.TIMESTEPS_THIS_ITER] = result["train"]["num_timesteps"]
        if "num_episodes" in result["train"]:
            result[tune_result.EPISODES_THIS_ITER] = result["train"]["num_episodes"]
        return self._postprocess_result(result)

    # override
    def save_checkpoint(self, tmp_checkpoint_dir: "str") -> "str":
        checkpoint_file = os.path.join(tmp_checkpoint_dir, "checkpoint.pt")
        torch.save(self._checkpoint, checkpoint_file)
        return checkpoint_file

    # override
    def load_checkpoint(self, checkpoint: "str") -> "None":
        self._checkpoint = torch.load(checkpoint, map_location=self._device)

    # override
    def cleanup(self) -> "None":
        self._train_env.close()
        if self._eval_env is not None:
            self._eval_env.close()

    # override
    def _export_model(
        self,
        export_formats: "Sequence[str]",
        export_dir: "str",
    ) -> "Dict[str, str]":
        if any(v.strip().lower() not in self._EXPORT_FORMATS for v in export_formats):
            raise ValueError(
                f"`export_formats` ({export_formats}) must be a subset of {self._EXPORT_FORMATS}"
            )
        exported = {}
        if ExportFormat.CHECKPOINT in export_formats:
            checkpoint_file = os.path.join(export_dir, "checkpoint.pt")
            torch.save(self._policy_network.state_dict(), checkpoint_file)
            exported[ExportFormat.CHECKPOINT] = checkpoint_file
        if ExportFormat.MODEL in export_formats:
            model_file = os.path.join(export_dir, "model.pkl")
            torch.save(self._policy_network, model_file, pickle_module=ray.cloudpickle)
            exported[ExportFormat.MODEL] = model_file
        return exported

    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = {
            "rng_state": torch.random.get_rng_state(),
            "train_agent": self._train_agent.state_dict(
                exclude_keys=["policy_network"]
            ),
            "policy_network": self._policy_network.state_dict(),
            "buffer_dataset": self._buffer_dataset.state_dict(exclude_keys=["buffer"]),
            "cumrewards": np.asarray(self._cumrewards),
        }
        if self.train_num_timesteps_per_iter is not None:
            checkpoint[
                "train_num_timesteps_per_iter"
            ] = self.train_num_timesteps_per_iter.state_dict()
        if self.train_num_episodes_per_iter is not None:
            checkpoint[
                "train_num_episodes_per_iter"
            ] = self.train_num_episodes_per_iter.state_dict()
        if self.eval_freq is not None:
            if self.eval_num_timesteps_per_iter is not None:
                checkpoint[
                    "eval_num_timesteps_per_iter"
                ] = self.eval_num_timesteps_per_iter.state_dict()
            if self.eval_num_episodes_per_iter is not None:
                checkpoint[
                    "eval_num_episodes_per_iter"
                ] = self.eval_num_episodes_per_iter.state_dict()
            checkpoint["eval_agent"] = self._eval_agent.state_dict(
                exclude_keys=["policy_network"]
            )
        if self.buffer_checkpoint:
            checkpoint["buffer"] = self._buffer.state_dict()
        return checkpoint

    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        torch.random.set_rng_state(value["rng_state"])
        self._train_agent.load_state_dict(value["train_agent"], strict=False)
        self._policy_network.load_state_dict(value["policy_network"])
        self._buffer_dataset.load_state_dict(value["buffer_dataset"], strict=False)
        self._cumrewards = deque(value["cumrewards"])
        if "train_num_timesteps_per_iter" in value:
            self.train_num_timesteps_per_iter.load_state_dict(
                value["train_num_timesteps_per_iter"]
            )
        if "train_num_episodes_per_iter" in value:
            self.train_num_episodes_per_iter.load_state_dict(
                value["train_num_episodes_per_iter"]
            )
        if "eval_num_timesteps_per_iter" in value:
            self.eval_num_timesteps_per_iter.load_state_dict(
                value["eval_num_timesteps_per_iter"]
            )
        if "eval_num_episodes_per_iter" in value:
            self.eval_num_episodes_per_iter.load_state_dict(
                value["eval_num_episodes_per_iter"]
            )
        if "eval_agent" in value:
            self._eval_agent.load_state_dict(value["eval_agent"], strict=False)
        if "buffer" in value:
            self._buffer.load_state_dict(value["buffer"])

    def _seed(self) -> "None":
        if self.seed < 0 or not float(self.seed).is_integer():
            raise ValueError(
                f"`seed` ({self.seed}) must be in the integer interval [0, inf)"
            )
        self.seed = int(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _enable_reproducibility(self) -> "None":
        import torch  # Fix TypeError: cannot pickle 'CudnnModule' object

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        # See https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def _build_train_env(self) -> "BatchedEnv":
        if self.train_env_config is None:
            self.train_env_config = {}

        try:
            train_env = self.train_env_builder(
                **self.train_env_config,
            )
        except TypeError:
            train_env = self.train_env_builder(**self.train_env_config)
        if not isinstance(train_env, BatchedEnv):
            train_env.close()
            train_env = SerialBatchedEnv(self.train_env_builder, self.train_env_config)

        train_env.seed(self.seed)
        train_env = BatchedUnflattenAction(BatchedFlattenObservation(train_env))
        return train_env

    def _build_eval_env(self) -> "BatchedEnv":
        if self.eval_env_builder is None:
            self.eval_env_builder = self.train_env_builder
            if self.eval_env_config is None:
                self.eval_env_config = self.train_env_config
        if self.eval_env_config is None:
            self.eval_env_config = {}

        try:
            eval_env = self.eval_env_builder(**self.eval_env_config)
        except TypeError:
            eval_env = self.eval_env_builder(**self.eval_env_config)
        if not isinstance(eval_env, BatchedEnv):
            eval_env.close()
            eval_env = SerialBatchedEnv(self.eval_env_builder, self.eval_env_config)

        eval_env.seed(self.seed + len(self._train_env))
        eval_env = BatchedUnflattenAction(BatchedFlattenObservation(eval_env))
        return eval_env

    def _build_policy_network(self) -> "PolicyNetwork":  # noqa: C901
        if self.policy_network_preprocessors is None:
            self.policy_network_preprocessors = {}
        unnested_single_observation_space = {
            f"observation{k}": v
            for k, v in self._train_env.single_observation_space.unnested.items()
        }
        unexpected_keys = (
            self.policy_network_preprocessors.keys()
            - unnested_single_observation_space.keys()
        )
        if not unexpected_keys:
            # Order matters
            self.policy_network_preprocessors = {
                key: self.policy_network_preprocessors.get(
                    key,
                    self._get_default_policy_network_preprocessor(space),
                )
                for key, space in unnested_single_observation_space.items()
            }

        if self.policy_network_model_builder is None:
            self.policy_network_model_builder = FCNet
            if self.policy_network_model_config is None:
                self.policy_network_model_config = {
                    "torso_fc_configs": [
                        {"out_features": 64, "bias": True},
                        {"out_features": 32, "bias": True},
                    ],
                    "torso_activation_builder": torch.nn.ReLU,
                    "torso_activation_config": {"inplace": False},
                    "head_fc_bias": True,
                    "head_activation_builder": torch.nn.Identity,
                    "head_activation_config": {},
                    "independent_heads": [],
                }
        if self.policy_network_model_config is None:
            self.policy_network_model_config = {}

        if self.policy_network_distribution_builders is None:
            self.policy_network_distribution_builders = {}
        if self.policy_network_distribution_parametrizations is None:
            self.policy_network_distribution_parametrizations = {}
        if self.policy_network_distribution_configs is None:
            self.policy_network_distribution_configs = {}
        unnested_single_action_space = {
            f"action{k}": v
            for k, v in self._train_env.single_action_space.unnested.items()
        }
        unexpected_keys = (
            self.policy_network_distribution_builders.keys()
            - unnested_single_action_space.keys()
        )
        if not unexpected_keys:
            # Order matters
            policy_network_distribution_builders = {}
            policy_network_distribution_parametrizations = {}
            policy_network_distribution_configs = {}
            for key, space in unnested_single_action_space.items():
                if key in self.policy_network_distribution_builders:
                    policy_network_distribution_builders[
                        key
                    ] = self.policy_network_distribution_builders[key]
                    if key not in self.policy_network_distribution_parametrizations:
                        raise ValueError(
                            f'`policy_network_distribution_parametrizations["{key}"]` must be '
                            f'set when `policy_network_distribution_builders["{key}"]` is set'
                        )
                    policy_network_distribution_parametrizations[
                        key
                    ] = self.policy_network_distribution_parametrizations[key]
                    policy_network_distribution_configs[
                        key
                    ] = self.policy_network_distribution_configs.get(key, {})
                else:
                    policy_network_distribution_builders[
                        key
                    ] = self._get_default_policy_network_distribution_builder(space)
                    policy_network_distribution_parametrizations[
                        key
                    ] = self.policy_network_distribution_parametrizations.get(
                        key,
                        self._get_default_policy_network_distribution_parametrization(
                            space
                        ),
                    )
                    policy_network_distribution_configs[
                        key
                    ] = self.policy_network_distribution_configs.get(
                        key,
                        self._get_default_policy_network_distribution_config(space),
                    )
            self.policy_network_distribution_builders = (
                policy_network_distribution_builders
            )
            for (
                key,
                policy_network_distribution_parametrization,
            ) in self.policy_network_distribution_parametrizations.items():
                if key not in policy_network_distribution_parametrizations:
                    policy_network_distribution_parametrizations[
                        key
                    ] = policy_network_distribution_parametrization
            self.policy_network_distribution_parametrizations = (
                policy_network_distribution_parametrizations
            )
            for (
                key,
                policy_network_distribution_config,
            ) in self.policy_network_distribution_configs.items():
                if key not in policy_network_distribution_configs:
                    policy_network_distribution_configs[
                        key
                    ] = policy_network_distribution_config
            self.policy_network_distribution_configs = (
                policy_network_distribution_configs
            )

        if self.policy_network_normalizing_flows is None:
            self.policy_network_normalizing_flows = {}

        if self.policy_network_sample_fn is None:
            self.policy_network_sample_fn = lambda distribution: distribution.mode

        if self.policy_network_prediction_fn is None:
            self.policy_network_prediction_fn = lambda sample: sample

        policy_network = PolicyNetwork(
            self.policy_network_preprocessors,
            self.policy_network_model_builder,
            self.policy_network_distribution_builders,
            self.policy_network_distribution_parametrizations,
            self.policy_network_model_config,
            self.policy_network_distribution_configs,
            self.policy_network_normalizing_flows,
            self.policy_network_sample_fn,
            self.policy_network_prediction_fn,
            self.policy_network_postprocessors,
        )
        self._log_graph(policy_network.wrapped_model.model, "policy_network_model")
        return policy_network

    def _build_train_agent(self) -> "Agent":
        if self.train_agent_builder is None:
            self.train_agent_builder = StochasticAgent
            if self.train_agent_config is None:
                self.train_agent_config = {
                    "clip_action": True,
                    "device": "cpu",
                    "num_random_timesteps": 0,
                }
        if self.train_agent_config is None:
            self.train_agent_config = {}
        return self.train_agent_builder(
            self._policy_network,
            self._train_env.observation_space,
            self._train_env.action_space,
            **self.train_agent_config,
        )

    def _build_eval_agent(self) -> "Agent":
        if self.eval_agent_builder is None:
            self.eval_agent_builder = Agent
            if self.eval_agent_builder is None:
                self.eval_agent_builder = {
                    "clip_action": True,
                    "device": "cpu",
                }
        if self.eval_agent_config is None:
            self.eval_agent_config = {}
        return self.eval_agent_builder(
            self._policy_network,
            self._eval_env.observation_space,
            self._eval_env.action_space,
            **self.eval_agent_config,
        )

    def _build_train_sampler(self) -> "Sampler":
        if self.train_sampler_builder is None:
            self.train_sampler_builder = Sampler
            if self.train_sampler_config is None:
                self.train_sampler_config = {"callbacks": []}
        if self.train_sampler_config is None:
            self.train_sampler_config = {}
        if (
            self.train_num_timesteps_per_iter is None
            and self.train_num_episodes_per_iter is None
        ):
            self.train_num_episodes_per_iter = 1
        if self.train_num_timesteps_per_iter is not None:
            if not isinstance(self.train_num_timesteps_per_iter, Schedule):
                self.train_num_timesteps_per_iter = ConstantSchedule(
                    self.train_num_timesteps_per_iter
                )
        if self.train_num_episodes_per_iter is not None:
            if not isinstance(self.train_num_episodes_per_iter, Schedule):
                self.train_num_episodes_per_iter = ConstantSchedule(
                    self.train_num_episodes_per_iter
                )
        return self.train_sampler_builder(
            self._train_env,
            self._train_agent,
            **self.train_sampler_config,
        )

    def _build_eval_sampler(self) -> "Sampler":
        if self.eval_sampler_builder is None:
            self.eval_sampler_builder = Sampler
            if self.eval_sampler_config is None:
                self.eval_sampler_config = {"callbacks": []}
        if self.eval_sampler_config is None:
            self.eval_sampler_config = {}
        if (
            self.eval_num_timesteps_per_iter is None
            and self.eval_num_episodes_per_iter is None
        ):
            self.eval_num_episodes_per_iter = 1
        if self.eval_num_timesteps_per_iter is not None:
            if not isinstance(self.eval_num_timesteps_per_iter, Schedule):
                self.eval_num_timesteps_per_iter = ConstantSchedule(
                    self.eval_num_timesteps_per_iter
                )
        if self.eval_num_episodes_per_iter is not None:
            if not isinstance(self.eval_num_episodes_per_iter, Schedule):
                self.eval_num_episodes_per_iter = ConstantSchedule(
                    self.eval_num_episodes_per_iter
                )
        return self.eval_sampler_builder(
            self._eval_env,
            self._eval_agent,
            **self.eval_sampler_config,
        )

    def _build_buffer(self) -> "Buffer":
        if self.buffer_builder is None:
            self.buffer_builder = UniformBuffer
        if self.buffer_config is None:
            self.buffer_config = {"capacity": max(1, len(self._train_env))}
        return self.buffer_builder(
            spec=self._get_default_buffer_spec(),
            **self.buffer_config,
        )

    def _build_buffer_dataset(self) -> "BufferDataset":
        if self.buffer_dataset_builder is None:
            self.buffer_dataset_builder = BufferDataset
            if self.buffer_dataset_config is None:
                self.buffer_dataset_config = {
                    "batch_size": 1,
                    "max_trajectory_length": 1,
                    "num_iters": 1,
                }
        if self.buffer_dataset_config is None:
            self.buffer_dataset_config = {}
        return self.buffer_dataset_builder(
            buffer=self._buffer,
            **self.buffer_dataset_config,
        )

    def _build_dataloader(self) -> "DataLoader":
        if self.dataloader_builder is None:
            self.dataloader_builder = DataLoader
            if self.dataloader_config is None:
                fork = torch.multiprocessing.get_start_method() == "fork"
                self.dataloader_config = {
                    "num_workers": 1 if fork else 0,
                    "pin_memory": True,
                    "timeout": 0,
                    "worker_init_fn": None,
                    "generator": None,
                    "prefetch_factor": 1 if fork else 2,
                    "pin_memory_device": "",
                }
        if self.dataloader_config is None:
            self.dataloader_config = {}
        return self.dataloader_builder(
            dataset=self._buffer_dataset,
            batch_size=None,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            collate_fn=None,
            drop_last=False,
            persistent_workers=False,
            **self.dataloader_config,
        )

    def _build_profiler(self) -> "profile":
        if not isinstance(self.enable_profiling, dict) and self.enable_profiling:
            self.enable_profiling = {
                "activities": None,
                "schedule": None,
                "on_trace_ready": tensorboard_trace_handler(
                    os.path.join(self.logdir, "profiler")
                ),
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": False,
                "with_flops": False,
                "with_modules": False,
            }
        return profile(**self.enable_profiling)

    def _train_step(self) -> "Dict[str, Any]":
        train_num_timesteps_per_iter = train_num_episodes_per_iter = None
        if self.train_num_timesteps_per_iter:
            train_num_timesteps_per_iter = self.train_num_timesteps_per_iter()
            self.train_num_timesteps_per_iter.step()
        if self.train_num_episodes_per_iter:
            train_num_episodes_per_iter = self.train_num_episodes_per_iter()
            self.train_num_episodes_per_iter.step()
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            for experience, done in self._train_sampler.sample(
                train_num_timesteps_per_iter,
                train_num_episodes_per_iter,
            ):
                self._buffer.add(experience, done)
        result = self._train_sampler.stats
        self._cumrewards += result["episode_cumreward"]

        if not self._UPDATE_BUFFER_DATASET_SCHEDULES_AFTER_TRAIN_EPOCH:
            for schedule in self._buffer_dataset.schedules.values():
                schedule.step()
        train_epoch_result = self._train_epoch()
        result.update(train_epoch_result)

        for schedule in self._train_agent.schedules.values():
            schedule.step()
        for schedule in self._buffer.schedules.values():
            schedule.step()
        if self._UPDATE_BUFFER_DATASET_SCHEDULES_AFTER_TRAIN_EPOCH:
            for schedule in self._buffer_dataset.schedules.values():
                schedule.step()
        return result

    def _train_epoch(self) -> "Dict[str, Any]":
        if not self._policy_network.training:
            self._policy_network.train()
        self._policy_network.to(self._device, non_blocking=True)
        with (self._profiler if self._profiler else contextlib.suppress()):
            train_on_batch_result = {}
            for idx, (experiences, is_weight, mask) in enumerate(self._dataloader):
                observation = experiences.pop("next_observation")
                observation = F.pad(
                    observation, (0, 0) * (observation.ndim - 2) + (1, 0)
                )
                observation[:, 0, ...] = experiences["observation"][:, 0, ...]
                experiences["observation"] = observation
                mask = F.pad(mask, [1, 0], value=True)
                for k, v in experiences.items():
                    experiences[k] = v.to(
                        self._device, dtype=torch.float32, non_blocking=True
                    )
                is_weight = is_weight.to(
                    self._device, dtype=torch.float32, non_blocking=True
                )
                mask = mask.to(self._device, non_blocking=True)
                with (
                    record_function("_train_on_batch")
                    if self._profiler
                    else contextlib.suppress()
                ):
                    train_on_batch_result, priority = self._train_on_batch(
                        idx,
                        experiences,
                        is_weight,
                        mask,
                    )
                if self._profiler:
                    self._profiler.step()
                if priority is not None:
                    self._buffer.update_priority(priority)
        return train_on_batch_result

    def _eval_step(self) -> "Dict[str, Any]":
        eval_num_timesteps_per_iter = eval_num_episodes_per_iter = None
        if self.eval_num_timesteps_per_iter:
            eval_num_timesteps_per_iter = self.eval_num_timesteps_per_iter()
            self.eval_num_timesteps_per_iter.step()
        if self.eval_num_episodes_per_iter:
            eval_num_episodes_per_iter = self.eval_num_episodes_per_iter()
            self.eval_num_episodes_per_iter.step()
        self._eval_sampler.reset()
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            for _ in self._eval_sampler.sample(
                eval_num_timesteps_per_iter,
                eval_num_episodes_per_iter,
            ):
                pass
        for schedule in self._eval_agent.schedules.values():
            schedule.step()
        return self._eval_sampler.stats

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_policy_network_preprocessor(
        self, space: "spaces.Space"
    ) -> "Processor":
        raise NotImplementedError(
            f"Unsupported space type: "
            f"`{type(space).__module__}.{type(space).__name__}`. "
            f"Register a custom space type through decorator "
            f"`{type(self).__module__}.{type(self).__name__}."
            f"_get_default_policy_network_preprocessor.register`"
        )

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_policy_network_distribution_builder(
        self,
        space: "spaces.Space",
    ) -> "Callable[..., Distribution]":
        raise NotImplementedError(
            f"Unsupported space type: "
            f"`{type(space).__module__}.{type(space).__name__}`. "
            f"Register a custom space type through decorator "
            f"`{type(self).__module__}.{type(self).__name__}."
            f"_get_default_policy_network_distribution_builder.register`"
        )

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_policy_network_distribution_parametrization(
        self,
        space: "spaces.Space",
    ) -> "Callable[..., DistributionParametrization]":
        raise NotImplementedError(
            f"Unsupported space type: "
            f"`{type(space).__module__}.{type(space).__name__}`. "
            f"Register a custom space type through decorator "
            f"`{type(self).__module__}.{type(self).__name__}."
            f"_get_default_policy_network_distribution_parametrization.register`"
        )

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_policy_network_distribution_config(
        self,
        space: "spaces.Space",
    ) -> "Dict[str, Any]":
        raise NotImplementedError(
            f"Unsupported space type: "
            f"`{type(space).__module__}.{type(space).__name__}`. "
            f"Register a custom space type through decorator "
            f"`{type(self).__module__}.{type(self).__name__}."
            f"_get_default_policy_network_distribution_config.register`"
        )

    def _get_default_buffer_spec(self) -> "Dict[str, Dict[str, Any]]":
        single_observation_space = self._train_env.single_observation_space
        single_action_space = self._train_env.single_action_space
        return {
            "observation": {
                "shape": single_observation_space.shape,
                "dtype": single_observation_space.dtype,
            },
            "next_observation": {
                "shape": single_observation_space.shape,
                "dtype": single_observation_space.dtype,
            },
            "action": {
                "shape": single_action_space.shape,
                "dtype": single_action_space.dtype,
            },
            "log_prob": {"shape": (), "dtype": np.float32},
            "reward": {"shape": (), "dtype": np.float32},
            "terminal": {"shape": (), "dtype": bool},
        }

    def _log_graph(self, model: "Model", name: "str" = "model") -> "None":
        trainable_count, non_trainable_count = count_params(model)
        separator = "---------------------------------------"
        header = f" {name} ".upper().center(len(separator), "-")
        with open(os.path.join(self.logdir, f"{name}.txt"), "w") as f:
            f.write(
                f"{header}\n{model}\n{separator}\n"
                f"Trainable params: {trainable_count}\n"
                f"Non-trainable params: {non_trainable_count}\n{separator}"
            )
        try:
            summary_writer = SummaryWriter(self.logdir, filename_suffix=".graph")
            input_to_model = list(model.get_example_inputs((2,)))
            if not input_to_model[1]:
                # If states is empty, add a dummy value to avoid
                # RuntimeError: Dictionary inputs must have entries
                input_to_model[1]["_"] = torch.empty(1)
            summary_writer.add_graph(
                model,
                input_to_model,
                use_strict_trace=False,
            )
            summary_writer.close()
        except Exception as e:
            _LOGGER.warning(f"Could not export graph of `{name}`: {e}")

    def _postprocess_result(self, result: "Dict[str, Any]") -> "Dict[str, Any]":
        postprocessed = {}
        for k, v in result.items():
            if isinstance(v, dict):
                postprocessed[k] = self._postprocess_result(v)
                continue
            try:
                if hasattr(v, "__len__"):
                    postprocessed[f"{k}/mean"] = float(np.mean(v))
                    postprocessed[f"{k}/stddev"] = float(np.std(v))
                else:
                    raise Exception
            except Exception:
                postprocessed.pop(f"{k}/mean", None)
                postprocessed.pop(f"{k}/stddev", None)
                postprocessed[k] = v
        return postprocessed

    def _resolve_future_refs_if_any(self, item: "Any") -> "Any":
        if isinstance(item, (tuple, list)):
            return type(item)(self._resolve_future_refs_if_any(v) for v in item)
        if isinstance(item, dict):
            return {k: self._resolve_future_refs_if_any(v) for k, v in item.items()}
        if isinstance(item, FutureRef):
            return item.resolve(self=self, cls=type(self))
        return deepcopy(item)

    def __getattr__(self, name: "str") -> "Optional[Any]":
        try:
            attr = self.config[name]
        except KeyError:
            attr = None
        self.__dict__[name] = attr
        if attr is not None:
            self.__dict__[name] = attr = self._resolve_future_refs_if_any(attr)
        return attr

    @abstractmethod
    def _train_on_batch(
        self,
        idx: "int",
        experiences: "Dict[str, Tensor]",
        is_weight: "Tensor",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Any], Optional[ndarray]]":
        """Train on a data batch.

        In the following, let `B` denote the batch size
        and `T` the maximum trajectory length.

        Parameters
        ----------
        idx:
            The batch index.
        experiences:
            The batched experiences, shape of ``experiences[name]``:
            ``[B, T + 1, *self._buffer.spec[name]["shape"]]``
            if `name` is equal to ``"observation"``,
            ``[B, T, *self._buffer.spec[name]["shape"]]`` otherwise.
        is_weight:
            The buffer batched importance sampling weight,
            shape: ``[B]``.
        mask:
            The boolean tensor indicating which trajectory
            elements are valid (True) and which are not (False),
            shape: ``[B, T + 1]``.

        Returns
        -------
            - The training result;
            - the updated buffer batched priority.

        """
        raise NotImplementedError


class DistributedDataParallelAlgorithm(SyncDistributedTrainable):
    """Distributed data parallel algorithm.

    See Also
    --------
    actorch.algorithms.algorithm.Algorithm

    """

    _ALGORITHM_CLS = Algorithm

    _REDUCTION_SUM_KEYS = [
        "num_timesteps",
        "num_episodes",
        tune_result.TIMESTEPS_THIS_ITER,
        tune_result.EPISODES_THIS_ITER,
    ]

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
            backend: "Tunable[Optional[str]]" = None,
            timeout_s: "Tunable[float]" = 1800.0,
            node_syncer_builder: "Tunable[Optional[Callable[..., NodeSyncer]]]" = None,
            node_syncer_config: "Tunable[Optional[Dict[str, Any]]]" = None,
            worker_init_fn: "Tunable[Optional[Callable[[], None]]]" = None,
            log_sys_usage: "Tunable[bool]" = False,
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
            log_sys_usage:
                True to log the system usage statistics, False otherwise.
            worker_config:
                The worker configuration.
                Default to ``{}``.

            Warnings
            --------
            Since the configuration is the same for each worker, the effective
            value of a parameter that is cumulative by nature is equal to the
            same parameter value multiplied by the number of workers (e.g.
            ``train_num_episodes_per_iter = num_workers * worker_config["train_num_episodes_per_iter"]``).

            """
            super().__init__(
                num_workers=num_workers,
                num_cpus_per_worker=num_cpus_per_worker,
                num_gpus_per_worker=num_gpus_per_worker,
                extra_resources_per_worker=extra_resources_per_worker,
                placement_strategy=placement_strategy,
                backend=backend,
                timeout_s=timeout_s,
                node_syncer_builder=node_syncer_builder,
                node_syncer_config=node_syncer_config,
                worker_init_fn=worker_init_fn,
                log_sys_usage=log_sys_usage,
                **worker_config,
            )

    # override
    @classmethod
    def get_worker_cls(cls) -> "Type[Trainable]":
        class Worker(cls._ALGORITHM_CLS):
            # override
            def setup(self, config: "Dict[str, Any]") -> "None":
                init_mock_train_session()
                super().setup(config)
                # Handle Ray local mode (i.e. single process) correctly
                try:
                    accelerate(amp=self.enable_amp["enabled"])
                except RuntimeError as e:
                    _LOGGER.warning(f"Could not enable training optimizations: {e}")
                self._dataloader = prepare_data_loader(
                    self._dataloader,
                    add_dist_sampler=False,
                )
                self._policy_network_state_dict_keys = list(
                    self._policy_network.state_dict().keys()
                )
                self._policy_network_wrapped_model = self._policy_network.wrapped_model
                self._policy_network_normalizing_flows = (
                    self._policy_network.normalizing_flows
                )
                prepare_model_kwargs = {}
                if not any(
                    x.requires_grad
                    for x in self._policy_network_wrapped_model.parameters()
                ):
                    prepare_model_kwargs = {"ddp_cls": None}
                self._policy_network.wrapped_model = prepare_model(
                    self._policy_network_wrapped_model,
                    **prepare_model_kwargs,
                )
                for k, v in self._policy_network_normalizing_flows.items():
                    model = v.model
                    prepare_model_kwargs = {}
                    if not any(x.requires_grad for x in model.parameters()):
                        prepare_model_kwargs = {"ddp_cls": None}
                    self._policy_network.normalizing_flows[k].model = prepare_model(
                        model,
                        **prepare_model_kwargs,
                    )
                self._prepared_policy_network_state_dict_keys = list(
                    self._policy_network.state_dict().keys()
                )

            # override
            @property
            def _checkpoint(self) -> "Dict[str, Any]":
                checkpoint = super()._checkpoint
                checkpoint["policy_network"] = {
                    self._policy_network_state_dict_keys[i]: v
                    for i, v in enumerate(checkpoint["policy_network"].values())
                }
                return checkpoint

            # override
            @_checkpoint.setter
            def _checkpoint(self, value: "Dict[str, Any]") -> "None":
                value["policy_network"] = {
                    self._prepared_policy_network_state_dict_keys[i]: v
                    for i, v in enumerate(value["policy_network"].values())
                }
                super()._checkpoint = value

            # override
            def _export_model(
                self,
                export_formats: "Sequence[str]",
                export_dir: "str",
            ) -> "Dict[str, str]":
                self._policy_network.wrapped_model = self._policy_network_wrapped_model
                self._policy_network.normalizing_flows = (
                    self._policy_network_normalizing_flows
                )
                return super()._export_model(export_formats, export_dir)

            # override
            def _seed(self) -> "None":
                self.seed += world_rank()
                super()._seed()

        return Worker

    @classmethod
    def rename(cls, name: "str") -> "Type[DistributedDataParallelAlgorithm]":
        """Return a copy of this class with
        name set to `name`.

        Parameters
        ----------
        name:
            The new name.

        Returns
        -------
            The renamed class.

        """
        return type(name, (cls,), {})

    # override
    def setup(self, config: "Dict[str, Any]") -> "None":
        config = DistributedDataParallelAlgorithm.Config(**config)
        self.log_sys_usage = config.pop("log_sys_usage")
        config["reduction_mode"] = "sum"
        super().setup(config)
        self.config["log_sys_usage"] = self.log_sys_usage

    # override
    def reset_config(self, new_config: "Dict[str, Any]") -> "bool":
        new_config = DistributedDataParallelAlgorithm.Config(**new_config)
        if self.log_sys_usage != new_config.pop("log_sys_usage"):
            return False
        new_config["reduction_mode"] = "sum"
        return super().reset_config(new_config)

    # override
    def _reduce(self, results: "Sequence[Dict[str, Any]]") -> "Dict[str, Any]":
        reduced = results[0]
        for k, v in reduced.items():
            if isinstance(v, dict):
                reduced[k] = self._reduce(
                    [result[k] for result in results if k in result]
                )
                continue
            if k.endswith("/mean"):
                reduced[k] = np.asarray(
                    [result[k] for result in results if k in result]
                ).mean(axis=0)
                continue
            if k.endswith("/stddev"):
                mean_key = k.replace("/stddev", "/mean")
                if mean_key in reduced:
                    means, stddevs = [], []
                    for result in results:
                        means.append(result[mean_key])
                        stddevs.append(result[k])
                    means = np.asarray(means)
                    stddevs = np.asarray(stddevs)
                else:
                    stddevs = np.asarray(
                        [result[k] for result in results if k in result]
                    )
                    means = np.zeros_like(stddevs)
                reduced[k] = np.sqrt(
                    (means**2 + stddevs**2).mean(axis=0) - means.mean(axis=0) ** 2
                )
                continue
            try:
                value = np.asarray([result[k] for result in results if k in result])
                reduced[k] = (
                    value.all()
                    if value.dtype == bool
                    else (
                        value.sum(axis=0)
                        if k in self._REDUCTION_SUM_KEYS
                        else value.mean(axis=0)
                    )
                )
            except Exception:
                pass
        return reduced


#####################################################################################################
# Algorithm._get_default_policy_network_preprocessor implementation
#####################################################################################################


@Algorithm._get_default_policy_network_preprocessor.register(spaces.Box)
@Algorithm._get_default_policy_network_preprocessor.register(spaces.MultiBinary)
def _get_default_policy_network_preprocessor_box_multi_binary(
    self,
    space: "Union[spaces.Box, spaces.MultiBinary]",
) -> "Identity":
    return Identity(space.shape)


@Algorithm._get_default_policy_network_preprocessor.register(spaces.Discrete)
def _get_default_policy_network_preprocessor_discrete(
    self,
    space: "spaces.Discrete",
) -> "OneHotEncode":
    return OneHotEncode(space.n)


@Algorithm._get_default_policy_network_preprocessor.register(spaces.MultiDiscrete)
def _get_default_policy_network_preprocessor_multi_discrete(
    self,
    space: "spaces.MultiDiscrete",
) -> "Independent":
    return Independent(OneHotEncode(space.nvec.max()), space.shape)


#####################################################################################################
# Algorithm._get_default_policy_network_distribution_builder implementation
#####################################################################################################


@Algorithm._get_default_policy_network_distribution_builder.register(spaces.Box)
def _get_default_policy_network_distribution_builder_box(
    self,
    space: "spaces.Box",
) -> "Callable[..., Normal]":
    return Normal


@Algorithm._get_default_policy_network_distribution_builder.register(spaces.Discrete)
@Algorithm._get_default_policy_network_distribution_builder.register(
    spaces.MultiDiscrete
)
def _get_default_policy_network_distribution_builder_discrete_multi_discrete(
    self,
    space: "Union[spaces.Discrete, spaces.MultiDiscrete]",
) -> "Callable[..., Categorical]":
    return Categorical


@Algorithm._get_default_policy_network_distribution_builder.register(spaces.MultiBinary)
def _get_default_policy_network_distribution_builder_multi_binary(
    self,
    space: "spaces.MultiBinary",
) -> "Callable[..., Bernoulli]":
    return Bernoulli


#####################################################################################################
# Algorithm._get_default_policy_network_distribution_parametrization implementation
#####################################################################################################


@Algorithm._get_default_policy_network_distribution_parametrization.register(spaces.Box)
def _get_default_policy_network_distribution_parametrization_box(
    self,
    space: "spaces.Box",
) -> "DistributionParametrization":
    return {
        "loc": (
            {"loc": space.shape},
            lambda x: x["loc"],
        ),
        "scale": (
            {"log_scale": space.shape},
            lambda x: x["log_scale"].exp(),
        ),
    }


@Algorithm._get_default_policy_network_distribution_parametrization.register(
    spaces.Discrete
)
def _get_default_policy_network_distribution_parametrization_discrete(
    self,
    space: "spaces.Discrete",
) -> "DistributionParametrization":
    return {
        "logits": (
            {"logits": (space.n,)},
            lambda x: x["logits"],
        ),
    }


@Algorithm._get_default_policy_network_distribution_parametrization.register(
    spaces.MultiBinary
)
def _get_default_policy_network_distribution_parametrization_multi_binary(
    self,
    space: "spaces.MultiBinary",
) -> "DistributionParametrization":
    return {
        "logits": (
            {"logits": space.shape},
            lambda x: x["logits"],
        ),
    }


@Algorithm._get_default_policy_network_distribution_parametrization.register(
    spaces.MultiDiscrete
)
def _get_default_policy_network_distribution_parametrization_multi_discrete(
    self,
    space: "spaces.MultiDiscrete",
) -> "DistributionParametrization":
    nvec = torch.as_tensor(space.nvec)
    mask = torch.arange(nvec.max().int()).expand(*nvec.shape, -1) < nvec[..., None]
    return {
        "logits": (
            {"logits": mask.shape},
            lambda x: x["logits"].masked_fill(~mask, -float("inf")),
        ),
    }


#####################################################################################################
# Algorithm._get_default_policy_network_distribution_config implementation
#####################################################################################################


@Algorithm._get_default_policy_network_distribution_config.register(spaces.Box)
def _get_default_policy_network_distribution_config_box(
    self,
    space: "spaces.Box",
) -> "Dict[str, Any]":
    return {"validate_args": False}


@Algorithm._get_default_policy_network_distribution_config.register(spaces.Discrete)
@Algorithm._get_default_policy_network_distribution_config.register(spaces.MultiBinary)
@Algorithm._get_default_policy_network_distribution_config.register(
    spaces.MultiDiscrete
)
def _get_default_policy_network_distribution_config_discrete_multi_binary_multi_discrete(
    self,
    space: "Union[spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete]",
) -> "Dict[str, Any]":
    return {"probs": None, "validate_args": False}
