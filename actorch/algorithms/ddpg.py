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

"""Deep Deterministic Policy Gradient."""

import contextlib
import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from gymnasium import Env, spaces
from numpy import ndarray
from ray.tune import Trainable
from torch import Tensor
from torch.cuda.amp import autocast
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from actorch.agents import Agent, GaussianNoiseAgent
from actorch.algorithms.a2c import A2C, DistributedDataParallelA2C, Loss, LRScheduler
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.algorithms.utils import freeze_params, prepare_model, sync_polyak
from actorch.algorithms.value_estimation import n_step_return
from actorch.buffers import Buffer
from actorch.datasets import BufferDataset
from actorch.distributions import Deterministic
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import DistributionParametrization, Network, Processor
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, Schedule
from actorch.utils import singledispatchmethod


__all__ = [
    "DDPG",
    "DistributedDataParallelDDPG",
]


class DDPG(A2C):
    """Deep Deterministic Policy Gradient.

    References
    ----------
    .. [1] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez,
           Y. Tassa, D. Silver, and D. Wierstra.
           "Continuous control with deep reinforcement learning".
           In: ICLR. 2016.
           URL: https://arxiv.org/abs/1509.02971

    """

    _UPDATE_BUFFER_DATASET_SCHEDULES_AFTER_TRAIN_EPOCH = True  # override

    _RESET_BUFFER = False  # override

    # override
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
            policy_network_postprocessors: "Tunable[RefOrFutureRef[Optional[Dict[str, Processor]]]]" = None,
            policy_network_optimizer_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Optimizer]]]]" = None,
            policy_network_optimizer_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            policy_network_optimizer_lr_scheduler_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., LRScheduler]]]]" = None,
            policy_network_optimizer_lr_scheduler_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            value_network_preprocessors: "Tunable[RefOrFutureRef[Optional[Dict[str, Processor]]]]" = None,
            value_network_model_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Model]]]]" = None,
            value_network_model_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            value_network_loss_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Loss]]]]" = None,
            value_network_loss_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            value_network_optimizer_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Optimizer]]]]" = None,
            value_network_optimizer_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            value_network_optimizer_lr_scheduler_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., LRScheduler]]]]" = None,
            value_network_optimizer_lr_scheduler_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            buffer_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Buffer]]]]" = None,
            buffer_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            buffer_checkpoint: "Tunable[RefOrFutureRef[bool]]" = False,
            dataloader_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., DataLoader]]]]" = None,
            dataloader_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            discount: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.99,
            num_return_steps: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 1,
            num_updates_per_iter: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 1000,
            batch_size: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 128,
            max_trajectory_length: "Tunable[RefOrFutureRef[Union[int, float, Schedule]]]" = float(  # noqa: B008
                "inf"
            ),
            sync_freq: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 1,
            polyak_weight: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.001,
            max_grad_l2_norm: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = float(  # noqa: B008
                "inf"
            ),
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
                policy_network_postprocessors=policy_network_postprocessors,
                policy_network_optimizer_builder=policy_network_optimizer_builder,
                policy_network_optimizer_config=policy_network_optimizer_config,
                policy_network_optimizer_lr_scheduler_builder=policy_network_optimizer_lr_scheduler_builder,
                policy_network_optimizer_lr_scheduler_config=policy_network_optimizer_lr_scheduler_config,
                value_network_preprocessors=value_network_preprocessors,
                value_network_model_builder=value_network_model_builder,
                value_network_model_config=value_network_model_config,
                value_network_loss_builder=value_network_loss_builder,
                value_network_loss_config=value_network_loss_config,
                value_network_optimizer_builder=value_network_optimizer_builder,
                value_network_optimizer_config=value_network_optimizer_config,
                value_network_optimizer_lr_scheduler_builder=value_network_optimizer_lr_scheduler_builder,
                value_network_optimizer_lr_scheduler_config=value_network_optimizer_lr_scheduler_config,
                buffer_builder=buffer_builder,
                buffer_config=buffer_config,
                buffer_checkpoint=buffer_checkpoint,
                dataloader_builder=dataloader_builder,
                dataloader_config=dataloader_config,
                discount=discount,
                num_return_steps=num_return_steps,
                num_updates_per_iter=num_updates_per_iter,
                batch_size=batch_size,
                max_trajectory_length=max_trajectory_length,
                sync_freq=sync_freq,
                polyak_weight=polyak_weight,
                max_grad_l2_norm=max_grad_l2_norm,
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

    # override
    def setup(self, config: "Dict[str, Any]") -> "None":
        self.config = DDPG.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        self._target_policy_network = copy.deepcopy(self._policy_network)
        self._target_policy_network.eval().to(self._device, non_blocking=True)
        self._target_policy_network.requires_grad_(False)
        self._target_value_network = copy.deepcopy(self._value_network)
        self._target_value_network.eval().to(self._device, non_blocking=True)
        self._target_value_network.requires_grad_(False)
        if not isinstance(self.num_updates_per_iter, Schedule):
            self.num_updates_per_iter = ConstantSchedule(self.num_updates_per_iter)
        if not isinstance(self.batch_size, Schedule):
            self.batch_size = ConstantSchedule(self.batch_size)
        if not isinstance(self.max_trajectory_length, Schedule):
            self.max_trajectory_length = ConstantSchedule(self.max_trajectory_length)
        if not isinstance(self.sync_freq, Schedule):
            self.sync_freq = ConstantSchedule(self.sync_freq)
        if not isinstance(self.polyak_weight, Schedule):
            self.polyak_weight = ConstantSchedule(self.polyak_weight)

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        checkpoint["target_policy_network"] = self._target_policy_network.state_dict()
        checkpoint["target_value_network"] = self._target_value_network.state_dict()
        checkpoint["sync_freq"] = self.sync_freq.state_dict()
        checkpoint["polyak_weight"] = self.polyak_weight.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        self._target_policy_network.load_state_dict(value["target_policy_network"])
        self._target_value_network.load_state_dict(value["target_value_network"])
        self.sync_freq.load_state_dict(value["sync_freq"])
        self.polyak_weight.load_state_dict(value["polyak_weight"])

    # override
    def _build_train_agent(self) -> "Agent":
        if self.train_agent_builder is None:
            self.train_agent_builder = GaussianNoiseAgent
            if self.train_agent_config is None:
                self.train_agent_config = {
                    "clip_action": True,
                    "device": "cpu",
                    "num_random_timesteps": 0,
                    "mean": 0.0,
                    "stddev": 0.1,
                }
        return super()._build_train_agent()

    # override
    def _build_buffer(self) -> "Buffer":
        if self.buffer_config is None:
            self.buffer_config = {"capacity": int(1e5)}
        return super()._build_buffer()

    # override
    def _build_buffer_dataset(self) -> "BufferDataset":
        if self.buffer_dataset_config is None:
            self.buffer_dataset_config = {
                "batch_size": self.batch_size,
                "max_trajectory_length": self.max_trajectory_length,
                "num_iters": self.num_updates_per_iter,
            }
        return super()._build_buffer_dataset()

    # override
    def _build_value_network(self) -> "Network":
        if self.value_network_preprocessors is None:
            self.value_network_preprocessors: "Dict[str, Processor]" = {}
        unnested_single_observation_space = {
            f"observation{k}": v
            for k, v in self._train_env.single_observation_space.unnested.items()
        }
        unnested_single_action_space = {
            f"action{k}": v
            for k, v in self._train_env.single_action_space.unnested.items()
        }
        unexpected_keys = (
            self.value_network_preprocessors.keys()
            - unnested_single_observation_space.keys()
            - unnested_single_action_space.keys()
        )
        if not unexpected_keys:
            # Order matters
            self.value_network_preprocessors = {
                key: self.value_network_preprocessors.get(
                    key,
                    self._get_default_value_network_preprocessor(space),
                )
                for key, space in (
                    *unnested_single_observation_space.items(),
                    *unnested_single_action_space.items(),
                )
            }
        return super()._build_value_network()

    # override
    def _train_step(self) -> "Dict[str, Any]":
        result = super()._train_step()
        self.sync_freq.step()
        self.polyak_weight.step()
        result["num_updates_per_iter"] = self.num_updates_per_iter()
        result["batch_size"] = self.batch_size()
        result["max_trajectory_length"] = (
            self.max_trajectory_length()
            if self.max_trajectory_length() != float("inf")
            else "inf"
        )
        result["sync_freq"] = self.sync_freq()
        result["polyak_weight"] = self.polyak_weight()
        result["buffer_num_experiences"] = self._buffer.num_experiences
        result["buffer_num_full_trajectories"] = self._buffer.num_full_trajectories
        result.pop("entropy_coeff", None)
        return result

    # override
    def _train_on_batch(
        self,
        idx: "int",
        experiences: "Dict[str, Tensor]",
        is_weight: "Tensor",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Any], Optional[ndarray]]":
        sync_freq = self.sync_freq()
        if sync_freq < 1 or not float(sync_freq).is_integer():
            raise ValueError(
                f"`sync_freq` ({sync_freq}) "
                f"must be in the integer interval [1, inf)"
            )
        sync_freq = int(sync_freq)

        result = {}

        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            target_actions, _ = self._target_policy_network(
                experiences["observation"], mask=mask
            )
            observations_target_actions = torch.cat(
                [
                    x[..., None] if x.shape == mask.shape else x
                    for x in [experiences["observation"], target_actions]
                ],
                dim=-1,
            )
            target_action_values, _ = self._target_value_network(
                observations_target_actions, mask=mask
            )
            # Discard next observation
            experiences["observation"] = experiences["observation"][:, :-1, ...]
            mask = mask[:, 1:]
            targets, _ = n_step_return(
                target_action_values,
                experiences["reward"],
                experiences["terminal"],
                mask,
                self.discount(),
                self.num_return_steps(),
            )

            # Compute action values
            observations_actions = torch.cat(
                [
                    x[..., None] if x.shape == mask.shape else x
                    for x in [experiences["observation"], experiences["action"]]
                ],
                dim=-1,
            )
            action_values, _ = self._value_network(observations_actions, mask=mask)

        result["value_network"], priority = self._train_on_batch_value_network(
            action_values,
            targets,
            is_weight,
            mask,
        )
        result["policy_network"] = self._train_on_batch_policy_network(
            experiences,
            mask,
        )

        # Synchronize
        if idx % sync_freq == 0:
            sync_polyak(
                self._policy_network,
                self._target_policy_network,
                self.polyak_weight(),
            )
            sync_polyak(
                self._value_network,
                self._target_value_network,
                self.polyak_weight(),
            )

        self._grad_scaler.update()
        return result, priority

    # override
    def _train_on_batch_policy_network(
        self,
        experiences: "Dict[str, Tensor]",
        mask: "Tensor",
    ) -> "Dict[str, Any]":
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            actions, _ = self._policy_network(experiences["observation"], mask=mask)
            observations_actions = torch.cat(
                [
                    x[..., None] if x.shape == mask.shape else x
                    for x in [experiences["observation"], actions]
                ],
                dim=-1,
            )
            with freeze_params(self._value_network):
                action_values, _ = self._value_network(observations_actions, mask=mask)
            loss = -action_values[mask].mean()
        optimize_result = self._optimize_policy_network(loss)
        result = {"loss": loss.item()}
        result.update(optimize_result)
        return result

    # override
    def _train_on_batch_value_network(
        self,
        action_values: "Tensor",
        targets: "Tensor",
        is_weight: "Tensor",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Any], Optional[ndarray]]":
        result, priority = super()._train_on_batch_value_network(
            action_values,
            targets,
            is_weight,
            mask,
        )
        result = {
            k if k != "state_value" else "action_value": v for k, v in result.items()
        }
        return result, priority

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_policy_network_distribution_builder(
        self,
        space: "spaces.Space",
    ) -> "Callable[..., Distribution]":
        raise NotImplementedError(
            f"Unsupported space type: "
            f"`{type(space).__module__}.{type(space).__name__}`"
        )

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_policy_network_distribution_parametrization(
        self,
        space: "spaces.Space",
    ) -> "Callable[..., DistributionParametrization]":
        raise NotImplementedError(
            f"Unsupported space type: "
            f"`{type(space).__module__}.{type(space).__name__}`"
        )

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_policy_network_distribution_config(
        self,
        space: "spaces.Space",
    ) -> "Dict[str, Any]":
        raise NotImplementedError(
            f"Unsupported space type: "
            f"`{type(space).__module__}.{type(space).__name__}`"
        )


class DistributedDataParallelDDPG(DistributedDataParallelA2C):
    """Distributed data parallel Deep Deterministic Policy Gradient.

    See Also
    --------
    actorch.algorithms.ddpg.DDPG

    """

    _ALGORITHM_CLS = DDPG  # override

    # override
    @classmethod
    def get_worker_cls(cls) -> "Type[Trainable]":
        class Worker(super().get_worker_cls()):
            # override
            def setup(self, config: "Dict[str, Any]") -> "None":
                super().setup(config)

                # Target policy network
                self._target_policy_network_state_dict_keys = list(
                    self._target_policy_network.state_dict().keys()
                )
                self._target_policy_network_wrapped_model = (
                    self._target_policy_network.wrapped_model
                )
                self._target_policy_network_normalizing_flows = (
                    self._target_policy_network.normalizing_flows
                )
                prepare_model_kwargs = {}
                if not any(
                    x.requires_grad
                    for x in self._target_policy_network_wrapped_model.parameters()
                ):
                    prepare_model_kwargs = {"ddp_cls": None}
                self._target_policy_network.wrapped_model = prepare_model(
                    self._target_policy_network_wrapped_model,
                    **prepare_model_kwargs,
                )
                for k, v in self._target_policy_network_normalizing_flows.items():
                    model = v.model
                    prepare_model_kwargs = {}
                    if not any(x.requires_grad for x in model.parameters()):
                        prepare_model_kwargs = {"ddp_cls": None}
                    self._target_policy_network.normalizing_flows[
                        k
                    ].model = prepare_model(
                        model,
                        **prepare_model_kwargs,
                    )
                self._prepared_target_policy_network_state_dict_keys = list(
                    self._target_policy_network.state_dict().keys()
                )

                # Target value network
                self._target_value_network_state_dict_keys = list(
                    self._target_value_network.state_dict().keys()
                )
                self._target_value_network_wrapped_model = (
                    self._target_value_network.wrapped_model
                )
                self._target_value_network_normalizing_flows = (
                    self._target_value_network.normalizing_flows
                )
                prepare_model_kwargs = {}
                if not any(
                    x.requires_grad
                    for x in self._target_value_network_wrapped_model.parameters()
                ):
                    prepare_model_kwargs = {"ddp_cls": None}
                self._target_value_network.wrapped_model = prepare_model(
                    self._target_value_network_wrapped_model,
                    **prepare_model_kwargs,
                )
                for k, v in self._target_value_network_normalizing_flows.items():
                    model = v.model
                    prepare_model_kwargs = {}
                    if not any(x.requires_grad for x in model.parameters()):
                        prepare_model_kwargs = {"ddp_cls": None}
                    self._target_value_network.normalizing_flows[
                        k
                    ].model = prepare_model(
                        model,
                        **prepare_model_kwargs,
                    )
                self._prepared_target_value_network_state_dict_keys = list(
                    self._target_value_network.state_dict().keys()
                )

            # override
            @property
            def _checkpoint(self) -> "Dict[str, Any]":
                checkpoint = super()._checkpoint
                checkpoint["target_policy_network"] = {
                    self._target_policy_network_state_dict_keys[i]: v
                    for i, v in enumerate(checkpoint["target_policy_network"].values())
                }
                checkpoint["target_value_network"] = {
                    self._target_value_network_state_dict_keys[i]: v
                    for i, v in enumerate(checkpoint["target_value_network"].values())
                }
                return checkpoint

            # override
            @_checkpoint.setter
            def _checkpoint(self, value: "Dict[str, Any]") -> "None":
                value["target_policy_network"] = {
                    self._prepared_target_policy_network_state_dict_keys[i]: v
                    for i, v in enumerate(value["target_policy_network"].values())
                }
                value["target_value_network"] = {
                    self._prepared_target_value_network_state_dict_keys[i]: v
                    for i, v in enumerate(value["target_value_network"].values())
                }
                super()._checkpoint = value

        return Worker


#####################################################################################################
# DDPG._get_default_policy_network_distribution_builder implementation
#####################################################################################################


@DDPG._get_default_policy_network_distribution_builder.register(spaces.Box)
def _get_default_policy_network_distribution_builder_box(
    self,
    space: "spaces.Box",
) -> "Callable[..., Deterministic]":
    return Deterministic


#####################################################################################################
# DDPG._get_default_policy_network_distribution_parametrization implementation
#####################################################################################################


@DDPG._get_default_policy_network_distribution_parametrization.register(spaces.Box)
def _get_default_policy_network_distribution_parametrization_box(
    self,
    space: "spaces.Box",
) -> "DistributionParametrization":
    return {
        "value": (
            {"value": space.shape},
            lambda x: x["value"],
        ),
    }


#####################################################################################################
# DDPG._get_default_policy_network_distribution_config implementation
#####################################################################################################


@DDPG._get_default_policy_network_distribution_config.register(spaces.Box)
def _get_default_policy_network_distribution_config_box(
    self,
    space: "spaces.Box",
) -> "Dict[str, Any]":
    return {"validate_args": False}
