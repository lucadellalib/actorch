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

"""Twin Delayed Deep Deterministic Policy Gradient."""

import contextlib
import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from gymnasium import Env
from numpy import ndarray
from ray.tune import Trainable
from torch import Tensor
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from actorch.agents import Agent
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.algorithms.ddpg import DDPG, DistributedDataParallelDDPG, Loss, LRScheduler
from actorch.algorithms.utils import prepare_model, sync_polyak
from actorch.algorithms.value_estimation import n_step_return
from actorch.buffers import Buffer
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import Processor
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, Schedule


__all__ = [
    "DistributedDataParallelTD3",
    "TD3",
]


class TD3(DDPG):
    """Twin Delayed Deep Deterministic Policy Gradient.

    References
    ----------
    .. [1] S. Fujimoto, H. van Hoof, and D. Meger.
           "Addressing Function Approximation Error in Actor-Critic Methods".
           In: ICML. 2018, pp. 1587–1596.
           URL: https://arxiv.org/abs/1802.09477

    """

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
            delay: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 2,
            noise_stddev: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.2,
            noise_clip: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.5,
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
                delay=delay,
                noise_stddev=noise_stddev,
                noise_clip=noise_clip,
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
        self.config = TD3.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        self._twin_value_network = (
            self._build_value_network().train().to(self._device, non_blocking=True)
        )
        self._twin_value_network_loss = (
            self._build_value_network_loss().train().to(self._device, non_blocking=True)
        )
        self._twin_value_network_optimizer = self._build_twin_value_network_optimizer()
        self._twin_value_network_optimizer_lr_scheduler = (
            self._build_twin_value_network_optimizer_lr_scheduler()
        )
        self._target_twin_value_network = copy.deepcopy(self._twin_value_network)
        self._target_twin_value_network.eval().to(self._device, non_blocking=True)
        self._target_twin_value_network.requires_grad_(False)
        if not isinstance(self.delay, Schedule):
            self.delay = ConstantSchedule(self.delay)
        if not isinstance(self.noise_stddev, Schedule):
            self.noise_stddev = ConstantSchedule(self.noise_stddev)
        if not isinstance(self.noise_clip, Schedule):
            self.noise_clip = ConstantSchedule(self.noise_clip)

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        checkpoint["twin_value_network"] = self._twin_value_network.state_dict()
        checkpoint[
            "twin_value_network_loss"
        ] = self._twin_value_network_loss.state_dict()
        checkpoint[
            "twin_value_network_optimizer"
        ] = self._twin_value_network_optimizer.state_dict()
        if self._twin_value_network_optimizer_lr_scheduler is not None:
            checkpoint[
                "twin_value_network_optimizer_lr_scheduler"
            ] = self._twin_value_network_optimizer_lr_scheduler.state_dict()
        checkpoint[
            "target_twin_value_network"
        ] = self._target_twin_value_network.state_dict()
        checkpoint["delay"] = self.delay.state_dict()
        checkpoint["noise_stddev"] = self.noise_stddev.state_dict()
        checkpoint["noise_clip"] = self.noise_clip.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        self._twin_value_network.load_state_dict(value["twin_value_network"])
        self._twin_value_network_loss.load_state_dict(value["twin_value_network_loss"])
        self._twin_value_network_optimizer.load_state_dict(
            value["twin_value_network_optimizer"]
        )
        if "twin_value_network_optimizer_lr_scheduler" in value:
            self._twin_value_network_optimizer_lr_scheduler.load_state_dict(
                value["twin_value_network_optimizer_lr_scheduler"]
            )
        self._target_twin_value_network.load_state_dict(
            value["target_twin_value_network"]
        )
        self.delay.load_state_dict(value["delay"])
        self.noise_stddev.load_state_dict(value["noise_stddev"])
        self.noise_clip.load_state_dict(value["noise_clip"])

    def _build_twin_value_network_optimizer(self) -> "Optimizer":
        if self.value_network_optimizer_builder is None:
            self.value_network_optimizer_builder = Adam
            if self.value_network_optimizer_config is None:
                self.value_network_optimizer_config = {
                    "lr": 1e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "foreach": None,
                    "maximize": False,
                    "capturable": False,
                }
        if self.value_network_optimizer_config is None:
            self.value_network_optimizer_config = {}
        return self.value_network_optimizer_builder(
            self._twin_value_network.parameters(),
            **self.value_network_optimizer_config,
        )

    def _build_twin_value_network_optimizer_lr_scheduler(
        self,
    ) -> "Optional[LRScheduler]":
        if self.value_network_optimizer_lr_scheduler_builder is None:
            return
        if self.value_network_optimizer_lr_scheduler_config is None:
            self.value_network_optimizer_lr_scheduler_config: "Dict[str, Any]" = {}
        return self.value_network_optimizer_lr_scheduler_builder(
            self._twin_value_network_optimizer,
            **self.value_network_optimizer_lr_scheduler_config,
        )

    # override
    def _train_step(self) -> "Dict[str, Any]":
        result = super()._train_step()
        self.delay.step()
        self.noise_stddev.step()
        self.noise_clip.step()
        result["delay"] = self.delay()
        result["noise_stddev"] = self.noise_stddev()
        result["noise_clip"] = self.noise_clip()
        result["buffer_num_experiences"] = result.pop("buffer_num_experiences")
        result["buffer_num_full_trajectories"] = result.pop(
            "buffer_num_full_trajectories"
        )
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

        delay = self.delay()
        if delay < 1 or not float(delay).is_integer():
            raise ValueError(
                f"`delay` ({delay}) " f"must be in the integer interval [1, inf)"
            )
        delay = int(delay)

        noise_stddev = self.noise_stddev()
        if noise_stddev < 0.0:
            raise ValueError(
                f"`noise_stddev` ({noise_stddev}) must be in the interval [0, inf)"
            )

        noise_clip = self.noise_clip()
        if noise_clip < 0.0:
            raise ValueError(
                f"`noise_clip` ({noise_clip}) must be in the interval [0, inf)"
            )

        result = {}

        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            target_actions, _ = self._target_policy_network(
                experiences["observation"], mask=mask
            )
            target_actions += torch.normal(
                0.0,
                noise_stddev,
                target_actions.shape,
                device=target_actions.device,
            ).clamp(-noise_clip, noise_clip)
            # TODO: find a more elegant solution
            low = torch.as_tensor(
                self._train_agent._flat_action_space.low, device=target_actions.device
            )
            high = torch.as_tensor(
                self._train_agent._flat_action_space.high, device=target_actions.device
            )
            if self._train_agent._batch_size is not None:
                # Batched environment
                low, high = low[0], high[0]
            target_actions = target_actions.clamp(low, high)
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
            target_twin_action_values, _ = self._target_twin_value_network(
                observations_target_actions, mask=mask
            )
            target_action_values = target_action_values.min(target_twin_action_values)
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
            twin_action_values, _ = self._twin_value_network(
                observations_actions, mask=mask
            )

        result["value_network"], priority = self._train_on_batch_value_network(
            action_values,
            targets,
            is_weight,
            mask,
        )

        (
            result["twin_value_network"],
            twin_priority,
        ) = self._train_on_batch_twin_value_network(
            twin_action_values,
            targets,
            is_weight,
            mask,
        )

        if priority is not None:
            priority += twin_priority
            priority /= 2.0

        if idx % delay == 0:
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
                sync_polyak(
                    self._twin_value_network,
                    self._target_twin_value_network,
                    self.polyak_weight(),
                )

        self._grad_scaler.update()
        return result, priority

    def _train_on_batch_twin_value_network(
        self,
        action_values: "Tensor",
        targets: "Tensor",
        is_weight: "Tensor",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Any], Optional[ndarray]]":
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            action_values = action_values[mask]
            target = targets[mask]
            loss = self._twin_value_network_loss(action_values, target)
            loss *= is_weight[:, None].expand_as(mask)[mask]
            loss = loss.mean()
        optimize_result = self._optimize_twin_value_network(loss)
        priority = None
        result = {
            "action_value": action_values.mean().item(),
            "target": target.mean().item(),
            "loss": loss.item(),
        }
        result.update(optimize_result)
        return result, priority

    def _optimize_twin_value_network(self, loss: "Tensor") -> "Dict[str, Any]":
        max_grad_l2_norm = self.max_grad_l2_norm()
        if max_grad_l2_norm <= 0.0:
            raise ValueError(
                f"`max_grad_l2_norm` ({max_grad_l2_norm}) must be in the interval (0, inf]"
            )
        self._twin_value_network_optimizer.zero_grad(set_to_none=True)
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._twin_value_network_optimizer)
        grad_l2_norm = clip_grad_norm_(
            self._twin_value_network.parameters(), max_grad_l2_norm
        )
        self._grad_scaler.step(self._twin_value_network_optimizer)
        result = {
            "lr": self._twin_value_network_optimizer.param_groups[0]["lr"],
            "grad_l2_norm": min(grad_l2_norm.item(), max_grad_l2_norm),
        }
        if self._twin_value_network_optimizer_lr_scheduler is not None:
            self._twin_value_network_optimizer_lr_scheduler.step()
        return result


class DistributedDataParallelTD3(DistributedDataParallelDDPG):
    """Distributed data parallel Twin Delayed Deep Deterministic Policy Gradient.

    See Also
    --------
    actorch.algorithms.td3.TD3

    """

    _ALGORITHM_CLS = TD3  # override

    # override
    @classmethod
    def get_worker_cls(cls) -> "Type[Trainable]":
        class Worker(super().get_worker_cls()):
            # override
            def setup(self, config: "Dict[str, Any]") -> "None":
                super().setup(config)

                # Twin value network
                self._twin_value_network_state_dict_keys = list(
                    self._twin_value_network.state_dict().keys()
                )
                self._twin_value_network_wrapped_model = (
                    self._twin_value_network.wrapped_model
                )
                self._twin_value_network_normalizing_flows = (
                    self._twin_value_network.normalizing_flows
                )
                prepare_model_kwargs = {}
                if not any(
                    x.requires_grad
                    for x in self._twin_value_network_wrapped_model.parameters()
                ):
                    prepare_model_kwargs = {"ddp_cls": None}
                self._twin_value_network.wrapped_model = prepare_model(
                    self._twin_value_network_wrapped_model,
                    **prepare_model_kwargs,
                )
                for k, v in self._twin_value_network_normalizing_flows.items():
                    model = v.model
                    prepare_model_kwargs = {}
                    if not any(x.requires_grad for x in model.parameters()):
                        prepare_model_kwargs = {"ddp_cls": None}
                    self._twin_value_network.normalizing_flows[k].model = prepare_model(
                        model,
                        **prepare_model_kwargs,
                    )
                self._prepared_twin_value_network_state_dict_keys = list(
                    self._twin_value_network.state_dict().keys()
                )

                # Target twin value network
                self._target_twin_value_network_state_dict_keys = list(
                    self._target_twin_value_network.state_dict().keys()
                )
                self._target_twin_value_network_wrapped_model = (
                    self._target_twin_value_network.wrapped_model
                )
                self._target_twin_value_network_normalizing_flows = (
                    self._target_twin_value_network.normalizing_flows
                )
                prepare_model_kwargs = {}
                if not any(
                    x.requires_grad
                    for x in self._target_twin_value_network_wrapped_model.parameters()
                ):
                    prepare_model_kwargs = {"ddp_cls": None}
                self._target_twin_value_network.wrapped_model = prepare_model(
                    self._target_twin_value_network_wrapped_model,
                    **prepare_model_kwargs,
                )
                for k, v in self._target_twin_value_network_normalizing_flows.items():
                    model = v.model
                    prepare_model_kwargs = {}
                    if not any(x.requires_grad for x in model.parameters()):
                        prepare_model_kwargs = {"ddp_cls": None}
                    self._target_twin_value_network.normalizing_flows[
                        k
                    ].model = prepare_model(
                        model,
                        **prepare_model_kwargs,
                    )
                self._prepared_target_twin_value_network_state_dict_keys = list(
                    self._target_twin_value_network.state_dict().keys()
                )

            # override
            @property
            def _checkpoint(self) -> "Dict[str, Any]":
                checkpoint = super()._checkpoint
                checkpoint["twin_value_network"] = {
                    self._twin_value_network_state_dict_keys[i]: v
                    for i, v in enumerate(checkpoint["twin_value_network"].values())
                }
                checkpoint["target_twin_value_network"] = {
                    self._target_twin_value_network_state_dict_keys[i]: v
                    for i, v in enumerate(
                        checkpoint["target_twin_value_network"].values()
                    )
                }
                return checkpoint

            # override
            @_checkpoint.setter
            def _checkpoint(self, value: "Dict[str, Any]") -> "None":
                value["twin_value_network"] = {
                    self._prepared_twin_value_network_state_dict_keys[i]: v
                    for i, v in enumerate(value["twin_value_network"].values())
                }
                value["target_twin_value_network"] = {
                    self._prepared_target_twin_value_network_state_dict_keys[i]: v
                    for i, v in enumerate(value["target_twin_value_network"].values())
                }
                super()._checkpoint = value

        return Worker
