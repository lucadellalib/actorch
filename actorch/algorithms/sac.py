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

"""Soft Actor-Critic (SAC)."""

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from gymnasium import Env, spaces
from numpy import ndarray
from torch import Tensor
from torch.cuda.amp import autocast
from torch.distributions import Distribution, Normal
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from actorch.agents import Agent
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.algorithms.td3 import TD3, DistributedDataParallelTD3, Loss, LRScheduler
from actorch.algorithms.utils import freeze_params, sync_polyak_
from actorch.algorithms.value_estimation import n_step_return
from actorch.buffers import Buffer
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import DistributionParametrization, NormalizingFlow, Processor
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, Schedule
from actorch.utils import singledispatchmethod


__all__ = [
    "DistributedDataParallelSAC",
    "SAC",
]


class SAC(TD3):
    """Soft Actor-Critic (SAC).

    References
    ----------
    .. [1] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine.
           "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor".
           In: ICML. 2018, pp. 1861-1870.
           URL: https://arxiv.org/abs/1801.01290
    .. [2] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar,
           H. Zhu, A. Gupta, P. Abbeel, and S. Levine.
           "Soft Actor-Critic Algorithms and Applications".
           In: arXiv. 2018.
           URL: https://arxiv.org/abs/1812.05905

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
            policy_network_distribution_builders: "Tunable[RefOrFutureRef[Optional[Dict[str, Callable[..., Distribution]]]]]" = None,
            policy_network_distribution_parametrizations: "Tunable[RefOrFutureRef[Optional[Dict[str, DistributionParametrization]]]]" = None,
            policy_network_distribution_configs: "Tunable[RefOrFutureRef[Optional[Dict[str, Dict[str, Any]]]]]" = None,
            policy_network_normalizing_flows: "Tunable[RefOrFutureRef[Optional[Dict[str, NormalizingFlow]]]]" = None,
            policy_network_sample_fn: "Tunable[RefOrFutureRef[Optional[Callable[[Distribution], Tensor]]]]" = None,
            policy_network_prediction_fn: "Tunable[RefOrFutureRef[Optional[Callable[[Tensor], Tensor]]]]" = None,
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
            temperature_optimizer_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Optimizer]]]]" = None,
            temperature_optimizer_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            temperature_optimizer_lr_scheduler_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., LRScheduler]]]]" = None,
            temperature_optimizer_lr_scheduler_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
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
            temperature: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.1,
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
                policy_network_distribution_builders=policy_network_distribution_builders,
                policy_network_distribution_parametrizations=policy_network_distribution_parametrizations,
                policy_network_distribution_configs=policy_network_distribution_configs,
                policy_network_normalizing_flows=policy_network_normalizing_flows,
                policy_network_sample_fn=policy_network_sample_fn,
                policy_network_prediction_fn=policy_network_prediction_fn,
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
                temperature_optimizer_builder=temperature_optimizer_builder,
                temperature_optimizer_config=temperature_optimizer_config,
                temperature_optimizer_lr_scheduler_builder=temperature_optimizer_lr_scheduler_builder,
                temperature_optimizer_lr_scheduler_config=temperature_optimizer_lr_scheduler_config,
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
                temperature=temperature,
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
        self.config = SAC.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        if self.temperature_optimizer_builder is not None:
            self._log_temperature = torch.zeros(
                1, device=self._device, requires_grad=True
            )
            self._target_entropy = torch.as_tensor(
                self._train_env.single_action_space.sample()
            ).numel()
            self._temperature_optimizer = self._build_temperature_optimizer()
            self._temperature_optimizer_lr_scheduler = (
                self._build_temperature_optimizer_lr_scheduler()
            )
            self.temperature = lambda *args: self._log_temperature.exp().item()
        elif not isinstance(self.temperature, Schedule):
            self.temperature = ConstantSchedule(self.temperature)

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        if self.temperature_optimizer_builder is not None:
            checkpoint["log_temperature"] = self._log_temperature
            checkpoint[
                "temperature_optimizer"
            ] = self._temperature_optimizer.state_dict()
        if self._temperature_optimizer_lr_scheduler is not None:
            checkpoint[
                "temperature_optimizer_lr_scheduler"
            ] = self._temperature_optimizer_lr_scheduler.state_dict()
        checkpoint["temperature"] = self.temperature.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        if "log_temperature" in value:
            self._log_temperature = value["log_temperature"]
        if "temperature_optimizer" in value:
            self._temperature_optimizer.load_state_dict(value["temperature_optimizer"])
        if "temperature_optimizer_lr_scheduler" in value:
            self._temperature_optimizer_lr_scheduler.load_state_dict(
                value["temperature_optimizer_lr_scheduler"]
            )
        self.temperature.load_state_dict(value["temperature"])

    def _build_temperature_optimizer(self) -> "Optimizer":
        if self.temperature_optimizer_config is None:
            self.temperature_optimizer_config: "Dict[str, Any]" = {}
        return self.temperature_optimizer_builder(
            [self._log_temperature],
            **self.temperature_optimizer_config,
        )

    def _build_temperature_optimizer_lr_scheduler(
        self,
    ) -> "Optional[LRScheduler]":
        if self.temperature_optimizer_lr_scheduler_builder is None:
            return
        if self.temperature_optimizer_lr_scheduler_config is None:
            self.temperature_optimizer_lr_scheduler_config: "Dict[str, Any]" = {}
        return self.temperature_optimizer_lr_scheduler_builder(
            self._temperature_optimizer,
            **self.temperature_optimizer_lr_scheduler_config,
        )

    # override
    def _train_step(self) -> "Dict[str, Any]":
        result = super()._train_step()
        if self.temperature_optimizer_builder is None:
            result["temperature"] = self.temperature()
        result["max_grad_l2_norm"] = result.pop("max_grad_l2_norm", None)
        result["buffer_num_experiences"] = result.pop("buffer_num_experiences", None)
        result["buffer_num_full_trajectories"] = result.pop(
            "buffer_num_full_trajectories", None
        )
        result.pop("delay", None)
        result.pop("noise_stddev", None)
        result.pop("noise_clip", None)
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

        temperature = self.temperature()
        if temperature <= 0.0:
            raise ValueError(
                f"`temperature` ({temperature}) must be in the interval (0, inf)"
            )

        result = {}

        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            self._policy_network(experiences["observation"], mask=mask)
            policy = self._policy_network.distribution
            target_actions = policy.rsample()
            target_log_probs = policy.log_prob(target_actions)
            observations_target_actions = torch.cat(
                [
                    x[..., None] if x.shape == mask.shape else x
                    for x in [experiences["observation"], target_actions]
                ],
                dim=-1,
            )
            with torch.no_grad():
                target_action_values, _ = self._target_value_network(
                    observations_target_actions, mask=mask
                )
                target_twin_action_values, _ = self._target_twin_value_network(
                    observations_target_actions, mask=mask
                )
            target_action_values = target_action_values.min(target_twin_action_values)
            with torch.no_grad():
                target_action_values -= temperature * target_log_probs

            # Discard next observation
            experiences["observation"] = experiences["observation"][:, :-1, ...]
            observations_target_actions = observations_target_actions[:, :-1, ...]
            target_log_probs = target_log_probs[:, :-1, ...]
            mask = mask[:, 1:]
            targets, _ = n_step_return(
                target_action_values,
                experiences["reward"],
                experiences["terminal"],
                mask,
                self.discount(),
                self.num_return_steps(),
                return_advantage=False,
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

        result["policy_network"] = self._train_on_batch_policy_network(
            observations_target_actions,
            target_log_probs,
            mask,
        )

        if self.temperature_optimizer_builder is not None:
            result["temperature"] = self._train_on_batch_temperature(
                target_log_probs,
                mask,
            )

        # Synchronize
        if idx % sync_freq == 0:
            sync_polyak_(
                self._value_network,
                self._target_value_network,
                self.polyak_weight(),
            )
            sync_polyak_(
                self._twin_value_network,
                self._target_twin_value_network,
                self.polyak_weight(),
            )

        self._grad_scaler.update()
        return result, priority

    # override
    def _train_on_batch_policy_network(
        self,
        observations_actions: "Tensor",
        log_probs: "Tensor",
        mask: "Tensor",
    ) -> "Dict[str, Any]":
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            with freeze_params(self._value_network, self._twin_value_network):
                action_values, _ = self._value_network(observations_actions, mask=mask)
                twin_action_values, _ = self._twin_value_network(
                    observations_actions, mask=mask
                )
            action_values = action_values.min(twin_action_values)
            log_prob = log_probs[mask]
            action_value = action_values[mask]
            loss = self.temperature() * log_prob
            loss -= action_value
            loss = loss.mean()
        optimize_result = self._optimize_policy_network(loss)
        result = {
            "log_prob": log_prob.mean().item(),
            "action_value": action_value.mean().item(),
            "loss": loss.item(),
        }
        result.update(optimize_result)
        return result

    def _train_on_batch_temperature(
        self,
        log_probs: "Tensor",
        mask: "Tensor",
    ) -> "Dict[str, Any]":
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            with torch.no_grad():
                log_prob = log_probs[mask]
            loss = log_prob + self._target_entropy
            loss *= -self._log_temperature
            loss = loss.mean()
        optimize_result = self._optimize_temperature(loss)
        result = {
            "temperature": self.temperature(),
            "log_prob": log_prob.mean().item(),
            "loss": loss.item(),
        }
        result.update(optimize_result)
        return result

    def _optimize_temperature(self, loss: "Tensor") -> "Dict[str, Any]":
        max_grad_l2_norm = self.max_grad_l2_norm()
        if max_grad_l2_norm <= 0.0:
            raise ValueError(
                f"`max_grad_l2_norm` ({max_grad_l2_norm}) must be in the interval (0, inf]"
            )
        self._temperature_optimizer.zero_grad(set_to_none=True)
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._temperature_optimizer)
        grad_l2_norm = clip_grad_norm_(self._log_temperature, max_grad_l2_norm)
        self._grad_scaler.step(self._temperature_optimizer)
        result = {
            "lr": self._temperature_optimizer.param_groups[0]["lr"],
            "grad_l2_norm": min(grad_l2_norm.item(), max_grad_l2_norm),
        }
        if self._temperature_optimizer_lr_scheduler is not None:
            self._temperature_optimizer_lr_scheduler.step()
        return result

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


class DistributedDataParallelSAC(DistributedDataParallelTD3):
    """Distributed data parallel Soft Actor-Critic (SAC).

    See Also
    --------
    actorch.algorithms.sac.SAC

    """

    _ALGORITHM_CLS = SAC  # override


#####################################################################################################
# SAC._get_default_policy_network_distribution_builder implementation
#####################################################################################################


@SAC._get_default_policy_network_distribution_builder.register(spaces.Box)
def _get_default_policy_network_distribution_builder_box(
    self,
    space: "spaces.Box",
) -> "Callable[..., Normal]":
    return Normal


#####################################################################################################
# SAC._get_default_policy_network_distribution_parametrization implementation
#####################################################################################################


@SAC._get_default_policy_network_distribution_parametrization.register(spaces.Box)
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


#####################################################################################################
# SAC._get_default_policy_network_distribution_config implementation
#####################################################################################################


@SAC._get_default_policy_network_distribution_config.register(spaces.Box)
def _get_default_policy_network_distribution_config_box(
    self,
    space: "spaces.Box",
) -> "Dict[str, Any]":
    return {"validate_args": False}
