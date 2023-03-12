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

"""Distributional Deep Deterministic Policy Gradient (D3PG)."""

import contextlib
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from gymnasium import Env
from numpy import ndarray
from torch import Tensor
from torch.cuda.amp import autocast
from torch.distributions import Distribution, kl_divergence
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from actorch.agents import Agent
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.algorithms.ddpg import DDPG, DistributedDataParallelDDPG, Loss, LRScheduler
from actorch.algorithms.utils import freeze_params, sync_polyak_
from actorch.algorithms.value_estimation import n_step_return
from actorch.buffers import Buffer, ProportionalBuffer
from actorch.distributions import Finite
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import (
    DistributionParametrization,
    Network,
    NormalizingFlow,
    Processor,
)
from actorch.samplers import Sampler
from actorch.schedules import Schedule


__all__ = [
    "D3PG",
    "DistributedDataParallelD3PG",
]


_LOGGER = logging.getLogger(__name__)


class D3PG(DDPG):
    """Distributional Deep Deterministic Policy Gradient (D3PG).

    References
    ----------
    .. [1] G. Barth-Maron, M. W. Hoffman, D. Budden, W. Dabney, D. Horgan, D. TB,
           A. Muldal, N. Heess, and T. Lillicrap.
           "Distributed Distributional Deterministic Policy Gradients".
           In: ICLR. 2018.
           URL: https://arxiv.org/abs/1804.08617

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
            value_network_distribution_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Distribution]]]]" = None,
            value_network_distribution_parametrization: "Tunable[RefOrFutureRef[Optional[DistributionParametrization]]]" = None,
            value_network_distribution_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            value_network_normalizing_flow: "Tunable[RefOrFutureRef[Optional[NormalizingFlow]]]" = None,
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
            num_return_steps: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 3,
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
                value_network_distribution_builder=value_network_distribution_builder,
                value_network_distribution_parametrization=value_network_distribution_parametrization,
                value_network_distribution_config=value_network_distribution_config,
                value_network_normalizing_flow=value_network_normalizing_flow,
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
        self.config = D3PG.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        self._warn_failed_logging = True

    # override
    def _build_buffer(self) -> "Buffer":
        if self.buffer_builder is None:
            self.buffer_builder = ProportionalBuffer
        if self.buffer_config is None:
            self.buffer_config = {
                "capacity": int(1e5),
                "prioritization": 1.0,
                "bias_correction": 0.4,
                "epsilon": 1e-5,
            }
        return super()._build_buffer()

    # override
    def _build_value_network(self) -> "Network":
        if self.value_network_distribution_builder is None:
            self.value_network_distribution_builder = Finite
            if self.value_network_distribution_parametrization is None:
                self.value_network_distribution_parametrization = {
                    "logits": (
                        {"logits": (51,)},
                        lambda x: x["logits"],
                    ),
                }
            if self.value_network_distribution_config is None:
                self.value_network_distribution_config = {
                    "atoms": torch.linspace(-10.0, 10.0, 51).to(self._device),
                    "validate_args": False,
                }
        if self.value_network_distribution_parametrization is None:
            self.value_network_distribution_parametrization = {}
        if self.value_network_distribution_config is None:
            self.value_network_distribution_config = {}

        if self.value_network_normalizing_flow is not None:
            self.value_network_normalizing_flows = {
                "value": self.value_network_normalizing_flow,
            }
        else:
            self.value_network_normalizing_flows: "Dict[str, NormalizingFlow]" = {}

        self.value_network_distribution_builders = {
            "value": self.value_network_distribution_builder,
        }
        self.value_network_distribution_parametrizations = {
            "value": self.value_network_distribution_parametrization,
        }
        self.value_network_distribution_configs = {
            "value": self.value_network_distribution_config,
        }
        return super()._build_value_network()

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
            self._target_value_network(observations_target_actions, mask=mask)
            target_action_values = self._target_value_network.distribution
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
            self._value_network(observations_actions, mask=mask)
            action_values = self._value_network.distribution

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
            sync_polyak_(
                self._policy_network,
                self._target_policy_network,
                self.polyak_weight(),
            )
            sync_polyak_(
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
                self._value_network(observations_actions, mask=mask)
            action_values = self._value_network.distribution
            try:
                action_values = action_values.mean
            except Exception as e:
                raise RuntimeError(f"Could not compute `action_values.mean`: {e}")
            action_value = action_values[mask]
            loss = -action_value.mean()
        optimize_result = self._optimize_policy_network(loss)
        result = {"loss": loss.item()}
        result.update(optimize_result)
        return result

    # override
    def _train_on_batch_value_network(
        self,
        action_values: "Distribution",
        targets: "Distribution",
        is_weight: "Tensor",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Any], Optional[ndarray]]":
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            loss = kl_divergence(targets, action_values)[mask]
            priority = None
            if self._buffer.is_prioritized:
                loss *= is_weight[:, None].expand_as(mask)[mask]
                priority = loss.detach().abs().to("cpu").numpy()
            loss = loss.mean()
        optimize_result = self._optimize_value_network(loss)
        result = {}
        try:
            result["action_value"] = action_values.mean[mask].mean().item()
            result["target"] = targets.mean[mask].mean().item()
        except Exception as e:
            if self._warn_failed_logging:
                _LOGGER.warning(f"Could not log `action_value` and/or `target`: {e}")
                self._warn_failed_logging = False
        result["loss"] = loss.item()
        result.update(optimize_result)
        return result, priority


class DistributedDataParallelD3PG(DistributedDataParallelDDPG):
    """Distributed data parallel Distributional Deep Deterministic Policy Gradient (D3PG).

    See Also
    --------
    actorch.algorithms.d3pg.D3PG

    """

    _ALGORITHM_CLS = D3PG  # override
