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

"""Advantage Actor-Critic."""

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from gymnasium import Env, spaces
from numpy import ndarray
from ray.tune import Trainable
from torch import Tensor
from torch.cuda.amp import autocast
from torch.distributions import Distribution
from torch.nn.modules import loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from actorch.agents import Agent
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.algorithms.reinforce import (
    REINFORCE,
    DistributedDataParallelREINFORCE,
    LRScheduler,
)
from actorch.algorithms.utils import prepare_model
from actorch.algorithms.value_estimation import n_step_return
from actorch.distributions import Deterministic
from actorch.envs import BatchedEnv
from actorch.models import FCNet, Model
from actorch.networks import (
    DistributionParametrization,
    Network,
    NormalizingFlow,
    Processor,
)
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, Schedule
from actorch.utils import singledispatchmethod


__all__ = [
    "A2C",
    "DistributedDataParallelA2C",
]


Loss = loss._Loss
"""Loss function."""


class A2C(REINFORCE):
    """Advantage Actor-Critic.

    References
    ----------
    .. [1] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu.
           "Asynchronous Methods for Deep Reinforcement Learning".
           In: ICML. 2016, pp. 1928-1937.
           URL: https://arxiv.org/abs/1602.01783

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
            dataloader_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., DataLoader]]]]" = None,
            dataloader_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            discount: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.99,
            num_return_steps: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 10,
            normalize_advantage: "Tunable[RefOrFutureRef[bool]]" = False,
            entropy_coeff: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.01,
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
                dataloader_builder=dataloader_builder,
                dataloader_config=dataloader_config,
                discount=discount,
                num_return_steps=num_return_steps,
                normalize_advantage=normalize_advantage,
                entropy_coeff=entropy_coeff,
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
        self.config = A2C.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        self._value_network = (
            self._build_value_network().train().to(self._device, non_blocking=True)
        )
        self._value_network_loss = (
            self._build_value_network_loss().train().to(self._device, non_blocking=True)
        )
        self._value_network_optimizer = self._build_value_network_optimizer()
        self._value_network_optimizer_lr_scheduler = (
            self._build_value_network_optimizer_lr_scheduler()
        )
        if not isinstance(self.num_return_steps, Schedule):
            self.num_return_steps = ConstantSchedule(self.num_return_steps)

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        checkpoint["value_network"] = self._value_network.state_dict()
        checkpoint["value_network_loss"] = self._value_network_loss.state_dict()
        checkpoint[
            "value_network_optimizer"
        ] = self._value_network_optimizer.state_dict()
        if self._value_network_optimizer_lr_scheduler is not None:
            checkpoint[
                "value_network_optimizer_lr_scheduler"
            ] = self._value_network_optimizer_lr_scheduler.state_dict()
        checkpoint["num_return_steps"] = self.num_return_steps.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        self._value_network.load_state_dict(value["value_network"])
        self._value_network_loss.load_state_dict(value["value_network_loss"])
        self._value_network_optimizer.load_state_dict(value["value_network_optimizer"])
        if "value_network_optimizer_lr_scheduler" in value:
            self._value_network_optimizer_lr_scheduler.load_state_dict(
                value["value_network_optimizer_lr_scheduler"]
            )
        self.num_return_steps.load_state_dict(value["num_return_steps"])

    def _build_value_network(self) -> "Network":
        if self.value_network_preprocessors is None:
            self.value_network_preprocessors: "Dict[str, Processor]" = {}
        unnested_single_observation_space = {
            f"observation{k}": v
            for k, v in self._train_env.single_observation_space.unnested.items()
        }
        unexpected_keys = (
            self.value_network_preprocessors.keys()
            - unnested_single_observation_space.keys()
        )
        if not unexpected_keys:
            # Order matters
            self.value_network_preprocessors = {
                key: self.value_network_preprocessors.get(
                    key,
                    self._get_default_value_network_preprocessor(space),
                )
                for key, space in unnested_single_observation_space.items()
            }

        if self.value_network_model_builder is None:
            self.value_network_model_builder = FCNet
            if self.value_network_model_config is None:
                self.value_network_model_config = {
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
        if self.value_network_model_config is None:
            self.value_network_model_config = {}

        if self.value_network_distribution_builders is None:
            self.value_network_distribution_builders = {
                "value": Deterministic,
            }
            if self.value_network_distribution_parametrizations is None:
                self.value_network_distribution_parametrizations = {
                    "value": {
                        "value": (
                            {"value": ()},
                            lambda x: x["value"],
                        ),
                    }
                }
            if self.value_network_distribution_configs is None:
                self.value_network_distribution_configs = {
                    "value": {"validate_args": False},
                }
        if self.value_network_distribution_parametrizations is None:
            self.value_network_distribution_parametrizations = {}
        if self.value_network_distribution_configs is None:
            self.value_network_distribution_configs = {}

        if self.value_network_normalizing_flows is None:
            self.value_network_normalizing_flows: "Dict[str, NormalizingFlow]" = {}

        value_network = Network(
            self.value_network_preprocessors,
            self.value_network_model_builder,
            self.value_network_distribution_builders,
            self.value_network_distribution_parametrizations,
            self.value_network_model_config,
            self.value_network_distribution_configs,
            self.value_network_normalizing_flows,
        )
        self._log_graph(value_network.wrapped_model.model, "value_network_model")
        return value_network

    def _build_value_network_loss(self) -> "Loss":
        if self.value_network_loss_builder is None:
            self.value_network_loss_builder = torch.nn.MSELoss
        if self.value_network_loss_config is None:
            self.value_network_loss_config: "Dict[str, Any]" = {}
        return self.value_network_loss_builder(
            reduction="none",
            **self.value_network_loss_config,
        )

    def _build_value_network_optimizer(self) -> "Optimizer":
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
            self._value_network.parameters(),
            **self.value_network_optimizer_config,
        )

    def _build_value_network_optimizer_lr_scheduler(
        self,
    ) -> "Optional[LRScheduler]":
        if self.value_network_optimizer_lr_scheduler_builder is None:
            return
        if self.value_network_optimizer_lr_scheduler_config is None:
            self.value_network_optimizer_lr_scheduler_config: "Dict[str, Any]" = {}
        return self.value_network_optimizer_lr_scheduler_builder(
            self._value_network_optimizer,
            **self.value_network_optimizer_lr_scheduler_config,
        )

    # override
    def _train_step(self) -> "Dict[str, Any]":
        result = super()._train_step()
        self.num_return_steps.step()
        result["num_return_steps"] = self.num_return_steps()
        return result

    # override
    def _train_on_batch(
        self,
        idx: "int",
        experiences: "Dict[str, Tensor]",
        is_weight: "Tensor",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Any], Optional[ndarray]]":
        result = {}

        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            state_values, _ = self._value_network(experiences["observation"], mask=mask)
            # Discard next observation
            experiences["observation"] = experiences["observation"][:, :-1, ...]
            mask = mask[:, 1:]
            with torch.no_grad():
                targets, advantages = n_step_return(
                    state_values,
                    experiences["reward"],
                    experiences["terminal"],
                    mask,
                    self.discount(),
                    self.num_return_steps(),
                )
            if self.normalize_advantage:
                length = mask.sum(dim=1, keepdim=True)
                advantages_mean = advantages.sum(dim=1, keepdim=True) / length
                advantages -= advantages_mean
                advantages *= mask
                advantages_stddev = (
                    ((advantages**2).sum(dim=1, keepdim=True) / length)
                    .sqrt()
                    .clamp(min=1e-6)
                )
                advantages /= advantages_stddev
                advantages *= mask

        # Discard next state value
        state_values = state_values[:, :-1]

        result["value_network"], priority = self._train_on_batch_value_network(
            state_values,
            targets,
            is_weight,
            mask,
        )
        result["policy_network"] = self._train_on_batch_policy_network(
            experiences,
            advantages,
            mask,
        )
        self._grad_scaler.update()
        return result, priority

    def _train_on_batch_value_network(
        self,
        state_values: "Tensor",
        targets: "Tensor",
        is_weight: "Tensor",
        mask: "Tensor",
    ) -> "Tuple[Dict[str, Any], Optional[ndarray]]":
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            state_value = state_values[mask]
            target = targets[mask]
            loss = self._value_network_loss(state_value, target)
            loss *= is_weight[:, None].expand_as(mask)[mask]
            loss = loss.mean()
        optimize_result = self._optimize_value_network(loss)
        priority = None
        result = {
            "state_value": state_value.mean().item(),
            "target": target.mean().item(),
            "loss": loss.item(),
        }
        result.update(optimize_result)
        return result, priority

    def _optimize_value_network(self, loss: "Tensor") -> "Dict[str, Any]":
        max_grad_l2_norm = self.max_grad_l2_norm()
        if max_grad_l2_norm <= 0.0:
            raise ValueError(
                f"`max_grad_l2_norm` ({max_grad_l2_norm}) must be in the interval (0, inf]"
            )
        self._value_network_optimizer.zero_grad(set_to_none=True)
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._value_network_optimizer)
        grad_l2_norm = clip_grad_norm_(
            self._value_network.parameters(), max_grad_l2_norm
        )
        self._grad_scaler.step(self._value_network_optimizer)
        result = {
            "lr": self._value_network_optimizer.param_groups[0]["lr"],
            "grad_l2_norm": min(grad_l2_norm.item(), max_grad_l2_norm),
        }
        if self._value_network_optimizer_lr_scheduler is not None:
            self._value_network_optimizer_lr_scheduler.step()
        return result

    @singledispatchmethod(use_weakrefs=False)
    def _get_default_value_network_preprocessor(
        self, space: "spaces.Space"
    ) -> "Processor":
        return self._get_default_policy_network_preprocessor(space)


class DistributedDataParallelA2C(DistributedDataParallelREINFORCE):
    """Distributed data parallel Advantage Actor-Critic.

    See Also
    --------
    actorch.algorithms.a2c.A2C

    """

    _ALGORITHM_CLS = A2C  # override

    # override
    @classmethod
    def get_worker_cls(cls) -> "Type[Trainable]":
        class Worker(super().get_worker_cls()):
            # override
            def setup(self, config: "Dict[str, Any]") -> "None":
                super().setup(config)
                self._value_network_state_dict_keys = list(
                    self._value_network.state_dict().keys()
                )
                self._value_network_wrapped_model = self._value_network.wrapped_model
                self._value_network_normalizing_flows = (
                    self._value_network.normalizing_flows
                )
                prepare_model_kwargs = {}
                if not any(
                    x.requires_grad
                    for x in self._value_network_wrapped_model.parameters()
                ):
                    prepare_model_kwargs = {"ddp_cls": None}
                self._value_network.wrapped_model = prepare_model(
                    self._value_network_wrapped_model,
                    **prepare_model_kwargs,
                )
                for k, v in self._value_network_normalizing_flows.items():
                    model = v.model
                    prepare_model_kwargs = {}
                    if not any(x.requires_grad for x in model.parameters()):
                        prepare_model_kwargs = {"ddp_cls": None}
                    self._value_network.normalizing_flows[k].model = prepare_model(
                        model,
                        **prepare_model_kwargs,
                    )
                self._prepared_value_network_state_dict_keys = list(
                    self._value_network.state_dict().keys()
                )

            # override
            @property
            def _checkpoint(self) -> "Dict[str, Any]":
                checkpoint = super()._checkpoint
                checkpoint["value_network"] = {
                    self._value_network_state_dict_keys[i]: v
                    for i, v in enumerate(checkpoint["value_network"].values())
                }
                return checkpoint

            # override
            @_checkpoint.setter
            def _checkpoint(self, value: "Dict[str, Any]") -> "None":
                value["value_network"] = {
                    self._prepared_value_network_state_dict_keys[i]: v
                    for i, v in enumerate(value["value_network"].values())
                }
                super()._checkpoint = value

        return Worker
