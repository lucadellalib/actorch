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

"""Trust Region Policy Optimization (TRPO)."""

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from gymnasium import Env
from numpy import ndarray
from torch import Tensor
from torch.cuda.amp import autocast
from torch.distributions import Distribution, kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from actorch.agents import Agent
from actorch.algorithms.a2c import A2C, DistributedDataParallelA2C, Loss, LRScheduler
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.algorithms.utils import normalize_
from actorch.algorithms.value_estimation import lambda_return
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import DistributionParametrization, NormalizingFlow, Processor
from actorch.optimizers import CGBLS
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, Schedule


__all__ = [
    "DistributedDataParallelTRPO",
    "TRPO",
]


class TRPO(A2C):
    """Trust Region Policy Optimization (TRPO).

    References
    ----------
    .. [1] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz.
           "Trust Region Policy Optimization".
           In: ICML. 2015, pp. 1889-1897.
           URL: https://arxiv.org/abs/1502.05477

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
            trace_decay: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.95,
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
                trace_decay=trace_decay,
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
        self.config = TRPO.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        if not isinstance(self.trace_decay, Schedule):
            self.trace_decay = ConstantSchedule(self.trace_decay)

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        checkpoint["trace_decay"] = self.trace_decay.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        self.trace_decay.load_state_dict(value["trace_decay"])

    # override
    def _build_policy_network_optimizer(self) -> "Optimizer":
        if self.policy_network_optimizer_builder is None:
            self.policy_network_optimizer_builder = CGBLS
            if self.policy_network_optimizer_config is None:
                self.policy_network_optimizer_config = {
                    "max_constraint": 0.01,
                    "num_cg_iters": 10,
                    "max_backtracks": 15,
                    "backtrack_ratio": 0.8,
                    "hvp_reg_coeff": 1e-5,
                    "accept_violation": False,
                    "epsilon": 1e-8,
                }
        if self.policy_network_optimizer_config is None:
            self.policy_network_optimizer_config = {}
        return self.policy_network_optimizer_builder(
            self._policy_network.parameters(),
            **self.policy_network_optimizer_config,
        )

    # override
    def _train_step(self) -> "Dict[str, Any]":
        result = super()._train_step()
        self.trace_decay.step()
        result["trace_decay"] = self.trace_decay()
        result["entropy_coeff"] = result.pop("entropy_coeff", None)
        result["max_grad_l2_norm"] = result.pop("max_grad_l2_norm", None)
        result.pop("num_return_steps", None)
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
                targets, advantages = lambda_return(
                    state_values,
                    experiences["reward"],
                    experiences["terminal"],
                    mask,
                    self.discount(),
                    self.trace_decay(),
                )
            if self.normalize_advantage:
                normalize_(advantages, dim=-1, mask=mask)

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

    # override
    def _train_on_batch_policy_network(
        self,
        experiences: "Dict[str, Tensor]",
        advantages: "Tensor",
        mask: "Tensor",
    ) -> "Dict[str, Any]":
        entropy_coeff = self.entropy_coeff()
        if entropy_coeff < 0.0:
            raise ValueError(
                f"`entropy_coeff` ({entropy_coeff}) must be in the interval [0, inf)"
            )
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            advantage = advantages[mask]
            old_log_prob = experiences["log_prob"][mask]
            with torch.no_grad():
                self._policy_network(experiences["observation"], mask=mask)
                old_policy = self._policy_network.distribution
        policy = loss = log_prob = entropy_bonus = kl_div = None

        def compute_loss() -> "Tensor":
            nonlocal policy, loss, log_prob, entropy_bonus
            with (
                autocast(**self.enable_amp)
                if self.enable_amp["enabled"]
                else contextlib.suppress()
            ):
                self._policy_network(experiences["observation"], mask=mask)
                policy = self._policy_network.distribution
                log_prob = policy.log_prob(experiences["action"])[mask]
                ratio = (log_prob - old_log_prob).exp()
                loss = -advantage * ratio
                entropy_bonus = None
                if entropy_coeff != 0.0:
                    entropy_bonus = -entropy_coeff * policy.entropy()[mask]
                    loss += entropy_bonus
                loss = loss.mean()
                return loss

        def compute_kl_div() -> "Tensor":
            nonlocal policy, kl_div
            with (
                autocast(**self.enable_amp)
                if self.enable_amp["enabled"]
                else contextlib.suppress()
            ):
                if policy is None:
                    self._policy_network(experiences["observation"], mask=mask)
                    policy = self._policy_network.distribution
                kl_div = kl_divergence(old_policy, policy)[mask].mean()
                policy = None
                return kl_div

        loss = compute_loss()
        optimize_result = self._optimize_policy_network(
            loss, compute_loss, compute_kl_div
        )
        result = {
            "advantage": advantage.mean().item(),
            "log_prob": log_prob.mean().item(),
            "old_log_prob": old_log_prob.mean().item(),
        }
        if kl_div is not None:
            result["kl_div"] = kl_div.item()
        result["loss"] = loss.item()
        if entropy_bonus is not None:
            result["entropy_bonus"] = entropy_bonus.mean().item()
        result.update(optimize_result)
        return result

    # override
    def _optimize_policy_network(
        self,
        loss: "Tensor",
        loss_fn: "Callable[[], Tensor]",
        constraint_fn: "Callable[[], Tensor]",
    ) -> "Dict[str, Any]":
        max_grad_l2_norm = self.max_grad_l2_norm()
        if max_grad_l2_norm <= 0.0:
            raise ValueError(
                f"`max_grad_l2_norm` ({max_grad_l2_norm}) must be in the interval (0, inf]"
            )
        self._policy_network_optimizer.zero_grad(set_to_none=True)
        self._grad_scaler.scale(loss).backward(retain_graph=True)
        self._grad_scaler.unscale_(self._policy_network_optimizer)
        grad_l2_norm = clip_grad_norm_(
            self._policy_network.parameters(), max_grad_l2_norm
        )
        self._grad_scaler.step(self._policy_network_optimizer, loss_fn, constraint_fn)
        result = {
            "grad_l2_norm": min(grad_l2_norm.item(), max_grad_l2_norm),
        }
        return result


class DistributedDataParallelTRPO(DistributedDataParallelA2C):
    """Distributed data parallel Trust Region Policy Optimization (TRPO).

    See Also
    --------
    actorch.algorithms.trpo.TRPO

    """

    _ALGORITHM_CLS = TRPO  # override
