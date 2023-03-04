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

"""Actor-Critic Kronecker-Factored Trust Region."""

from typing import Any, Callable, Dict, Optional, Union

from gymnasium import Env
from torch import Tensor
from torch.distributions import Distribution
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from actorch.agents import Agent
from actorch.algorithms.a2c import A2C, DistributedDataParallelA2C, Loss, LRScheduler
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import DistributionParametrization, NormalizingFlow, Processor
from actorch.preconditioners import KFAC, Preconditioner
from actorch.samplers import Sampler
from actorch.schedules import Schedule


__all__ = [
    "ACKTR",
    "DistributedDataParallelACKTR",
]


class ACKTR(A2C):
    """Actor-Critic Kronecker-Factored Trust Region.

    References
    ----------
    .. [1] Y. Wu, E. Mansimov, R. B. Grosse, S. Liao, and J. Ba.
           "Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation".
           In: NeurIPS. 2017, pp. 5279-5288.
           URL: https://arxiv.org/abs/1708.05144

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
            policy_network_preconditioner_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., Preconditioner]]]]" = None,
            policy_network_preconditioner_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            policy_network_preconditioner_lr_scheduler_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., LRScheduler]]]]" = None,
            policy_network_preconditioner_lr_scheduler_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
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
                policy_network_preconditioner_builder=policy_network_preconditioner_builder,
                policy_network_preconditioner_config=policy_network_preconditioner_config,
                policy_network_preconditioner_lr_scheduler_builder=policy_network_preconditioner_lr_scheduler_builder,
                policy_network_preconditioner_lr_scheduler_config=policy_network_preconditioner_lr_scheduler_config,
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
        self.config = ACKTR.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        self._policy_network_preconditioner = (
            self._build_policy_network_preconditioner()
        )
        self._policy_network_preconditioner_lr_scheduler = (
            self._build_policy_network_preconditioner_lr_scheduler()
        )

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        checkpoint[
            "policy_network_preconditioner"
        ] = self._policy_network_preconditioner.state_dict(
            include_module=False,
            include_grad_scaler=False,
        )
        if self._policy_network_preconditioner_lr_scheduler is not None:
            checkpoint[
                "policy_network_preconditioner_lr_scheduler"
            ] = self._policy_network_preconditioner_lr_scheduler.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        self._policy_network_preconditioner.load_state_dict(
            value["policy_network_preconditioner"]
        )
        if "policy_network_preconditioner_lr_scheduler" in value:
            self._policy_network_preconditioner_lr_scheduler.load_state_dict(
                value["policy_network_preconditioner_lr_scheduler"]
            )

    def _build_policy_network_preconditioner(self) -> "Optimizer":
        if self.policy_network_preconditioner_builder is None:
            self.policy_network_preconditioner_builder = KFAC
            if self.policy_network_preconditioner_config is None:
                self.policy_network_preconditioner_config = {
                    "lr": 0.1,
                    "factor_decay": 0.95,
                    "damping": 0.001,
                    "kl_clip": 0.001,
                    "factor_update_freq": 10,
                    "kfac_update_freq": 100,
                    "epsilon": 1e-6,
                }
        if self.policy_network_preconditioner_config is None:
            self.policy_network_preconditioner_config = {}
        return self.policy_network_preconditioner_builder(
            self._policy_network,
            grad_scaler=self._grad_scaler,
            **self.policy_network_preconditioner_config,
        )

    def _build_policy_network_preconditioner_lr_scheduler(
        self,
    ) -> "Optional[LRScheduler]":
        if self.policy_network_preconditioner_lr_scheduler_builder is None:
            return
        if self.policy_network_preconditioner_lr_scheduler_config is None:
            self.policy_network_preconditioner_lr_scheduler_config = {}
        return self.policy_network_preconditioner_lr_scheduler_builder(
            self._policy_network_preconditioner,
            **self.policy_network_preconditioner_lr_scheduler_config,
        )

    # override
    def _optimize_policy_network(self, loss: "Tensor") -> "Dict[str, Any]":
        max_grad_l2_norm = self.max_grad_l2_norm()
        if max_grad_l2_norm <= 0.0:
            raise ValueError(
                f"`max_grad_l2_norm` ({max_grad_l2_norm}) must be in the interval (0, inf]"
            )
        self._policy_network_optimizer.zero_grad(set_to_none=True)
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._policy_network_optimizer)
        grad_l2_norm = clip_grad_norm_(
            self._policy_network.parameters(), max_grad_l2_norm
        )
        self._policy_network_preconditioner.step()
        self._grad_scaler.step(self._policy_network_optimizer)
        result = {
            "lr": self._policy_network_optimizer.param_groups[0]["lr"],
            "grad_l2_norm": min(grad_l2_norm.item(), max_grad_l2_norm),
        }
        if self._policy_network_optimizer_lr_scheduler is not None:
            self._policy_network_optimizer_lr_scheduler.step()
        return result


class DistributedDataParallelACKTR(DistributedDataParallelA2C):
    """Distributed data parallel Actor-Critic Kronecker-Factored Trust Region.

    See Also
    --------
    actorch.algorithms.acktr.ACKTR

    """

    _ALGORITHM_CLS = ACKTR  # override
