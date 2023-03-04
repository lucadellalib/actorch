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

"""REINFORCE."""

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

from gymnasium import Env
from numpy import ndarray
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Distribution
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from actorch.agents import Agent
from actorch.algorithms.algorithm import (
    Algorithm,
    DistributedDataParallelAlgorithm,
    RefOrFutureRef,
    Tunable,
)
from actorch.algorithms.value_estimation import monte_carlo_return
from actorch.buffers import Buffer
from actorch.datasets import BufferDataset
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import DistributionParametrization, NormalizingFlow, Processor
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, LambdaSchedule, Schedule


__all__ = [
    "DistributedDataParallelREINFORCE",
    "REINFORCE",
]


LRScheduler = lr_scheduler._LRScheduler
"""Learning rate scheduler."""


class REINFORCE(Algorithm):
    """REINFORCE.

    References
    ----------
    .. [1] R. J. Williams.
           "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning".
           In: Mach. Learn. 1992, pp. 229-256.
           URL: https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

    """

    _UPDATE_BUFFER_DATASET_SCHEDULES_AFTER_TRAIN_EPOCH = False  # override

    _RESET_BUFFER = True

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
            dataloader_builder: "Tunable[RefOrFutureRef[Optional[Callable[..., DataLoader]]]]" = None,
            dataloader_config: "Tunable[RefOrFutureRef[Optional[Dict[str, Any]]]]" = None,
            discount: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.99,
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
                dataloader_builder=dataloader_builder,
                dataloader_config=dataloader_config,
                discount=discount,
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
        self.config = REINFORCE.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        self._policy_network_optimizer = self._build_policy_network_optimizer()
        self._policy_network_optimizer_lr_scheduler = (
            self._build_policy_network_optimizer_lr_scheduler()
        )
        self._grad_scaler = GradScaler(enabled=self.enable_amp["enabled"])
        if not isinstance(self.discount, Schedule):
            self.discount = ConstantSchedule(self.discount)
        if not isinstance(self.entropy_coeff, Schedule):
            self.entropy_coeff = ConstantSchedule(self.entropy_coeff)
        if not isinstance(self.max_grad_l2_norm, Schedule):
            self.max_grad_l2_norm = ConstantSchedule(self.max_grad_l2_norm)

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        checkpoint[
            "policy_network_optimizer"
        ] = self._policy_network_optimizer.state_dict()
        if self._policy_network_optimizer_lr_scheduler is not None:
            checkpoint[
                "policy_network_optimizer_lr_scheduler"
            ] = self._policy_network_optimizer_lr_scheduler.state_dict()
        checkpoint["grad_scaler"] = self._grad_scaler.state_dict()
        checkpoint["discount"] = self.discount.state_dict()
        checkpoint["entropy_coeff"] = self.entropy_coeff.state_dict()
        checkpoint["max_grad_l2_norm"] = self.max_grad_l2_norm.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        self._policy_network_optimizer.load_state_dict(
            value["policy_network_optimizer"]
        )
        if "policy_network_optimizer_lr_scheduler" in value:
            self._policy_network_optimizer_lr_scheduler.load_state_dict(
                value["policy_network_optimizer_lr_scheduler"]
            )
        self._grad_scaler.load_state_dict(value["grad_scaler"])
        self.discount.load_state_dict(value["discount"])
        self.entropy_coeff.load_state_dict(value["entropy_coeff"])
        self.max_grad_l2_norm.load_state_dict(value["max_grad_l2_norm"])

    # override
    def _build_buffer(self) -> "Buffer":
        if self.buffer_config is None:
            self.buffer_config = {"capacity": float("inf")}
        return super()._build_buffer()

    # override
    def _build_buffer_dataset(self) -> "BufferDataset":
        if self.buffer_dataset_config is None:
            batch_size = self.train_num_episodes_per_iter
            if batch_size is None:
                batch_size = LambdaSchedule(
                    lambda *args, **kwargs: self._buffer.num_full_trajectories
                )
            self.buffer_dataset_config = {
                "batch_size": batch_size,
                "max_trajectory_length": float("inf"),
                "num_iters": 1,
            }
        return super()._build_buffer_dataset()

    def _build_policy_network_optimizer(self) -> "Optimizer":
        if self.policy_network_optimizer_builder is None:
            self.policy_network_optimizer_builder = Adam
            if self.policy_network_optimizer_config is None:
                self.policy_network_optimizer_config = {
                    "lr": 1e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "foreach": None,
                    "maximize": False,
                    "capturable": False,
                }
        if self.policy_network_optimizer_config is None:
            self.policy_network_optimizer_config = {}
        return self.policy_network_optimizer_builder(
            self._policy_network.parameters(),
            **self.policy_network_optimizer_config,
        )

    def _build_policy_network_optimizer_lr_scheduler(
        self,
    ) -> "Optional[LRScheduler]":
        if self.policy_network_optimizer_lr_scheduler_builder is None:
            return
        if self.policy_network_optimizer_lr_scheduler_config is None:
            self.policy_network_optimizer_lr_scheduler_config = {}
        return self.policy_network_optimizer_lr_scheduler_builder(
            self._policy_network_optimizer,
            **self.policy_network_optimizer_lr_scheduler_config,
        )

    # override
    def _train_step(self) -> "Dict[str, Any]":
        result = super()._train_step()
        if self._RESET_BUFFER:
            self._buffer.reset()
        self.discount.step()
        self.entropy_coeff.step()
        self.max_grad_l2_norm.step()
        result["discount"] = self.discount()
        result["entropy_coeff"] = self.entropy_coeff()
        result["max_grad_l2_norm"] = (
            self.max_grad_l2_norm()
            if self.max_grad_l2_norm() != float("inf")
            else "inf"
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
        result = {}

        # Discard next observation
        experiences["observation"] = experiences["observation"][:, :-1, ...]
        mask = mask[:, 1:]
        _, advantages = monte_carlo_return(
            experiences["reward"],
            mask,
            self.discount(),
        )

        result["policy_network"] = self._train_on_batch_policy_network(
            experiences,
            advantages,
            mask,
        )
        self._grad_scaler.update()
        return result, None

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
            self._policy_network(experiences["observation"], mask=mask)
            policy = self._policy_network.distribution
            log_prob = policy.log_prob(experiences["action"])[mask]
            loss = -advantage * log_prob
            entropy_bonus = None
            if entropy_coeff != 0.0:
                entropy_bonus = -entropy_coeff * policy.entropy()[mask]
                loss += entropy_bonus
            loss = loss.mean()
        optimize_result = self._optimize_policy_network(loss)
        result = {
            "advantage": advantage.mean().item(),
            "log_prob": log_prob.mean().item(),
            "loss": loss.item(),
        }
        if entropy_bonus is not None:
            result["entropy_bonus"] = entropy_bonus.mean().item()
        result.update(optimize_result)
        return result

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
        self._grad_scaler.step(self._policy_network_optimizer)
        result = {
            "lr": self._policy_network_optimizer.param_groups[0]["lr"],
            "grad_l2_norm": min(grad_l2_norm.item(), max_grad_l2_norm),
        }
        if self._policy_network_optimizer_lr_scheduler is not None:
            self._policy_network_optimizer_lr_scheduler.step()
        return result


class DistributedDataParallelREINFORCE(DistributedDataParallelAlgorithm):
    """Distributed data parallel REINFORCE.

    See Also
    --------
    actorch.algorithms.reinforce.REINFORCE

    """

    _ALGORITHM_CLS = REINFORCE  # override
