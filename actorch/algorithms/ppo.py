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

"""Proximal Policy Optimization."""

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from gymnasium import Env
from numpy import ndarray
from torch import Tensor
from torch.cuda.amp import autocast
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from actorch.agents import Agent
from actorch.algorithms.a2c import A2C, DistributedDataParallelA2C, Loss, LRScheduler
from actorch.algorithms.algorithm import RefOrFutureRef, Tunable
from actorch.algorithms.value_estimation import lambda_return
from actorch.envs import BatchedEnv
from actorch.models import Model
from actorch.networks import DistributionParametrization, NormalizingFlow, Processor
from actorch.samplers import Sampler
from actorch.schedules import ConstantSchedule, Schedule


__all__ = [
    "DistributedDataParallelPPO",
    "PPO",
]


class PPO(A2C):
    """Proximal Policy Optimization.

    References
    ----------
    .. [1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov.
           "Proximal Policy Optimization Algorithms".
           In: arXiv. 2017.
           URL: https://arxiv.org/abs/1707.06347

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
            trace_decay: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.95,
            num_epochs: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 10,
            minibatch_size: "Tunable[RefOrFutureRef[Union[int, Schedule]]]" = 64,
            ratio_clip: "Tunable[RefOrFutureRef[Union[float, Schedule]]]" = 0.2,
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
                trace_decay=trace_decay,
                num_epochs=num_epochs,
                minibatch_size=minibatch_size,
                ratio_clip=ratio_clip,
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
        self.config = PPO.Config(**self.config)
        self.config["_accept_kwargs"] = True
        super().setup(config)
        if not isinstance(self.trace_decay, Schedule):
            self.trace_decay = ConstantSchedule(self.trace_decay)
        if not isinstance(self.num_epochs, Schedule):
            self.num_epochs = ConstantSchedule(self.num_epochs)
        if not isinstance(self.minibatch_size, Schedule):
            self.minibatch_size = ConstantSchedule(self.minibatch_size)
        if not isinstance(self.ratio_clip, Schedule):
            self.ratio_clip = ConstantSchedule(self.ratio_clip)

    # override
    @property
    def _checkpoint(self) -> "Dict[str, Any]":
        checkpoint = super()._checkpoint
        checkpoint["trace_decay"] = self.trace_decay.state_dict()
        checkpoint["num_epochs"] = self.num_epochs.state_dict()
        checkpoint["minibatch_size"] = self.minibatch_size.state_dict()
        checkpoint["ratio_clip"] = self.ratio_clip.state_dict()
        return checkpoint

    # override
    @_checkpoint.setter
    def _checkpoint(self, value: "Dict[str, Any]") -> "None":
        super()._checkpoint = value
        self.trace_decay.load_state_dict(value["trace_decay"])
        self.num_epochs.load_state_dict(value["num_epochs"])
        self.minibatch_size.load_state_dict(value["minibatch_size"])
        self.ratio_clip.load_state_dict(value["ratio_clip"])

    # override
    def _train_step(self) -> "Dict[str, Any]":
        result = super()._train_step()
        self.trace_decay.step()
        self.num_epochs.step()
        self.minibatch_size.step()
        self.ratio_clip.step()
        result["trace_decay"] = self.trace_decay()
        result["num_epochs"] = self.num_epochs()
        result["minibatch_size"] = self.minibatch_size()
        result["ratio_clip"] = self.ratio_clip()
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
        minibatch_size = self.minibatch_size()
        if minibatch_size < 1 or not float(minibatch_size).is_integer():
            raise ValueError(
                f"`minibatch_size` ({minibatch_size}) must be in the integer interval [1, inf)"
            )
        minibatch_size = int(minibatch_size)
        num_epochs = self.num_epochs()
        if num_epochs < 1 or not float(num_epochs).is_integer():
            raise ValueError(
                f"`num_epochs` ({num_epochs}) must be in the integer interval [1, inf)"
            )
        num_epochs = int(num_epochs)

        result = {}

        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            with torch.no_grad():
                state_values, _ = self._value_network(
                    experiences["observation"], mask=mask
                )
            # Discard next observation
            experiences["observation"] = experiences["observation"][:, :-1, ...]
            mask = mask[:, 1:]
            targets, advantages = lambda_return(
                state_values,
                experiences["reward"],
                experiences["terminal"],
                mask,
                self.discount(),
                self.trace_decay(),
            )

        # Iterate over minibatches
        args = [targets, advantages, mask, *experiences.values()]
        for i, v in enumerate(args):
            args[i] = v.movedim(0, 1)
        dataset = TensorDataset(*args)
        dataloader = DataLoader(dataset, batch_size=minibatch_size)
        priority = None
        for _ in range(num_epochs):
            for batch in dataloader:
                for i, v in enumerate(batch):
                    batch[i] = v.movedim(0, 1)
                targets_batch, advantages_batch, mask_batch = batch[:3]
                experiences_batch = {k: v for k, v in zip(experiences, batch[3:])}

                with (
                    autocast(**self.enable_amp)
                    if self.enable_amp["enabled"]
                    else contextlib.suppress()
                ):
                    if self.normalize_advantage:
                        length_batch = mask_batch.sum(dim=1, keepdim=True)
                        advantages_batch_mean = (
                            advantages_batch.sum(dim=1, keepdim=True) / length_batch
                        )
                        advantages_batch -= advantages_batch_mean
                        advantages_batch *= mask_batch
                        advantages_batch_stddev = (
                            (
                                (advantages_batch**2).sum(dim=1, keepdim=True)
                                / length_batch
                            )
                            .sqrt()
                            .clamp(min=1e-6)
                        )
                        advantages_batch /= advantages_batch_stddev
                        advantages_batch *= mask_batch
                    state_values_batch, _ = self._value_network(
                        experiences_batch["observation"], mask=mask_batch
                    )

                result["value_network"], priority = self._train_on_batch_value_network(
                    state_values_batch,
                    targets_batch,
                    is_weight,
                    mask_batch,
                )
                result["policy_network"] = self._train_on_batch_policy_network(
                    experiences_batch,
                    advantages_batch,
                    mask_batch,
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
        ratio_clip = self.ratio_clip()
        if ratio_clip < 0.0:
            raise ValueError(
                f"`ratio_clip` ({ratio_clip}) must be in the interval [0, inf)"
            )
        with (
            autocast(**self.enable_amp)
            if self.enable_amp["enabled"]
            else contextlib.suppress()
        ):
            advantage = advantages[mask]
            old_log_prob = experiences["log_prob"][mask]
            self._policy_network(experiences["observation"], mask=mask)
            policy = self._policy_network.distribution
            log_prob = policy.log_prob(experiences["action"])[mask]
            ratio = (log_prob - old_log_prob).exp()
            surrogate_loss = advantage * ratio
            clipped_surrogate_loss = advantage * ratio.clamp(
                1.0 - ratio_clip, 1.0 + ratio_clip
            )
            loss = -surrogate_loss.min(clipped_surrogate_loss)
            entropy_bonus = None
            if entropy_coeff != 0.0:
                entropy_bonus = -entropy_coeff * policy.entropy()[mask]
                loss += entropy_bonus
            loss = loss.mean()
        optimize_result = self._optimize_policy_network(loss)
        result = {
            "advantage": advantage.mean().item(),
            "log_prob": log_prob.mean().item(),
            "old_log_prob": old_log_prob.mean().item(),
            "loss": loss.item(),
        }
        if entropy_bonus is not None:
            result["entropy_bonus"] = entropy_bonus.mean().item()
        result.update(optimize_result)
        return result


class DistributedDataParallelPPO(DistributedDataParallelA2C):
    """Distributed data parallel Proximal Policy Optimization.

    See Also
    --------
    actorch.algorithms.ppo.PPO

    """

    _ALGORITHM_CLS = PPO  # override
