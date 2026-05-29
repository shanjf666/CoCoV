# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Type

import numpy as np
import ray
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    PASS_GRPO = "pass_grpo"
    PASS_GRPO_PENALIZED = "pass_grpo_penalized"

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch

from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    diversity_density_config: dict = None,
):
    """
    Compute advantage for different estimators.
    
    Args:
        data: DataProto containing batch data
        adv_estimator: AdvantageEstimator enum value
        gamma: Discount factor for GAE/REINFORCE++
        lam: Lambda for GAE
        num_repeat: Number of repeats
        diversity_density_config: Dict containing PASS_GRPO specific configs
    
    Returns:
        data: DataProto with advantages and returns added
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.PASS_GRPO:
        # Pass@k reweighted GRPO advantage
        if diversity_density_config is None:
            diversity_density_config = {}
            
        k = diversity_density_config.get("k", 4)
        
        # Need answer_types from non_tensor_batch
        if "answer_types" not in data.non_tensor_batch:
            raise ValueError("PASS_GRPO requires 'answer_types' in data.non_tensor_batch")
        
        advantages, returns = core_algos.compute_pass_grpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            answer_types=data.non_tensor_batch["answer_types"],
            k=k,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.PASS_GRPO_PENALIZED:
        if diversity_density_config is None:
            diversity_density_config = {}
            
        k = diversity_density_config.get("k", 4)
        epsilon = diversity_density_config.get("epsilon", 1e-6)
        
        if "answer_types" not in data.non_tensor_batch:
            raise ValueError("PASS_GRPO_PENALIZED requires 'answer_types' in data.non_tensor_batch")
        
        consistency_rates = data.non_tensor_batch.get("consistency_rate", None)
        if consistency_rates is None:
            raise ValueError("PASS_GRPO_PENALIZED requires 'consistency_rate' in data.non_tensor_batch")
            
        advantages, returns, metrics = core_algos.compute_pass_grpo_penalized_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            answer_types=data.non_tensor_batch["answer_types"],
            consistency_rates=consistency_rates,
            diversity_density_config=diversity_density_config,
            k=k,
            epsilon=epsilon
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
        # Log scalar metrics to meta_info
        for k_met, v_met in metrics.items():
            if isinstance(v_met, (int, float)):
                data.meta_info[k_met] = v_met
            elif isinstance(v_met, torch.Tensor) and v_met.numel() == 1:
                data.meta_info[k_met] = float(v_met.item())
        
        del metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        data.meta_info["pass_grpo_penalized/avg_total_advantage"] = advantages.mean().item()
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.PASS_GRPO,
            AdvantageEstimator.PASS_GRPO_PENALIZED,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        if self.config.reward_model.reward_manager in ["ttrl", "truelabel_ttrl"]:
            self.use_ttrl = True
            self.n_samples_per_prompt = self.config.reward_model.reward_kwargs.n_samples_per_prompt
            self.n_votes_per_prompt = self.config.reward_model.reward_kwargs.n_votes_per_prompt
        else:
            self.use_ttrl = False

        # Two-stage self-verification configuration
        self.two_stage_verify = getattr(self.config, 'two_stage_verify', False)
        if self.two_stage_verify:
            self.two_stage_mode = getattr(self.config, 'two_stage_mode', 'greedy')  # 'greedy' or 'sampling'
            self.two_stage_n = getattr(self.config, 'two_stage_n', 4)  # N for sampling mode
            self.two_stage_hc_max_candidates = getattr(self.config, 'two_stage_hc_max_candidates', 3)
            self.two_stage_lc_max_candidates = getattr(self.config, 'two_stage_lc_max_candidates', 5)
            self.two_stage_max_new_tokens = getattr(self.config, 'two_stage_max_new_tokens', 2048)
            self.two_stage_fallback = getattr(self.config, 'two_stage_fallback', 'majority')  # 'majority' or 'penalize'
            self.two_stage_fallback_mode = getattr(self.config, 'two_stage_fallback_mode', 'no_update_second')  # 'no_update_second' or 'no_update_both'
            self.two_stage_micro_batch_size = getattr(self.config, 'two_stage_micro_batch_size', 0)  # 0 = auto
            self.two_stage_top_p = getattr(self.config, 'two_stage_top_p', 0.85)

            # Dynamic temperature settings
            self.two_stage_hc_temperature = getattr(self.config, 'two_stage_hc_temperature', 1.0)
            self.two_stage_lc_temperature = getattr(self.config, 'two_stage_lc_temperature', 0.6)
            self.two_stage_high_consistency_threshold = getattr(
                self.config,
                'two_stage_high_consistency_threshold',
                0.6,
            )
            self.two_stage_low_consistency_threshold = getattr(
                self.config,
                'two_stage_low_consistency_threshold',
                0.4,
            )
            # Backward-compatible alias for external code paths that still read the old key.
            self.two_stage_candidate_pad_seed = getattr(self.config, 'two_stage_candidate_pad_seed', None)
            self.two_stage_high_consistency_topk_padding = getattr(
                self.config,
                'two_stage_high_consistency_topk_padding',
                True,
            )

            if self.two_stage_low_consistency_threshold > self.two_stage_high_consistency_threshold:
                print(
                    "[TwoStage] Warning: two_stage_low_consistency_threshold is greater than "
                    "two_stage_high_consistency_threshold. Clamping low threshold to high threshold."
                )
                self.two_stage_low_consistency_threshold = self.two_stage_high_consistency_threshold

            print(f"[TwoStage] Enabled: mode={self.two_stage_mode}, n={self.two_stage_n}, "
                  f"hc_max_candidates={self.two_stage_hc_max_candidates}, lc_max_candidates={self.two_stage_lc_max_candidates}, max_new_tokens={self.two_stage_max_new_tokens}, "
                  f"hc_temp={self.two_stage_hc_temperature}, lc_temp={self.two_stage_lc_temperature}, "
                  f"hc_threshold={self.two_stage_high_consistency_threshold}, "
                f"lc_threshold={self.two_stage_low_consistency_threshold}, top_p={self.two_stage_top_p}, "
                f"hc_topk_padding={self.two_stage_high_consistency_topk_padding}")

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            if (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
            ):
                assert config.actor_rollout_ref.model.use_remove_padding, (
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, seed=1):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type

        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(
                    f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                    f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset"
                )
        else:
            dataset_cls = RLHFDataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
            suffix_prompt=self.config.data.suffix_prompt,
        )

        # print the first example with all keys and values line by line
        for key, value in self.train_dataset[0].items():
            print(f"{key}: {value}")

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", seed))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
            suffix_prompt=self.config.data.suffix_prompt,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) == 1, (
            "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."
        )

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # add ttrl metrics
        if self.use_ttrl and "ttrl_info" in result:
            for key, val in result["ttrl_info"].items():
                metric_dict[f"val-ttrl/{key}"] = val

        # Clean up memory explicitly to prevent OOM in subsequent generation
        import gc
        del test_batch, test_gen_batch, test_gen_batch_padded, test_output_gen_batch
        del input_ids, output_ids
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _select_top_k_per_prompt(self, data, n_votes_per_prompt, n_samples_per_prompt):
        assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
        num_prompts = len(data) // n_votes_per_prompt

        selected_indices = []
        for i in range(num_prompts):
            start = i * n_votes_per_prompt
            selected_indices.extend(range(start, start + n_samples_per_prompt))

        return data[selected_indices]


    def _run_verification_inference(self, verification_batch, n_samples, target_chunk_tokens):
        """Runs the micro-batched generation for a verification DataProto.
        
        Returns:
            outputs_by_local_idx: dict mapping local row index in verification_batch -> list of n_samples responses
            all_chunk_outputs: list of DataProto outputs for concatenated batch_second
            info_for_reorder: list of (local_idx, chunk_offset) mappings to help reassembly
        """
        import gc
        from verl.utils.seqlen_balancing import rearrange_micro_batches
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.utils.reward_score.ttrl.two_stage_utils import decode_verification_outputs

        if len(verification_batch) == 0:
            return {}, [], []

        micro_batches, micro_bsz_idx = rearrange_micro_batches(
            batch=verification_batch.batch,
            max_token_len=target_chunk_tokens,
            dp_group=None
        )

        outputs_by_local_idx = {}
        all_chunk_outputs = []
        info_for_reorder = []
        current_unordered_offset = 0

        for chunk_idx, (micro_batch_tensors, indices) in enumerate(zip(micro_batches, micro_bsz_idx)):
            nt_batch = None
            if verification_batch.non_tensor_batch is not None:
                nt_batch = {}
                for k, v in verification_batch.non_tensor_batch.items():
                    nt_batch[k] = v[indices]
            
            chunk = DataProto(batch=micro_batch_tensors, non_tensor_batch=nt_batch, meta_info=verification_batch.meta_info.copy())
            chunk_padded, pad_size = pad_dataproto_to_divisor(chunk, self.actor_rollout_wg.world_size)

            try:
                chunk_output_padded = self.actor_rollout_wg.generate_sequences(chunk_padded)
                chunk_output = unpad_dataproto(chunk_output_padded, pad_size=pad_size)
                chunk_decoded = decode_verification_outputs(chunk_output, self.tokenizer)
                
                for i, local_idx in enumerate(indices):
                    resp_start = i * n_samples
                    resp_end = resp_start + n_samples
                    outputs_by_local_idx[local_idx] = chunk_decoded[resp_start:resp_end]
                    
                    # Store information needed for global reorder_indices
                    info_for_reorder.append((local_idx, current_unordered_offset + resp_start))
                
                current_unordered_offset += len(indices) * n_samples
                all_chunk_outputs.append(chunk_output.to("cpu"))
            except Exception as e:
                print(f"[TwoStage] Error during verification chunk {chunk_idx}: {e}")
                for local_idx in indices:
                    outputs_by_local_idx[local_idx] = [""] * n_samples
            finally:
                if 'chunk_output_padded' in locals(): del chunk_output_padded
                if 'chunk_output' in locals(): del chunk_output
                if 'chunk_padded' in locals(): del chunk_padded
        
        return outputs_by_local_idx, all_chunk_outputs, info_for_reorder

    def _log_high_consistency_advantage_view_metrics(
        self,
        verify_mapping: list[dict],
        groups_to_verify: list[dict],
        verified_routes: list[str],
        metrics: dict,
    ):
        """Log Stage2 high-consistency view metrics without resampling.

        This mirrors the sample-view observability metrics while keeping the
        original full-data training behavior unchanged.
        """
        metrics["train/two_stage_advantage_sampling_enabled"] = 0.0

        original_rows = len(verify_mapping) if verify_mapping is not None else 0
        affected_group_count = 0
        majority_sample_count = 0
        non_majority_original_count = 0

        if (
            verify_mapping
            and groups_to_verify
            and verified_routes
        ):
            route_by_group = {}
            majority_by_group = {}
            for i, group in enumerate(groups_to_verify):
                group_idx = group["prompt_group_idx"]
                if i < len(verified_routes):
                    route_by_group[group_idx] = verified_routes[i]
                majority_by_group[group_idx] = str(group.get("majority_answer", ""))

            candidate_rows = defaultdict(lambda: defaultdict(int))
            for mapping in verify_mapping:
                group_idx = mapping["prompt_group_idx"]
                cand = str(mapping["candidate_answer"])
                candidate_rows[group_idx][cand] += 1

            for group_idx, cand_counter in candidate_rows.items():
                if route_by_group.get(group_idx, "") != "A":
                    continue

                affected_group_count += 1
                majority_answer = majority_by_group.get(group_idx, "")
                for cand, count in cand_counter.items():
                    if cand == majority_answer:
                        majority_sample_count += count
                    else:
                        non_majority_original_count += count

        non_majority_sample_count = non_majority_original_count
        sampled_rows = original_rows

        # metrics["train/two_stage_high_consistency_advantage_group_count"] = float(affected_group_count)
        # metrics["train/two_stage_high_consistency_advantage_sample_target"] = 0.0
        # metrics["train/two_stage_high_consistency_majority_sample_count"] = float(majority_sample_count)
        # metrics["train/two_stage_high_consistency_non_majority_original_count"] = float(non_majority_original_count)
        # metrics["train/two_stage_high_consistency_non_majority_sample_count"] = float(non_majority_sample_count)
        # metrics["train/two_stage_advantage_sample_with_replacement_count"] = 0.0
        # metrics["train/two_stage_advantage_sample_expansion_ratio"] = (
        #     float(sampled_rows) / float(max(1, original_rows))
        # )
        # metrics["train/two_stage_advantage_sample_with_replacement_ratio"] = 0.0

    def _run_two_stage_verification(self, batch: DataProto, metrics: dict) -> list:
        """Run two-stage self-verification on the current batch.

        This method:
        1. Extracts candidate answers from the Pass 1 rollout data
        2. Constructs verification prompts for each candidate
        3. Runs micro-batched verification inference via generate_sequences
        4. Parses verification results and resolves pseudo-labels

        Args:
            batch: The DataProto containing Pass 1 rollout data (already repeated & union'd).
            metrics: Dict to log verification metrics into.

        Returns:
            Tuple of:
                - List of pseudo-labels, one per prompt group
                - batch_second: DataProto containing all verification trajectories for Stage 2 training (or None)
                - all_verification_outputs: List of decoded verification response strings
                - verification_mapping: List mapping each output back to its prompt_group_idx and candidate
                - groups_to_verify: List of prompt_groups that triggered verification
                - pl_consistencies: List of float consistencies for the chosen pseudo-labels
        """
        import gc
        from verl.utils.reward_score.ttrl.two_stage_utils import (
            extract_candidate_answers,
            classify_consistency_bucket,
            construct_verification_dataproto,
            decode_verification_outputs,
            # pad_high_consistency_candidates_to_topk,
            select_final_pseudo_labels,
        )
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        # Determine the task type from the first sample
        first_item = batch[0]
        data_source = first_item.non_tensor_batch.get("data_source", "math")
        ds = str(data_source).lower()
        if any(k in ds for k in ["gpqa"]):
            task = "gpqa"
        else:
            task = "math"

        print(f"[TwoStage] Starting verification: task={task}, mode={self.two_stage_mode}")

        # Determine base extraction limit
        extract_max = 0
        if self.two_stage_hc_max_candidates > 0 and self.two_stage_lc_max_candidates > 0:
            extract_max = max(self.two_stage_hc_max_candidates, self.two_stage_lc_max_candidates)

        # Step 1: Extract candidate answers from Pass 1
        prompt_groups = extract_candidate_answers(
            pass1_data=batch,
            tokenizer=self.tokenizer,
            n_votes_per_prompt=self.n_votes_per_prompt,
            task=task,
            max_candidates=extract_max,
        )

        # Step 1.5: Pre-compute routing and truncate candidates based on consistency
        group_consistency_routes = {}
        high_group_count = 0
        low_group_count = 0
        middle_group_count = 0
        for g in prompt_groups:
            group_idx = g["prompt_group_idx"]
            route_bucket = classify_consistency_bucket(
                majority_rate=g.get("majority_rate", 0.0),
                high_consistency_threshold=self.two_stage_high_consistency_threshold,
                low_consistency_threshold=self.two_stage_low_consistency_threshold,
            )
            group_consistency_routes[group_idx] = route_bucket
            if route_bucket == "high":
                high_group_count += 1
                if self.two_stage_hc_max_candidates > 0:
                    g["candidates"] = g["candidates"][:self.two_stage_hc_max_candidates]
            else:
                if route_bucket == "low":
                    low_group_count += 1
                else:
                    middle_group_count += 1
                if self.two_stage_lc_max_candidates > 0:
                    g["candidates"] = g["candidates"][:self.two_stage_lc_max_candidates]

        pad_seed = self.two_stage_candidate_pad_seed
        if pad_seed is not None:
            pad_seed = int(pad_seed) + int(getattr(self, "global_steps", 0))
        # if self.two_stage_high_consistency_topk_padding:
        #     pad_stats = pad_high_consistency_candidates_to_topk(
        #         prompt_groups=prompt_groups,
        #         max_candidates=self.two_stage_hc_max_candidates,
        #         high_consistency_threshold=self.two_stage_high_consistency_threshold,
        #         low_consistency_threshold=self.two_stage_low_consistency_threshold,
        #         seed=pad_seed,
        #     )
        # else:
        pad_stats = {
            "high_consistency_group_count": high_group_count,
            # "padded_group_count": 0,
            # "padded_candidate_count": 0,
        }

        groups_to_verify = prompt_groups

        # metrics["train/two_stage_hc_topk_padding_enabled"] = (
        #     1.0 if self.two_stage_high_consistency_topk_padding else 0.0
        # )
        metrics["train/two_stage_trigger_rate"] = 1.0 if prompt_groups else 0.0
        metrics["train/two_stage_hc_group_count"] = float(pad_stats["high_consistency_group_count"])
        # metrics["train/two_stage_hc_padded_group_count"] = float(pad_stats["padded_group_count"])
        # metrics["train/two_stage_hc_padded_candidate_count"] = float(pad_stats["padded_candidate_count"])
        avg_majority_rate = sum(g.get("majority_rate", 0.0) for g in prompt_groups) / len(prompt_groups) if prompt_groups else 0.0
        metrics["train/two_stage_majority_rate_mean"] = avg_majority_rate

        total_candidates = sum(len(g["candidates"]) for g in groups_to_verify)
        print(f"[TwoStage] Triggering Verification Batch: {high_group_count + low_group_count}/{len(prompt_groups)} groups (A={high_group_count}, M={middle_group_count}[Skip], B={low_group_count})")
        metrics["train/two_stage_total_candidates"] = total_candidates

        # Step 2: Construct verification DataProto
        verification_batch, verification_mapping = construct_verification_dataproto(
            prompt_groups=groups_to_verify,
            tokenizer=self.tokenizer,
            verification_mode=self.two_stage_mode,
            verification_n=self.two_stage_n,  # Pass the config value (use -1 for dynamic frequency)
            max_prompt_length=self.config.data.max_prompt_length * 2,  # verification prompts are longer
        )

        if verification_batch is None or len(verification_batch) == 0:
            print("[TwoStage] Empty verification batch. Skipping.")
            return [g.get("majority_answer") for g in prompt_groups], None, [], [], [], [], []

        print(f"[TwoStage] Verification batch size: {len(verification_batch)}")

        # Step 3: Set common generation config
        verification_batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        verification_batch.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        verification_batch.meta_info["recompute_log_prob"] = False
        verification_batch.meta_info["do_vote"] = False
        verification_batch.meta_info["verification_mode"] = self.two_stage_mode
        verification_batch.meta_info["verification_max_new_tokens"] = self.two_stage_max_new_tokens

        total_verification_size = len(verification_batch)
        verification_n = verification_batch.meta_info.get("verification_n", 1)
        torch.cuda.empty_cache()

        # verification_mapping is sample-level when n_samples > 1, but batch slicing
        # must operate on row-level indices.
        row_verification_mapping = verification_mapping[::verification_n] if verification_n > 1 else verification_mapping
        if len(row_verification_mapping) != total_verification_size:
            raise ValueError(
                "Verification mapping size mismatch: "
                f"{len(row_verification_mapping)} row mappings for {total_verification_size} verification rows"
            )

        metrics["train/two_stage_hc_group_route_count"] = float(high_group_count)
        metrics["train/two_stage_lc_group_route_count"] = float(low_group_count)
        metrics["train/two_stage_mid_group_route_count"] = float(middle_group_count)

        # Step 4: Split batch by consistency and run inference
        # Identify HC vs LC indices
        hc_indices = []
        lc_indices = []
        mid_indices = []
        for i, mapping in enumerate(row_verification_mapping):
            g_idx = mapping["prompt_group_idx"]
            route_bucket = group_consistency_routes.get(g_idx, "low")
            if route_bucket == "high":
                hc_indices.append(i)
            elif route_bucket == "middle":
                mid_indices.append(i)
            else:
                lc_indices.append(i)
        
        metrics["train/two_stage_hc_verify_count"] = len(hc_indices)
        metrics["train/two_stage_lc_verify_count"] = len(lc_indices)
        metrics["train/two_stage_mid_count"] = len(mid_indices)
        # metrics["train/two_stage_hc_temperature"] = self.two_stage_hc_temperature
        # metrics["train/two_stage_lc_temperature"] = self.two_stage_lc_temperature

        # Calculate target chunk token capacity
        max_prompt_len = self.config.data.max_prompt_length * 2
        max_token_len = max_prompt_len + self.two_stage_max_new_tokens
        micro_bs = self.two_stage_micro_batch_size if self.two_stage_micro_batch_size > 0 else max(4, self.config.data.train_batch_size)
        target_chunk_tokens = max(20480, min(int(micro_bs * max_token_len * 1.2), 131072))

        # Run HC Inference (T=hc_temp)
        hc_outputs = {}
        hc_chunks = []
        hc_reorder_info = []
        if hc_indices:
            print(f"[TwoStage] Running HC verification ({len(hc_indices)} samples) at T={self.two_stage_hc_temperature}")
            hc_batch = verification_batch[hc_indices]
            hc_batch.meta_info["do_sample"] = (self.two_stage_hc_temperature > 0)
            hc_batch.meta_info["verification_temperature"] = self.two_stage_hc_temperature
            hc_batch.meta_info["verification_top_p"] = self.two_stage_top_p
            hc_outputs, hc_chunks, hc_reorder_info = self._run_verification_inference(hc_batch, verification_n, target_chunk_tokens)

        # Run LC Inference (T=lc_temp)
        lc_outputs = {}
        lc_chunks = []
        lc_reorder_info = []
        if lc_indices:
            print(f"[TwoStage] Running LC verification ({len(lc_indices)} samples) at T={self.two_stage_lc_temperature}")
            lc_batch = verification_batch[lc_indices]
            lc_batch.meta_info["do_sample"] = (self.two_stage_lc_temperature > 0)
            lc_batch.meta_info["verification_temperature"] = self.two_stage_lc_temperature
            lc_batch.meta_info["verification_top_p"] = self.two_stage_top_p
            lc_outputs, lc_chunks, lc_reorder_info = self._run_verification_inference(lc_batch, verification_n, target_chunk_tokens)

        # Step 5: Merge Results
        all_verification_outputs = [None] * (total_verification_size * verification_n)
        for local_idx, resps in hc_outputs.items():
            orig_idx = hc_indices[local_idx]
            all_verification_outputs[orig_idx * verification_n : (orig_idx + 1) * verification_n] = resps
        for local_idx, resps in lc_outputs.items():
            orig_idx = lc_indices[local_idx]
            all_verification_outputs[orig_idx * verification_n : (orig_idx + 1) * verification_n] = resps
        
        # Merge batch_second
        all_chunk_outputs = hc_chunks + lc_chunks
        if all_chunk_outputs:
            batch_second_unordered = DataProto.concat(all_chunk_outputs)
            reorder_indices = [0] * (total_verification_size * verification_n)
            
            # Reorder logic: first part of batch_second_unordered is from hc_chunks
            for local_idx, unordered_offset in hc_reorder_info:
                orig_idx = hc_indices[local_idx]
                for j in range(verification_n):
                    reorder_indices[orig_idx * verification_n + j] = unordered_offset + j
            
            # Second part is from lc_chunks
            hc_total_unordered = len(hc_indices) * verification_n
            for local_idx, unordered_offset in lc_reorder_info:
                orig_idx = lc_indices[local_idx]
                for j in range(verification_n):
                    reorder_indices[orig_idx * verification_n + j] = hc_total_unordered + unordered_offset + j
            
            # Verify if reorder_indices is complete (on total samples including native n-samples)
            has_error = (len(hc_reorder_info) + len(lc_reorder_info)) < len(hc_indices) + len(lc_indices)
            if not has_error:
                batch_second = batch_second_unordered[reorder_indices]
                
                # Setup UIDs for GRPO grouping.
                # All samples for the same prompt share a UID so second-stage
                # normalization happens at the prompt-group level.
                uids = []
                for m in verification_mapping:
                    uid = f"verify_prompt_{m['prompt_group_idx']}"
                    uids.append(uid)
                batch_second.non_tensor_batch["uid"] = np.array(uids, dtype=object)
            else:
                print("[TwoStage] Warning: Chunks dropped. Discarding batch_second.")
                batch_second = None
        else:
            batch_second = None

        # Clean up lists of None if any
        all_verification_outputs = [out if out is not None else "" for out in all_verification_outputs]
        print(f"[TwoStage] Final decoded {len(all_verification_outputs)} verification outputs")

        # Step 5: Resolve pseudo-labels
        verified_labels, verified_consistencies, should_update_flags, verified_routes = select_final_pseudo_labels(
            verification_outputs=all_verification_outputs,
            verification_mapping=verification_mapping,
            prompt_groups=groups_to_verify,
            n_votes_per_prompt=self.n_votes_per_prompt,
            high_consistency_threshold=self.two_stage_high_consistency_threshold,
            low_consistency_threshold=self.two_stage_low_consistency_threshold,
            consistency_route_by_group=group_consistency_routes,
            low_consistency_strategy="majority",
            fallback_mode=self.two_stage_fallback_mode,
            verification_total_n=self.two_stage_n,
        )

        from verl.utils.reward_score.ttrl.two_stage_utils import parse_verification_result

        # Filter batch_second and verification data for Stage2 training
        if batch_second is not None and len(batch_second) > 0:
            keep_indices = []
            for i, m in enumerate(verification_mapping):
                g_idx = m["prompt_group_idx"]
                if should_update_flags[g_idx]:
                    keep_indices.append(i)

            if keep_indices:
                batch_second = batch_second[keep_indices]
                verification_mapping = [verification_mapping[i] for i in keep_indices]
                all_verification_outputs = [all_verification_outputs[i] for i in keep_indices]
            else:
                batch_second = None
                verification_mapping = []
                all_verification_outputs = []
                print("[TwoStage] All samples skipped Stage2 training due to no valid low-consistency candidates")

        # Attach per-sample high-consistency route mask for Stage2 loss diagnostics.
        if batch_second is not None and len(batch_second) > 0 and len(verification_mapping) > 0:
            stage2_high_consistency_mask = [
                1.0 if verified_routes[m["prompt_group_idx"]] == "A" else 0.0
                for m in verification_mapping
            ]
            batch_second.batch["two_stage_high_consistency_mask"] = torch.tensor(
                stage2_high_consistency_mask,
                dtype=torch.float32,
            )

        final_pseudo_labels = verified_labels
        final_consistencies = verified_consistencies

        # Count how many samples were filtered to a different pseudo-label
        success_filter_count = 0
        flip_correct = 0
        flip_incorrect = 0
        route_b_count = 0
        route_b_flip_count = 0

        route_m_count = 0
        route_m_correct = 0
        route_a_count = 0
        route_a_correct = 0
        route_low_count = 0
        route_low_majority_correct = 0
        route_low_two_stage_correct = 0
        route_b1_count = 0
        route_b1_correct = 0

        if len(groups_to_verify) > 0:
            from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import math_equal, extract_answer
            
            for i, g in enumerate(groups_to_verify):
                y_pseudo = str(verified_labels[i])
                y_maj = str(g.get("majority_answer"))
                is_changed = not math_equal(y_pseudo, y_maj)
                route = verified_routes[i] if i < len(verified_routes) else ""
                gt_raw = g.get("ground_truth", "")
                has_gt = bool(gt_raw)
                gt_ext = ""
                if has_gt:
                    gt_str = str(gt_raw)
                    gt_ext = extract_answer(gt_str, data_name="math") or gt_str

                is_pseudo_correct = math_equal(y_pseudo, gt_ext) if has_gt else False
                is_maj_correct = math_equal(y_maj, gt_ext) if has_gt else False

                if route == "A":
                    route_a_count += 1
                    if is_maj_correct:
                        route_a_correct += 1

                if route == "M":
                    route_m_count += 1
                    if is_maj_correct:
                        route_m_correct += 1

                if route in ("B1", "B2"):
                    route_low_count += 1
                    if is_maj_correct:
                        route_low_majority_correct += 1
                    if is_pseudo_correct:
                        route_low_two_stage_correct += 1

                if route == "B1":
                    route_b1_count += 1
                    if is_pseudo_correct:
                        route_b1_correct += 1

                if is_changed:
                    success_filter_count += 1
                
                # Check Route B flip logic for new metrics
                if route in ("B1", "B2"):
                    route_b_count += 1
                    if is_changed:
                        route_b_flip_count += 1
                    if is_changed:
                        if has_gt:
                            if is_pseudo_correct and not is_maj_correct:
                                flip_correct += 1
                            elif not is_pseudo_correct and is_maj_correct:
                                flip_incorrect += 1

            metrics["train/two_stage_high_consistency_label_accuracy"] = route_a_correct / max(1, route_a_count)
            metrics["train/two_stage_mid_consistency_label_accuracy"] = route_m_correct / max(1, route_m_count)
            metrics["train/two_stage_low_consistency_first_stage_label_accuracy"] = (
                route_low_majority_correct / max(1, route_low_count)
            )
            metrics["train/two_stage_low_consistency_two_stage_label_accuracy"] = (
                route_low_two_stage_correct / max(1, route_low_count)
            )
            metrics["train/two_stage_low_consistency_true_set_label_accuracy"] = (
                route_b1_correct / max(1, route_b1_count)
            )
            metrics["train/two_stage_low_consistency_updateable_group_count"] = float(route_b1_count)
            metrics["train/two_stage_low_consistency_true_set_coverage"] = (
                route_b1_count / max(1, route_low_count)
            )
            
            route_updated_count = route_a_count + route_m_count + route_b1_count
            route_updated_correct = route_a_correct + route_m_correct + route_b1_correct
            metrics["train/two_stage_updated_label_accuracy"] = route_updated_correct / max(1, route_updated_count)
            metrics["train/two_stage_updated_group_count"] = float(route_updated_count)

            
            route_b2_count = sum(1 for r in verified_routes if r == "B2")
            metrics["train/two_stage_noise_mask_rate"] = route_b2_count / len(groups_to_verify)
        else:
            metrics["train/two_stage_high_consistency_label_accuracy"] = 0.0
            metrics["train/two_stage_mid_consistency_label_accuracy"] = 0.0
            metrics["train/two_stage_low_consistency_first_stage_label_accuracy"] = 0.0
            metrics["train/two_stage_low_consistency_two_stage_label_accuracy"] = 0.0
            metrics["train/two_stage_low_consistency_true_set_label_accuracy"] = 0.0
            metrics["train/two_stage_low_consistency_updateable_group_count"] = 0.0
            metrics["train/two_stage_low_consistency_true_set_coverage"] = 0.0
            metrics["train/two_stage_updated_label_accuracy"] = 0.0
            metrics["train/two_stage_updated_group_count"] = 0.0

            metrics["train/two_stage_noise_mask_rate"] = 0.0

        # Log verification-specific metrics
        true_count = sum(1 for out in all_verification_outputs if parse_verification_result(out) is True)
        false_count = sum(1 for out in all_verification_outputs if parse_verification_result(out) is False)
        none_count = sum(1 for out in all_verification_outputs if parse_verification_result(out) is None)

        print(f"[TwoStage] Results: True={true_count}, False={false_count}, ParseFail={none_count}")
        print(f"[TwoStage] Successfully filtered to new candidate in {success_filter_count}/{max(1, route_low_count)} triggered Low groups")

        # Cleanup
        del verification_batch
        gc.collect()
        torch.cuda.empty_cache()

        return final_pseudo_labels, batch_second, all_verification_outputs, verification_mapping, groups_to_verify, final_consistencies, verified_routes

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.meta_info["do_vote"] = False
                if self.use_ttrl:
                    self.config.actor_rollout_ref.rollout.n = self.n_votes_per_prompt
                    batch.meta_info["do_vote"] = True

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                        meta_info_keys=["do_vote"]
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                        meta_info_keys=["do_vote"]
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        if self.use_ttrl:
                            assert len(gen_batch_output) == len(batch) * self.n_votes_per_prompt
                        else:
                            pass
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # ================================================================
                    # Pass 2: Two-Stage Self-Verification
                    
                    # Initialize GT metrics to 0 in case Verification is not triggered
                    metrics.update({
                        "train/gt_tp_rate": 0.0,
                        "train/gt_tn_rate": 0.0,
                        "train/gt_fp_rate": 0.0,
                        "train/gt_fn_rate": 0.0,
                        "train/test_minority_zeroed_ratio": 0.0,
                        "train/test_minority_low_consistency_count": 0.0,
                        # "train/two_stage_hc_topk_padding_enabled": 0.0,
                        "train/two_stage_advantage_sampling_enabled": 0.0,
                        "train/two_stage_high_consistency_label_accuracy": 0.0,
                        "train/two_stage_mid_consistency_label_accuracy": 0.0,
                        "train/two_stage_low_consistency_first_stage_label_accuracy": 0.0,
                        "train/two_stage_low_consistency_two_stage_label_accuracy": 0.0,
                        "train/two_stage_low_consistency_true_set_label_accuracy": 0.0,
                        "train/two_stage_low_consistency_updateable_group_count": 0.0,
                        "train/two_stage_low_consistency_true_set_coverage": 0.0,
                        "train/two_stage_updated_label_accuracy": 0.0,
                        "train/two_stage_updated_group_count": 0.0,
                    })

                    batch_second = None
                    verified_routes = None
                    if self.use_ttrl and self.two_stage_verify:
                        with _timer("two_stage_verify", timing_raw):
                            verified_pseudo_labels, batch_second, verify_outputs, verify_mapping, groups_to_verify, pl_consistencies, verified_routes = self._run_two_stage_verification(
                                batch=batch,
                                metrics=metrics,
                            )

                            # Attach per-sample high-consistency route mask for Stage1 loss diagnostics.
                            if verified_routes is not None and len(verified_routes) > 0:
                                expanded_high_consistency_mask = []
                                for route in verified_routes:
                                    flag = 1.0 if route == "A" else 0.0
                                    expanded_high_consistency_mask.extend([flag] * self.n_votes_per_prompt)

                                if len(expanded_high_consistency_mask) == len(batch):
                                    batch.batch["two_stage_high_consistency_mask"] = torch.tensor(
                                        expanded_high_consistency_mask,
                                        dtype=torch.float32,
                                    )
                                else:
                                    print(
                                        "[TwoStage] Warning: high-consistency mask length mismatch "
                                        f"({len(expanded_high_consistency_mask)} vs {len(batch)}). Skip Stage1 mask injection."
                                    )
                            
                            # [no_update_both] Inject zero_advantage_mask for Route B2 groups
                            if self.two_stage_fallback_mode == "no_update_both" and verified_routes is not None:
                                skip_groups = [i for i, r in enumerate(verified_routes) if r == "B2"]
                                if skip_groups:
                                    mask = np.zeros(len(batch), dtype=np.float32)
                                    for g_idx in skip_groups:
                                        start = g_idx * self.n_votes_per_prompt
                                        end = start + self.n_votes_per_prompt
                                        mask[start:end] = 1.0
                                    batch.non_tensor_batch["zero_advantage_mask"] = mask
                                    print(f"[no_update_both] Marked {len(skip_groups)} Route B2 groups ({int(mask.sum())}/{len(mask)} samples) for Stage1 advantage zeroing")
                            if verified_pseudo_labels is not None and verified_routes is not None:
                                n_a = sum(1 for r in verified_routes if r == "A")
                                n_m = sum(1 for r in verified_routes if r == "M")
                                n_b1 = sum(1 for r in verified_routes if r == "B1")
                                n_b2 = sum(1 for r in verified_routes if r == "B2")
                                
                                n_updated = n_a + n_m + n_b1
                                n_total = len(verified_routes)
                                
                                expanded_labels = []
                                for label in verified_pseudo_labels:
                                    expanded_labels.extend([label] * self.n_votes_per_prompt)
                                batch.non_tensor_batch["verified_pseudo_label"] = np.array(
                                    expanded_labels, dtype=object
                                )
                                metrics["train/two_stage_verified_ratio"] = n_updated / n_total if n_total > 0 else 0.0
                                print(f"[TwoStage] Routing Breakdown: A(High)={n_a}, M(Mid)={n_m}, B1(Fix)={n_b1}, B2(Fail)={n_b2}")
                                print(f"[TwoStage] Verified {n_updated}/{n_total} prompt groups (Routes A+M+B1)")

                            self._log_high_consistency_advantage_view_metrics(
                                verify_mapping=verify_mapping,
                                groups_to_verify=groups_to_verify,
                                verified_routes=verified_routes,
                                metrics=metrics,
                            )

                            # ================================================================
                            # Stage 2 Preprocessing: compute proxy rewards and advantages
                            # ================================================================
                            if batch_second is not None and len(batch_second) > 0:
                                from verl.utils.reward_score.ttrl.two_stage_utils import compute_proxy_cm_reward
                                from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
                                
                                # 1. Compute proxy CM reward
                                final_pseudo_labels_dict = {
                                    g["prompt_group_idx"]: verified_pseudo_labels[g["prompt_group_idx"]]
                                    for g in groups_to_verify
                                }
                                consistency_scores = {
                                    g["prompt_group_idx"]: pl_consistencies[g["prompt_group_idx"]]
                                    for g in groups_to_verify
                                }
                                
                                # Calculate GT correctness for candidates
                                from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import math_equal, extract_answer
                                gt_correct_scores = []
                                for m in verify_mapping:
                                    g_idx = m["prompt_group_idx"]
                                    cand = m["candidate_answer"]
                                    gt_raw = ""
                                    for g in groups_to_verify:
                                        if g["prompt_group_idx"] == g_idx:
                                            gt_raw = g.get("ground_truth", "")
                                            break
                                    
                                    if not gt_raw:
                                        is_correct = False
                                    else:
                                        # Ensure both are strings
                                        gt_str = str(gt_raw)
                                        cand_str = str(cand)
                                        
                                        # Robust extraction for GT as well (it might be boxed)
                                        # We use "math" as the data_name for the parser
                                        gt_extracted = extract_answer(gt_str, data_name="math")
                                        if not gt_extracted:
                                            # Fallback to raw if extraction fails (might already be raw)
                                            gt_extracted = gt_str
                                        
                                        # Final comparison using math_equal
                                        is_correct = math_equal(cand_str, gt_extracted)
                                        
                                    gt_correct_scores.append(is_correct)

                                rewards, cm_metrics = compute_proxy_cm_reward(
                                    verify_outputs, verify_mapping, final_pseudo_labels_dict, consistency_scores, gt_correct_scores
                                )
                                metrics.update({"train/" + k: v for k, v in cm_metrics.items()})
                                
                                # 2. Inject token_level_rewards into batch_second at last valid response token
                                response_length = batch_second.batch["responses"].shape[-1]
                                token_level_rewards = torch.zeros(len(batch_second), response_length)
                                prompt_length = batch_second.batch["prompts"].shape[-1]
                                batch_second.batch["response_mask"] = batch_second.batch["attention_mask"][:, prompt_length:]
                                
                                for i in range(len(batch_second)):
                                    valid_resp_len = int(batch_second.batch["response_mask"][i].sum().item())
                                    if valid_resp_len > 0:
                                        token_level_rewards[i, valid_resp_len - 1] = rewards[i]
                                batch_second.batch["token_level_rewards"] = token_level_rewards
                                
                                # 3. Compute old_log_probs
                                print(f"[TwoStage] Computing old_log_probs for {len(batch_second)} verification samples")
                                old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_second)
                                batch_second.batch["old_log_probs"] = old_log_prob_output.batch["old_log_probs"]
                                
                                # 4. Compute GRPO advantages using candidate-level uid
                                advantages, returns = compute_grpo_outcome_advantage(
                                    token_level_rewards=batch_second.batch["token_level_rewards"],
                                    response_mask=batch_second.batch["response_mask"],
                                    index=batch_second.non_tensor_batch["uid"]
                                )
                                
                                # 5. Scale advantages by per-sample consistency as dynamic lambda
                                # Build per-sample consistency weights from verify_mapping
                                per_sample_consistency = np.zeros(len(batch_second), dtype=np.float32)
                                for si, m in enumerate(verify_mapping):
                                    g_idx = m["prompt_group_idx"]
                                    per_sample_consistency[si] = consistency_scores.get(g_idx, 1.0)
                                consistency_weights = torch.tensor(per_sample_consistency, dtype=advantages.dtype, device=advantages.device)
                                # dynamic_lambda = consistency_per_sample
                                dynamic_lambda = consistency_weights.unsqueeze(-1)
                                advantages = advantages * dynamic_lambda
                                # Store per-sample consistency for downstream logging
                                batch_second.non_tensor_batch["consistency_rate"] = per_sample_consistency
                                # Completely skip Stage 2 policy update (Verification is only used for diagnostics)
                                batch_second = None

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item(), "train/entropy": entropy_loss.detach().item()}
                        # Added: average token entropy in eval stage for confirming entropy calculation
                        old_log_prob_metrics["actor/entropy_mean_eval"] = entropys.mean().detach().item()
                        metrics.update(old_log_prob_metrics)
                        #old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_ttrl:
                            sorted_indices = sorted(range(len(batch)), key=lambda i: batch[i].non_tensor_batch["extra_info"]["index"])
                            batch = batch[sorted_indices]
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                            if self.use_ttrl:
                                from copy import deepcopy
                                ttrl_metrics = reward_result["ttrl_info"]
                                for k, v in ttrl_metrics.items():
                                    if not k.startswith("_"):  # Skip per-sample arrays
                                        metrics.update({f"train/{k}": v})
                                
                                # Log accuracy comparison to terminal
                                if "label_accuracy_majority" in ttrl_metrics and "label_accuracy_two_stage" in ttrl_metrics:
                                    maj_acc = ttrl_metrics["label_accuracy_majority"]
                                    # Use the accuracy of overall actually updated samples
                                    ts_updated_acc = metrics.get("train/two_stage_updated_label_accuracy", ttrl_metrics["label_accuracy_two_stage"])
                                    n_updated = int(metrics.get("train/two_stage_updated_group_count", 0))
                                    n_skipped = sum(1 for r in (verified_routes or []) if r == "B2")
                                    print(f"[TwoStage] Accuracy Comparison: Majority={maj_acc:.4f}, Two-Stage(Updated)={ts_updated_acc:.4f}, Updated={n_updated}, Skipped={n_skipped}")
                                    metrics["train/update_reward_acc"] = ts_updated_acc - maj_acc
                                
                                # Down Sampling
                                batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
                                self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt
                                
                                if "_answer_types" in ttrl_metrics:
                                    batch.non_tensor_batch["answer_types"] = ttrl_metrics["_answer_types"]
                                if "_oracle_answer_types" in ttrl_metrics:
                                    batch.non_tensor_batch["oracle_answer_types"] = ttrl_metrics["_oracle_answer_types"]
                                if "_consistency_rate" in ttrl_metrics:
                                    batch.non_tensor_batch["consistency_rate"] = ttrl_metrics["_consistency_rate"]
                                if "_accuracy_rate" in ttrl_metrics:
                                    batch.non_tensor_batch["accuracy_rate"] = ttrl_metrics["_accuracy_rate"]
                                if "_label_accuracy" in ttrl_metrics:
                                    batch.non_tensor_batch["label_accuracy"] = ttrl_metrics["_label_accuracy"]
                                if "_zero_advantage_mask" in ttrl_metrics:
                                    new_mask = ttrl_metrics["_zero_advantage_mask"]
                                    if "zero_advantage_mask" in batch.non_tensor_batch:
                                        # Merge with existing mask (e.g. from no_update_both) via logical OR
                                        existing_mask = batch.non_tensor_batch["zero_advantage_mask"]
                                        batch.non_tensor_batch["zero_advantage_mask"] = np.maximum(existing_mask, new_mask)
                                    else:
                                        batch.non_tensor_batch["zero_advantage_mask"] = new_mask
                                
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        batch.batch["token_level_scores"] = reward_tensor

                        # Note: balance_batch already called once at line 1427
                        # Skip the second balance_batch in TTRL mode to avoid redundant sorting
                        # which causes OOM with large batch sizes (80x samples)
                        # if self.use_ttrl:
                        #     self._balance_batch(batch, metrics=metrics)

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        # Prepare configuration for advantage estimators
                        diversity_density_config = {
                            "k": getattr(self.config.algorithm, "k", 4),
                            "lam_div": getattr(self.config.algorithm, "lam_div", 0.05),
                            "c_max": getattr(self.config.algorithm, "c_max", 2.0),
                            "div_sc_threshold": getattr(self.config.algorithm, "div_sc_threshold", 0.3),
                            "epsilon": getattr(self.config.algorithm, "epsilon", 1e-6),
                        }

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator, 
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            diversity_density_config=diversity_density_config,
                        )
                        
                        # === Minority Bias Mitigation Metrics ===
                        # Always compute the low-consistency count if we have the mask
                        if "two_stage_high_consistency_mask" in batch.batch:
                            n_lc_total = int((1.0 - batch.batch["two_stage_high_consistency_mask"]).sum().item())
                            metrics["train/test_minority_low_consistency_count"] = float(n_lc_total)
                        
                        # Apply advantage zeroing and compute ratio
                        if "zero_advantage_mask" in batch.non_tensor_batch:
                            zero_mask = torch.tensor(
                                batch.non_tensor_batch["zero_advantage_mask"],
                                dtype=torch.float32,
                                device=batch.batch["advantages"].device
                            ).unsqueeze(-1)  # (batch_size, 1)
                            
                            batch.batch["advantages"] = batch.batch["advantages"] * (1.0 - zero_mask)
                            n_zeroed = int(zero_mask.sum().item())
                            
                            # Use n_lc_total if available, else fallback to n_zeroed
                            denom = n_lc_total if 'n_lc_total' in locals() else n_zeroed
                            
                            print(f"[test_minority] Applied zero_advantage_mask: zeroed {n_zeroed}/{denom} LC samples")
                            metrics["train/test_minority_zeroed_ratio"] = (
                                float(n_zeroed) / max(1.0, float(denom))
                            )
                        
                        # === Advantage Bias Diagnostics (PASS_GRPO only) ===
                        if (
                            "oracle_answer_types" in batch.non_tensor_batch
                            and self.config.algorithm.adv_estimator == AdvantageEstimator.PASS_GRPO
                        ):
                            try:
                                # Use default k value for oracle advantage calculation
                                oracle_adv, _ = core_algos.compute_pass_grpo_advantage(
                                    token_level_rewards=batch.batch["token_level_rewards"],
                                    response_mask=batch.batch["response_mask"],
                                    index=batch.non_tensor_batch["uid"],
                                    answer_types=batch.non_tensor_batch["oracle_answer_types"],
                                    k=4,
                                )
                                tta_adv = batch.batch["advantages"]
                                
                                # Per-sample scalar advantages
                                tta_scalar = tta_adv.sum(-1)
                                oracle_scalar = oracle_adv.sum(-1)
                                valid = batch.batch["response_mask"].sum(-1) > 0
                                
                                if valid.any():
                                    # Sign match rate: how often TTA and Oracle agree on direction
                                    sign_match = ((tta_scalar > 0) == (oracle_scalar > 0)).float()
                                    metrics["diag/adv_sign_match_rate"] = sign_match[valid].mean().item()
                                    
                                    # MSE between TTA and Oracle advantages
                                    metrics["diag/adv_mse"] = ((tta_scalar - oracle_scalar) ** 2)[valid].mean().item()
                                    
                                    # Mean bias (positive = TTA overestimates)
                                    metrics["diag/adv_mean_bias"] = (tta_scalar - oracle_scalar)[valid].mean().item()
                            except Exception as e:
                                print(f"Warning: Advantage bias diagnostics failed: {e}")
                        
                        # Log pass_grpo diagnostic metrics if available
                        if "pass_grpo/correct_ratio" in batch.meta_info:
                            metrics["train/pass_grpo_correct_ratio"] = float(batch.meta_info["pass_grpo/correct_ratio"])
                        if "pass_grpo/avg_correct_advantage" in batch.meta_info:
                            metrics["train/pass_grpo_avg_correct_adv"] = float(batch.meta_info["pass_grpo/avg_correct_advantage"])
                        if "pass_grpo/avg_incorrect_advantage" in batch.meta_info:
                            metrics["train/pass_grpo_avg_incorrect_adv"] = float(batch.meta_info["pass_grpo/avg_incorrect_advantage"])
                        if "pass_grpo/avg_total_advantage" in batch.meta_info:
                            metrics["train/pass_grpo_avg_total_adv"] = float(batch.meta_info["pass_grpo/avg_total_advantage"])
                        
                        # Log diversity density advantage metrics
                        if "diversity/avg_advantage" in batch.meta_info:
                            metrics["train/diversity_avg_adv"] = float(batch.meta_info["diversity/avg_advantage"])
                        # Log pass_grpo_penalized metrics if available
                        for pp_key in [
                            "pass_grpo_penalized/avg_r_div",
                            "pass_grpo_penalized/r_div_triggered_ratio",
                            "pass_grpo_penalized/avg_raw_a_passk",
                            "pass_grpo_penalized/avg_adv_raw",
                            "pass_grpo_penalized/avg_total_advantage",
                            "pass_grpo_penalized/positive_clamped_ratio",
                        ]:
                            if pp_key in batch.meta_info:
                                metrics[f"train/{pp_key.replace('/', '_')}"] = float(batch.meta_info[pp_key])

                        # === Length & Pass@K Training Diagnostics ===
                        if self.use_ttrl:
                            resp_lengths = batch.batch["response_mask"].sum(dim=-1).float().cpu()

                            # Length by majority / non-majority (pseudo-label based)
                            if "answer_types" in batch.non_tensor_batch:
                                at = np.array(batch.non_tensor_batch["answer_types"])
                                maj_mask = (at == 0)
                                if maj_mask.any():
                                    ml = resp_lengths[maj_mask]
                                    metrics["train/length_majority_mean"] = ml.mean().item()
                                    metrics["train/length_majority_std"] = ml.std().item() if len(ml) > 1 else 0.0
                                if (~maj_mask).any():
                                    nml = resp_lengths[~maj_mask]
                                    metrics["train/length_non_majority_mean"] = nml.mean().item()
                                    metrics["train/length_non_majority_std"] = nml.std().item() if len(nml) > 1 else 0.0

                            # Length by correct / incorrect (GT-based, when available)
                            if "oracle_answer_types" in batch.non_tensor_batch:
                                oat = np.array(batch.non_tensor_batch["oracle_answer_types"])
                                correct_mask = (oat == 0)
                                if correct_mask.any():
                                    cl = resp_lengths[correct_mask]
                                    metrics["train/length_correct_mean"] = cl.mean().item()
                                    metrics["train/length_correct_std"] = cl.std().item() if len(cl) > 1 else 0.0
                                if (~correct_mask).any():
                                    il = resp_lengths[~correct_mask]
                                    metrics["train/length_incorrect_mean"] = il.mean().item()
                                    metrics["train/length_incorrect_std"] = il.std().item() if len(il) > 1 else 0.0

                            # Training Pass@K (pseudo-label based)
                            from math import comb as math_comb
                            uids = batch.non_tensor_batch["uid"]
                            # Use binary accuracy instead of reward sum to avoid KL/penalty interference
                            acc_tensor = batch.batch["acc"].cpu()
                            unique_uids = list(dict.fromkeys(uids))

                            for k_val in [1, 4, 16]:
                                pass_at_k_list = []
                                for uid in unique_uids:
                                    group_mask = np.array([u == uid for u in uids])
                                    group_acc = acc_tensor[group_mask]
                                    n = len(group_acc)
                                    c = int(group_acc.sum().item())
                                    if n >= k_val and math_comb(n, k_val) > 0:
                                        p = 1.0 - math_comb(n - c, k_val) / math_comb(n, k_val)
                                        pass_at_k_list.append(p)
                                    # if n < k_val, we don't append to avoid biased estimation
                                if pass_at_k_list:
                                    metrics[f"train/pass@{k_val}"] = float(np.mean(pass_at_k_list))

                            # Oracle Pass@K (GT-based, when available)
                            if "oracle_answer_types" in batch.non_tensor_batch:
                                oat = np.array(batch.non_tensor_batch["oracle_answer_types"])
                                for k_val in [1, 4, 16]:
                                    oracle_pass_list = []
                                    for uid in unique_uids:
                                        group_mask = np.array([u == uid for u in uids])
                                        group_oat = oat[group_mask]
                                        n = len(group_oat)
                                        c = int((group_oat == 0).sum())
                                        if n >= k_val and math_comb(n, k_val) > 0:
                                            p = 1.0 - math_comb(n - c, k_val) / math_comb(n, k_val)
                                            oracle_pass_list.append(p)
                                    if oracle_pass_list:
                                        metrics[f"train/pass@{k_val}_oracle"] = float(np.mean(oracle_pass_list))

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            if 'batch_second' in locals() and batch_second is not None and len(batch_second) > 0:
                                batch_second.meta_info["has_second_stage"] = True
                                actor_output = self.actor_rollout_wg.update_actor(batch, batch_second)
                            else:
                                # Fix: match world_size so chunking works (1 sample per GPU)
                                dummy_second = batch[:self.actor_rollout_wg.world_size]
                                dummy_second.meta_info["has_second_stage"] = False
                                actor_output = self.actor_rollout_wg.update_actor(batch, dummy_second)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        if self.use_ttrl:
                            # Logging-only metrics; compute them after the actor update path.
                            post_reward_result = self.reward_fn.compute_post_ttrl_metrics(batch)
                            for k, v in post_reward_result.items():
                                metrics.update({f"train/{k}": v})

                            post_entropy_loss = agg_loss(
                                loss_mat=batch.batch["entropys"], loss_mask=batch.batch["response_mask"], loss_agg_mode=loss_agg_mode
                            )
                            metrics.update({"train/post_entropy": post_entropy_loss.detach().item()})

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Explicitly free batch and metrics from this step to prevent memory leaks from python's delayed GC
                del batch, batch_dict, metrics, timing_raw
                if 'gen_batch' in locals():
                    del gen_batch
                if 'gen_batch_output' in locals():
                    del gen_batch_output
                if 'batch_second' in locals():
                    del batch_second
                if 'dummy_second' in locals():
                    del dummy_second
                import gc
                gc.collect()
                
                # Force GPU cache cleanup to prevent accumulation of unused tensors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1