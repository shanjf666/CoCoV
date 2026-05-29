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
Single Process Actor
"""

import itertools
from typing import Tuple

import numpy as np
import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, compute_topk=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len) or None
            log_probs: # (bs, response_len)
            topk: # (bs, response_len, K) or None — only when compute_topk=True
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy or compute_topk:
                    inplace_backward = False  # need logits alive for entropy/topk
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward
                )

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # compute top-k log probabilities
                if compute_topk:
                    topk_rmpad = verl_F.topk_values_from_logits(logits_rmpad, k=5)  # ((total_nnz / sp) + pad, 5)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if compute_topk:
                        topk_rmpad = gather_outpus_and_unpad(
                            topk_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                    )
                if compute_topk:
                    full_topk = pad_input(
                        hidden_states=topk_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if compute_topk:
                    topk = full_topk[:, -response_length - 1 : -1, :]  # (bsz, response_length, 5)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                if compute_topk:
                    topk = verl_F.topk_values_from_logits(logits, k=5)  # (bsz, response_length, 5)

            return entropy, log_probs, topk

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            response_mask = micro_batch["attention_mask"][:, -micro_batch["responses"].size(-1) :]
            with torch.no_grad():
                entropy, log_probs, _ = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    def update_policy(self, data: DataProto, data_second: DataProto = None):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        high_consistency_key = "two_stage_high_consistency_mask"

        # Diagnostic accumulators for high-consistency samples (Route A)
        stage1_hc_pg_loss_token_sum = 0.0
        stage1_hc_token_count = 0.0
        stage1_hc_sample_count = 0.0
        stage2_hc_pg_loss_token_sum = 0.0
        stage2_hc_token_count = 0.0
        stage2_hc_sample_count = 0.0


        def _compute_seq_pg_token_stats(
            old_log_prob_t,
            log_prob_t,
            advantages_t,
            response_mask_t,
            cliprange_t,
            cliprange_low_t,
            cliprange_high_t,
            clip_ratio_c_t,
        ):
            """Return per-sequence token-sum pg losses and token counts under PPO clipping."""
            negative_approx_kl_t = log_prob_t - old_log_prob_t
            ratio_t = torch.exp(negative_approx_kl_t)

            pg_losses1_t = -advantages_t * ratio_t
            if cliprange_low_t is None:
                cliprange_low_t = cliprange_t
            if cliprange_high_t is None:
                cliprange_high_t = cliprange_t

            pg_losses2_t = -advantages_t * torch.clamp(
                ratio_t,
                1 - cliprange_low_t,
                1 + cliprange_high_t,
            )
            clip_pg_losses1_t = torch.maximum(pg_losses1_t, pg_losses2_t)

            pg_losses3_t = -advantages_t * clip_ratio_c_t
            clip_pg_losses2_t = torch.min(pg_losses3_t, clip_pg_losses1_t)
            pg_losses_t = torch.where(advantages_t < 0, clip_pg_losses2_t, clip_pg_losses1_t)

            seq_token_sums_t = torch.sum(pg_losses_t * response_mask_t, dim=-1)
            seq_token_counts_t = torch.sum(response_mask_t, dim=-1).clamp(min=1)
            return seq_token_sums_t, seq_token_counts_t

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if high_consistency_key in data.batch.keys():
            select_keys.append(high_consistency_key)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        # ==========================================================
        # Stage 2 Setup: Group by UID and split evenly across mini-batches
        # ==========================================================
        verification_mini_batches = [[] for _ in range(max(1, num_mini_batches))]

        if data_second is not None:
            # Group Stage 2 samples by UID (same prompt & candidate)
            uids = data_second.non_tensor_batch.get("uid", None)
            if uids is not None:
                unique_uids = list(set(uids))
                verification_groups = []
                for uid in unique_uids:
                    mask = np.array([u == uid for u in uids])
                    group_data = data_second[np.where(mask)[0].tolist()]
                    # Filter keys that exist in both select_keys and group_data.batch
                    available_keys = [k for k in select_keys if k in group_data.batch.keys()]
                    group_batch = group_data.select(batch_keys=available_keys).batch
                    verification_groups.append(group_batch)
                
                # Distribute groups evenly into verification_mini_batches
                for i, group in enumerate(verification_groups):
                    verification_mini_batches[i % max(1, num_mini_batches)].append(group)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    compute_topk = self.config.get("compute_topk_metrics", True)  # Can disable to save memory
                    entropy, log_prob, topk = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, compute_topk=compute_topk
                    )

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    high_consistency_mask = data.get(high_consistency_key, None)
                    if high_consistency_mask is not None:
                        seq_token_sums, seq_token_counts = _compute_seq_pg_token_stats(
                            old_log_prob_t=old_log_prob,
                            log_prob_t=log_prob,
                            advantages_t=advantages,
                            response_mask_t=response_mask,
                            cliprange_t=clip_ratio,
                            cliprange_low_t=clip_ratio_low,
                            cliprange_high_t=clip_ratio_high,
                            clip_ratio_c_t=clip_ratio_c,
                        )
                        hc_mask = high_consistency_mask > 0.5
                        if hc_mask.any():
                            stage1_hc_pg_loss_token_sum += seq_token_sums[hc_mask].sum().detach().item()
                            stage1_hc_token_count += seq_token_counts[hc_mask].sum().detach().item()
                            stage1_hc_sample_count += hc_mask.sum().detach().item()

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        metrics["actor/entropy_coeff"] = float(entropy_coeff)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode
                        )

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    # Top-k token distribution metrics: Λ^(k)
                    if topk is not None:
                        for k_idx in range(topk.size(-1)):
                            k_th_log_prob = topk[:, :, k_idx]  # (bsz, response_length)
                            avg_kth = (k_th_log_prob * response_mask).sum() / response_mask.sum().clamp(min=1)
                            data[f"actor/topk_logprob_k{k_idx+1}"] = avg_kth.detach().item()
                    # Explicitly release topk tensor to free GPU memory
                        del topk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    append_to_dict(metrics, data)

                # ==========================================================
                # Stage 2 Backward Pass (for this mini-batch)
                # ==========================================================
                current_v_groups = verification_mini_batches[batch_idx] if batch_idx < len(verification_mini_batches) else []
                
                # Extract all micro batches for this mini batch
                current_v_mbs = []
                for group_batch in current_v_groups:
                    mbs = group_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    current_v_mbs.extend(mbs)
                
                # Synchronize maximum micro-batches across FSDP ranks
                device = torch.cuda.current_device()
                num_mbs_local = torch.tensor([len(current_v_mbs)], dtype=torch.int32, device=device)
                
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(num_mbs_local, op=torch.distributed.ReduceOp.MAX)
                
                num_mbs_max = num_mbs_local.item()

                if num_mbs_max > 0:
                    group_losses = []
                    # Use a micro-batch from Stage 1 as DUMMY fallback if no Stage 2 data exists on this rank
                    fallback_dummy_mb = micro_batches[0] 
                    
                    for t in range(num_mbs_max):
                        is_dummy = (t >= len(current_v_mbs))
                        if is_dummy:
                            mb_s = current_v_mbs[-1] if len(current_v_mbs) > 0 else fallback_dummy_mb
                        else:
                            mb_s = current_v_mbs[t]
                            
                        mb_s = mb_s.to(device)
                        response_mask_s = mb_s["attention_mask"][:, -mb_s["responses"].size(1):]
                        
                        _, log_prob_s, _ = self._forward_micro_batch(
                            micro_batch=mb_s, temperature=temperature, calculate_entropy=False, compute_topk=False
                        )
                        
                        # GRPO loss is just pg_loss since we already computed advantages based on group 
                        pg_loss_s, _, _, _ = compute_policy_loss(
                            old_log_prob=mb_s["old_log_probs"],
                            log_prob=log_prob_s,
                            advantages=mb_s["advantages"],
                            response_mask=response_mask_s,
                            cliprange=self.config.clip_ratio,
                            cliprange_low=self.config.get("clip_ratio_low", self.config.clip_ratio),
                            cliprange_high=self.config.get("clip_ratio_high", self.config.clip_ratio),
                            clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                            loss_agg_mode=self.config.loss_agg_mode,
                        )

                        if not is_dummy:
                            high_consistency_mask_s = mb_s.get(high_consistency_key, None)
                            if high_consistency_mask_s is not None:
                                seq_token_sums_s, seq_token_counts_s = _compute_seq_pg_token_stats(
                                    old_log_prob_t=mb_s["old_log_probs"],
                                    log_prob_t=log_prob_s,
                                    advantages_t=mb_s["advantages"],
                                    response_mask_t=response_mask_s,
                                    cliprange_t=self.config.clip_ratio,
                                    cliprange_low_t=self.config.get("clip_ratio_low", self.config.clip_ratio),
                                    cliprange_high_t=self.config.get("clip_ratio_high", self.config.clip_ratio),
                                    clip_ratio_c_t=self.config.get("clip_ratio_c", 3.0),
                                )
                                hc_mask_s = high_consistency_mask_s > 0.5
                                if hc_mask_s.any():
                                    stage2_hc_pg_loss_token_sum += seq_token_sums_s[hc_mask_s].sum().detach().item()
                                    stage2_hc_token_count += seq_token_counts_s[hc_mask_s].sum().detach().item()
                                    stage2_hc_sample_count += hc_mask_s.sum().detach().item()

                        if is_dummy:
                            loss_s = pg_loss_s * 0.0
                        else:
                            # Divide by len(current_v_mbs) locally. DDP will average this across world_size
                            loss_s = pg_loss_s / max(1, len(current_v_mbs))
                            group_losses.append(pg_loss_s.detach().item())
                            
                        loss_s.backward()
                        
                    if group_losses:
                        metrics["actor/second_stage_loss"] = sum(group_losses) / len(group_losses)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)

        stage1_hc_pg_loss_token_mean = (
            stage1_hc_pg_loss_token_sum / stage1_hc_token_count if stage1_hc_token_count > 0 else 0.0
        )
        stage2_hc_pg_loss_token_mean = (
            stage2_hc_pg_loss_token_sum / stage2_hc_token_count if stage2_hc_token_count > 0 else 0.0
        )
        stage2_to_stage1_hc_abs_ratio = (
            abs(stage2_hc_pg_loss_token_sum) / max(abs(stage1_hc_pg_loss_token_sum), 1e-8)
            if stage1_hc_token_count > 0
            else 0.0
        )
        stage2_to_stage1_hc_signed_ratio = (
            stage2_hc_pg_loss_token_sum / stage1_hc_pg_loss_token_sum
            if abs(stage1_hc_pg_loss_token_sum) > 1e-8
            else 0.0
        )

        append_to_dict(
            metrics,
            {
                "actor/high_consistency_stage1_pg_loss_token_sum": float(stage1_hc_pg_loss_token_sum),
                "actor/high_consistency_stage2_pg_loss_token_sum": float(stage2_hc_pg_loss_token_sum),
                "actor/high_consistency_stage1_pg_loss_token_mean": float(stage1_hc_pg_loss_token_mean),
                "actor/high_consistency_stage2_pg_loss_token_mean": float(stage2_hc_pg_loss_token_mean),
                "actor/high_consistency_stage1_token_count": float(stage1_hc_token_count),
                "actor/high_consistency_stage2_token_count": float(stage2_hc_token_count),
                "actor/high_consistency_stage1_sample_count": float(stage1_hc_sample_count),
                "actor/high_consistency_stage2_sample_count": float(stage2_hc_sample_count),
                "actor/high_consistency_stage2_to_stage1_pg_loss_abs_sum_ratio": float(stage2_to_stage1_hc_abs_ratio),
                "actor/high_consistency_stage2_to_stage1_pg_loss_signed_sum_ratio": float(stage2_to_stage1_hc_signed_ratio),
            },
        )

        self.actor_optimizer.zero_grad()
        return metrics
