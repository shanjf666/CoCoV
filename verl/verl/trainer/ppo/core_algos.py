# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import torch
from scipy.special import gammaln

import verl.utils.torch_functional as verl_F


# ==============================================================================
# Delta Pass@k: Label-Free Expected Marginal Contribution Utilities
# ==============================================================================

def compute_delta_pass_k(N: int, c: int, k: int) -> float:
    """
    Compute the marginal contribution Δ(c) of one sample to a cluster of size c
    in the Pass@k probability.
    
    Δ(c) = S(c) - S(c-1) = C(N-c, k-1) / C(N, k)
    
    This measures how much a single sample's existence raises the probability
    of its answer cluster being sampled in a random k-subset.
    
    Properties:
        - Strictly monotonically decreasing in c
        - Maximum at c=1: Δ(1) = k/N
        - Approaches 0 as c → N-k+1
    
    Args:
        N: Total number of samples (rollout universe size)
        c: Current cluster size (how many times this answer appeared)
        k: Target k for pass@k
    
    Returns:
        float: The marginal contribution value
    """
    if c > N or c < 1:
        return 0.0
    if (N - c) < (k - 1):
        # When cluster is so big that remaining samples can't fill k-1 slots,
        # the marginal contribution is 0 (already saturated)
        return 0.0
    if k <= 0 or N <= 0:
        return 0.0
    return math.comb(N - c, k - 1) / math.comb(N, k)


def compute_expected_marginal_passk_rewards(
    cluster_sizes: List[int],
    N: int,
    k: int,
) -> List[float]:
    """
    Compute the expected marginal Pass@k reward for each sample based on its
    cluster size. This is the core "bandpass filter" reward function:
    
        g(c) = P(correct) * Δ(c) = (c/N) * Δ(c)
    
    where P(correct) ≈ c/N is the Bayesian prior that this cluster is the
    correct answer (self-consistency frequency).
    
    The resulting function has an inverted-U shape:
        - c=1 (hallucination): P(correct)≈0 → g≈0 (filters noise)
        - c≈N (majority):      Δ(c)≈0     → g≈0 (penalizes over-exploitation)
        - c moderate:           g peaks     → rewards exploration
    
    Args:
        cluster_sizes: List of cluster sizes c_i, one per sample (length N)
        N: Total number of samples
        k: Target k for pass@k
    
    Returns:
        List[float]: Reward values for each sample
    """
    rewards = []
    for c in cluster_sizes:
        delta = compute_delta_pass_k(N, c, k)
        p_correct = c / N if N > 0 else 0.0
        rewards.append(p_correct * delta)
    return rewards


# ==============================================================================
# Hypergeometric Distribution Based Diversity Density Advantage
# ==============================================================================

def _log_comb(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Compute log(C(n, k)) using log-gamma for numerical stability.
    
    Uses: log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
                      = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    
    Args:
        n: Array of n values (can be negative, will return -inf)
        k: Array of k values
    
    Returns:
        Array of log(C(n,k)) values
    """
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    
    # Handle invalid cases: n < k or n < 0 or k < 0
    valid = (n >= k) & (n >= 0) & (k >= 0)
    result = np.full_like(n, -np.inf, dtype=np.float64)
    
    # Only compute for valid cases
    valid_n = n[valid]
    valid_k = k[valid]
    result[valid] = (gammaln(valid_n + 1) - gammaln(valid_k + 1) - 
                     gammaln(valid_n - valid_k + 1))
    
    return result


def _prob_not_in_group(N: int, s_i: int, k: int) -> float:
    """
    Compute P(u_i not in G) = C(N - s_i, k) / C(N, k).
    
    Uses product form for numerical stability(stable way for k):
    P = prod_{j=0}^{k-1} (N - s_i - j) / (N - j)
    
    Args:
        N: Total number of samples
        s_i: Count of answer type i
        k: Group size
    
    Returns:
        Probability that answer type i is not in a random group of size k
    """
    if s_i >= N or k > N - s_i:
        return 0.0
    
    log_prob = 0.0
    for j in range(k):
        if N - j <= 0:
            return 0.0
        log_prob += np.log(max(N - s_i - j, 1e-10)) - np.log(N - j)
    
    return np.exp(log_prob)


def _prob_not_in_group_vectorized(N: int, s_arr: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized version: compute P(u_i not in G) for all answer types.
    
    P(u_i not in G) = C(N - s_i, k) / C(N, k)
    
    Args:
        N: Total number of samples
        s_arr: Array of counts for each answer type [s_1, s_2, ..., s_D]
        k: Group size
    
    Returns:
        Array of probabilities for each answer type
    """
    # Use log-space for numerical stability
    log_comb_N_k = _log_comb(np.array([N]), np.array([k]))[0]
    log_comb_N_minus_s_k = _log_comb(N - s_arr, np.full_like(s_arr, k))
    
    log_probs = log_comb_N_minus_s_k - log_comb_N_k
    
    # Convert back to probability, handle -inf
    probs = np.where(np.isfinite(log_probs), np.exp(log_probs), 0.0)
    
    return probs


def _prob_not_in_group_excluding_one(N: int, s_j: int, k: int) -> float:
    """
    Compute P(u_j not in G | x in G where x belongs to type t).
    
    This is C(N - 1 - s_j, k - 1) / C(N - 1, k - 1) for j != t.
    
    Essentially, we remove one sample (x) from the pool, so N -> N-1, k -> k-1.
    
    Args:
        N: Original total number of samples
        s_j: Count of answer type j (j != t)
        k: Original group size
    
    Returns:
        Conditional probability
    """
    new_N = N - 1
    new_k = k - 1
    
    if new_k <= 0:
        return 1.0  # No other samples in group
    
    if s_j >= new_N or new_k > new_N - s_j:
        return 0.0
    
    log_prob = 0.0
    for j in range(new_k):
        if new_N - j <= 0:
            return 0.0
        log_prob += np.log(max(new_N - s_j - j, 1e-10)) - np.log(new_N - j)
    
    return np.exp(log_prob)


def compute_pass_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    answer_types: np.ndarray,
    k: int,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pass@k advantage using the correct mathematical formulation.
    
    References:
    - Group Mean Reward (R_bar): Eq. 11
    - Group Std (sigma): Eq. 12
    - Positive Advantage (A_pos): Eq. 14
    - Negative Advantage (A_neg): Eq. 15
    
    Args:
        token_level_rewards: (torch.Tensor) shape (bs, response_length)
        response_mask: (torch.Tensor) shape (bs, response_length)
        index: (np.ndarray) prompt indices for each sample
        answer_types: (np.ndarray) answer type/id for each sample (0 is correct)
        k: Group size
        epsilon: Small value for numerical stability
    
    Returns:
        advantages: (torch.Tensor) shape (bs, response_length)
        returns: (torch.Tensor) same as advantages
    """
    bs, response_length = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    
    # Group samples by prompt
    prompt_to_samples = defaultdict(list)
    prompt_to_answers = defaultdict(list)
    
    for i in range(bs):
        prompt_idx = index[i]
        prompt_to_samples[prompt_idx].append(i)
        prompt_to_answers[prompt_idx].append(answer_types[i])
    
    # Initialize advantages
    advantages = torch.zeros(bs, 1, dtype=dtype, device=device)
    
    for prompt_idx, sample_indices in prompt_to_samples.items():
        N = len(sample_indices)
        answers = prompt_to_answers[prompt_idx]
        
        # Count outcomes
        # Assume 0 is the correct answer type
        c_maj = sum(1 for a in answers if a == 0)
        c_neg = N - c_maj
        
        # 1. Compute Group Mean Reward (Prob of at least one correct in k samples)
        # R_mean = 1 - C(N_neg, k) / C(N, k)
        # Use log-space for stability: log(P_fail) = log_comb(N_neg, k) - log_comb(N, k)
        log_prob_fail = _log_comb(np.array([c_neg]), np.array([k]))[0] - \
                        _log_comb(np.array([N]), np.array([k]))[0]
        
        prob_fail = np.exp(log_prob_fail) if np.isfinite(log_prob_fail) else 0.0
        prob_fail = np.clip(prob_fail, 0.0, 1.0)
        r_mean = 1.0 - prob_fail
        
        # 2. Compute Group Standard Deviation
        variance = r_mean * (1.0 - r_mean)
        std = np.sqrt(variance)
        
        if std < epsilon:
            # Deterministic outcome (always pass or always fail) -> 0 advantage
            a_pos = 0.0
            a_neg = 0.0
        else:
            # 3. Positive Advantage (Eq. 14)
            # A_pos = (1 - R_mean) / sigma = P_fail / sigma
            a_pos = prob_fail / std
            
            # 4. Negative Advantage (Eq. 15)
            # A_neg = ( (1 - R_mean) - P_cond ) / sigma
            # P_cond = C(N_neg - 1, k - 1) / C(N - 1, k - 1) (Prob failure given we picked a negative)
            
            if c_neg >= 1:
                log_p_cond_fail = _log_comb(np.array([c_neg - 1]), np.array([k - 1]))[0] - \
                                  _log_comb(np.array([N - 1]), np.array([k - 1]))[0]
                p_cond_fail = np.exp(log_p_cond_fail) if np.isfinite(log_p_cond_fail) else 0.0
                p_cond_fail = np.clip(p_cond_fail, 0.0, 1.0)
            else:
                p_cond_fail = 0.0 # Should not happen if we are assigning to a negative sample
            
            # A_neg = (P_fail - P_cond_fail) / sigma
            a_neg = (prob_fail - p_cond_fail) / std
            
        # Assign advantages based on correctness
        # We need to map back to the original indices
        
        for local_i, global_i in enumerate(sample_indices):
            is_correct = (answers[local_i] == 0)
            val = a_pos if is_correct else a_neg
            advantages[global_i] = val
            
    # Expand to response length and mask
    advantages = advantages * response_mask
    returns = advantages.clone()
    
    return advantages, returns


def compute_pass_grpo_penalized_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    answer_types: np.ndarray,
    consistency_rates: np.ndarray,
    diversity_density_config: dict,
    k: int,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute pass_grpo advantage with length diversity reward.
    
    All intermediate computation is done on CPU/numpy to avoid CUDA memory
    fragmentation from many small GPU tensor allocations in loops.
    Only the final advantage tensor is created on GPU.
    
    Components:
    - A_pass@k: pass@k-based advantage (combinatorial)
    - R_div: dynamic length diversity reward for correct answers when sc > threshold
    """
    bs, response_length = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    
    lam_div = diversity_density_config.get("lam_div", 0.2)
    c_max = diversity_density_config.get("c_max", 2.0)
    div_sc_threshold = diversity_density_config.get("div_sc_threshold", 0.3)
    with torch.no_grad():
        prompt_to_samples = defaultdict(list)
        prompt_to_answers = defaultdict(list)
        prompt_to_consistency = {}
        
        for i in range(bs):
            prompt_idx = index[i]
            prompt_to_samples[prompt_idx].append(i)
            prompt_to_answers[prompt_idx].append(answer_types[i])
            prompt_to_consistency[prompt_idx] = consistency_rates[i]
        
        # Move actual_lengths to CPU once, avoid per-item .item() GPU sync in loop
        actual_lengths_cpu = response_mask.sum(dim=-1).long().cpu().numpy()  # (bs,)
        
        # All intermediate computation on CPU/numpy
        advantages_raw_np = np.zeros(bs, dtype=np.float64)
        
        # === Logging accumulators ===
        total_r_div = 0.0
        r_div_count = 0
        total_adv_raw = 0.0
        total_a_passk = 0.0
        total_clamped = 0
        total_positive = 0
        
        for prompt_idx, sample_indices in prompt_to_samples.items():
            N = len(sample_indices)
            answers = prompt_to_answers[prompt_idx]
            sc_ratio = prompt_to_consistency[prompt_idx]
            
            c_maj = sum(1 for a in answers if a == 0)
            c_neg = N - c_maj
            
            log_prob_fail = _log_comb(np.array([c_neg]), np.array([k]))[0] - _log_comb(np.array([N]), np.array([k]))[0]
            prob_fail = np.exp(log_prob_fail) if np.isfinite(log_prob_fail) else 0.0
            prob_fail = np.clip(prob_fail, 0.0, 1.0)
            r_mean = 1.0 - prob_fail
            variance = r_mean * (1.0 - r_mean)
            std = np.sqrt(variance)
            
            if std < epsilon:
                a_pos, a_neg = 0.0, 0.0
            else:
                a_pos = prob_fail / std
                if c_neg >= 1:
                    log_p_cond_fail = _log_comb(np.array([c_neg - 1]), np.array([k - 1]))[0] - _log_comb(np.array([N - 1]), np.array([k - 1]))[0]
                    p_cond_fail = np.exp(log_p_cond_fail) if np.isfinite(log_p_cond_fail) else 0.0
                    p_cond_fail = np.clip(p_cond_fail, 0.0, 1.0)
                else:
                    p_cond_fail = 0.0
                a_neg = (prob_fail - p_cond_fail) / std
            
            # All group-level computation on CPU with numpy
            group_raw_adv = np.zeros(N, dtype=np.float64)
            group_r_div = np.zeros(N, dtype=np.float64)
            
            correct_lengths = []
            for local_i, global_i in enumerate(sample_indices):
                if answers[local_i] == 0:
                    correct_lengths.append(int(actual_lengths_cpu[global_i]))

            if sc_ratio > div_sc_threshold and len(correct_lengths) > 0:
                mu_l = sum(correct_lengths) / len(correct_lengths)
                var_l = sum((x - mu_l)**2 for x in correct_lengths) / len(correct_lengths)
                sigma_l = np.sqrt(var_l)
                

                current_lam_div = lam_div
                for local_i, global_i in enumerate(sample_indices):
                    if answers[local_i] == 0:
                        l_i = int(actual_lengths_cpu[global_i])
                        div_val = abs(l_i - mu_l) / (sigma_l + 1e-5)
                        reward_div = current_lam_div * min(div_val, c_max)
                        group_r_div[local_i] = reward_div
                    
                        if reward_div > 0:
                            total_r_div += reward_div
                            r_div_count += 1
    
            for local_i, global_i in enumerate(sample_indices):
                a_pass_k = a_pos if answers[local_i] == 0 else a_neg
                
                # raw_adv_i = A_pass@k + R_div
                raw_v = a_pass_k + group_r_div[local_i]
                group_raw_adv[local_i] = raw_v
                
                total_a_passk += a_pass_k
                total_adv_raw += raw_v
                
               
            # Conditional Normalization: Only re-normalize if rewards were added
            # If not added, raw_v is the raw Pass@k advantage which is already theoretically normalized
            if N > 1 and sc_ratio > div_sc_threshold:
                mu_adv = np.mean(group_raw_adv)
                std_adv = np.std(group_raw_adv, ddof=0)
                group_final_adv = (group_raw_adv - mu_adv) / (std_adv + epsilon)
            else:
                group_final_adv = group_raw_adv
            
            # Positive-sample clamping: ensure correct answers never get negative advantage
            # For positive samples, take max(a_pos, A_normalized) so that the advantage
            # is at least the theoretical Pass@k base value
            for local_i, global_i in enumerate(sample_indices):
                if answers[local_i] == 0:  # correct answer
                    if group_final_adv[local_i] < a_pos:
                        group_final_adv[local_i] = a_pos
                        total_clamped += 1
                    total_positive += 1
                advantages_raw_np[global_i] = group_final_adv[local_i]
    
        metrics = {
            "pass_grpo_penalized/avg_r_div": total_r_div / r_div_count if r_div_count > 0 else 0.0,
            "pass_grpo_penalized/r_div_triggered_ratio": r_div_count / bs if bs > 0 else 0.0,
            "pass_grpo_penalized/avg_raw_a_passk": total_a_passk / bs if bs > 0 else 0.0,
            "pass_grpo_penalized/avg_adv_raw": total_adv_raw / bs if bs > 0 else 0.0,
            "pass_grpo_penalized/lam_div": lam_div,
            "pass_grpo_penalized/div_sc_threshold": div_sc_threshold,
            "pass_grpo_penalized/positive_clamped_ratio": total_clamped / total_positive if total_positive > 0 else 0.0,
        }

    
        # Create GPU tensors only once at the end
        advantages_raw_tensor = torch.tensor(advantages_raw_np, dtype=dtype, device=device)
        advantages = advantages_raw_tensor.unsqueeze(-1) * response_mask
        returns = advantages.clone()  # outcome-based: returns == advantages, ensure independent tensors

        # [FIX #2] Explicit GPU memory cleanup to prevent fragmentation
        # When response_length is large (4096+), intermediate tensors can consume significant memory
        del actual_lengths_cpu
        del advantages_raw_np
        del advantages_raw_tensor  # Explicitly delete intermediate GPU tensor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Force cache cleanup to free reserved but unallocated memory
        
    return advantages, returns, metrics


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, (
        f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
