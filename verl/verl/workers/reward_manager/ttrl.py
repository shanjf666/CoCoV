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

from collections import defaultdict, Counter
from functools import partial

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics, test_time_train_metrics)
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer


class TTRLRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", compute_score=None, n_votes_per_prompt=1, n_samples_per_prompt=1, mode="eval", eval_n_samples=1, pseudo_label_file=None, enable_hybrid=False, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        self.pseudo_label_file = pseudo_label_file
        self.enable_hybrid = enable_hybrid

        self.offline_pseudo_labels = {}
        if pseudo_label_file:
            import json
            import os
            if os.path.exists(pseudo_label_file):
                print(f"TTRLRewardManager: Loading offline pseudo-labels from {pseudo_label_file}")
                count = 0
                with open(pseudo_label_file, "r", encoding="utf-8") as f:      
                    for line in f:
                        try:
                            item = json.loads(line)
                            # Normalizing key for robust lookup
                            problem = item.get("instruction", item.get("problem", "")).strip()
                            if problem:
                                self.offline_pseudo_labels[problem] = item["voted_answer"]
                                count += 1
                        except Exception as e:
                            print(f"Warning: Failed to parse line in pseudo_label_file: {e}")
                print(f"Successfully loaded {count} pseudo-labels.")
            else:
                print(f"Warning: pseudo_label_file {pseudo_label_file} not found.")

        assert n_votes_per_prompt >= n_samples_per_prompt, f"For TTRL settings, n_votes_per_prompt {n_votes_per_prompt} should be greater than or equal to n_samples_per_prompt {n_samples_per_prompt}"

        print(f"TTRLRewardManager initialized with n_votes_per_prompt {n_votes_per_prompt}, n_samples_per_prompt {n_samples_per_prompt}, eval_n_samples {eval_n_samples}")


    def _data_source_to_task(self, data_source):
        # Standardize
        ds = str(data_source)
        if ds in ["MATH-TTT", "AIME-TTT", "AMC-TTT", "AIME25", "DAPO-TTT"]:
            return "math"
        if ds in ["GPQA-TTT", "MMLU-TTT"]:
            return "gpqa"
        if ds in ["BBEH", "bbeh", "BigBench-Extra-Hard"]:
            return "bbeh"

        dsl = ds.lower()
        # Keyword matching (more robust)
        if any(key in dsl for key in ["gpqa", "mmlu"]):
            return "gpqa"
        if any(key in dsl for key in ["aime", "math", "amc", "aime25", "dapo"]):
            return "math"
        if "bbeh" in dsl or "bigbench" in dsl:
            return "bbeh"

        raise NotImplementedError(f"Data source {data_source} is not supported for TTRLRewardManager")

    def _extract_final_answers(self, task: str, outputs: list[str]) -> list[str]:
        """
        Extract final answers from outputs for diversity calculation.
        Uses task-specific answer extraction logic with precise LaTeX normalization.
        """
        extract_fn = partial(extract_answer, data_name=task)
        normalized_outputs = [normalize_latex(x) for x in outputs]
        final_answers = [extract_fn(text) or "<empty>" for text in normalized_outputs]
        return final_answers

    def _compute_strategy_entropy(self, data_items):
        """
        Calculate strategy entropy (normalized negative log-likelihood) efficiently.
        Silent computation without print statements for production use.
        """
        if not data_items:
            return 0.0

        all_log_probs = []
        all_lens = []

        for data_item in data_items:
            try:
                if not hasattr(data_item, 'batch') or "old_log_probs" not in data_item.batch:
                    continue

                prompt_length = data_item.batch["prompts"].shape[-1]
                attention_mask = data_item.batch.get("attention_mask", None)

                if attention_mask is None or len(attention_mask) <= prompt_length:
                    continue

                response_length = int(attention_mask[prompt_length:].sum().item())
                if response_length <= 0:
                    continue

                old_log_probs = data_item.batch["old_log_probs"]
                if not isinstance(old_log_probs, torch.Tensor) or old_log_probs.numel() == 0:
                    continue

                # Slicing logic with multiple cases
                log_probs_length = old_log_probs.shape[-1]
                if log_probs_length == response_length:
                    response_log_probs = old_log_probs
                elif log_probs_length == prompt_length + response_length:
                    response_log_probs = old_log_probs[prompt_length:prompt_length + response_length]
                elif log_probs_length > response_length:
                    response_log_probs = old_log_probs[-response_length:]
                else:
                    continue

                if response_log_probs.numel() > 0:
                    log_prob_sum = float(torch.sum(response_log_probs).item())
                    all_log_probs.append(log_prob_sum)
                    all_lens.append(response_length)
            except Exception:
                continue

        if not all_log_probs:
            return 0.0

        # Vectorized calculation
        log_probs_array = np.array(all_log_probs)
        lens_array = np.array(all_lens)
        neg_log_likelihoods = -log_probs_array / lens_array
        
        return float(np.mean(neg_log_likelihoods))

    def compute_post_ttrl_metrics(self, data: DataProto):
        """
        Compute post TTRL metrics for the given data.
        """
        assert len(data) % self.n_samples_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_samples_per_prompt}"
        prompt_num = len(data) // self.n_samples_per_prompt

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
                group_vote_rewards = []
                group_pred_outputs = []
                group_labels = []
                group_extra_info = []
                task = None

                for i in range(self.n_samples_per_prompt):
                    data_item = data[prompt_i * self.n_samples_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]
                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    vote_reward = data_item.batch["acc"]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_vote_rewards.append(vote_reward)
                    group_extra_info.append(extra_info)
                
                post_ttrl_metrics = post_test_time_train_metrics(group_pred_outputs, group_labels, group_vote_rewards, task=task, extra_info=group_extra_info)
                for k, v in post_ttrl_metrics.items():
                    post_ttrl_metrics_list[k].append(v)

        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    def _compute_ttrl_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            assert len(data) % self.n_votes_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_votes_per_prompt}"
            
            prompt_num = len(data) // self.n_votes_per_prompt

            reward_tensor = torch.zeros_like(data.batch["responses"][:prompt_num*self.n_samples_per_prompt], dtype=torch.float32)

            already_print_data_sources = {}

            all_ttrl_metrics = defaultdict(list)

            scores = [0.0 for _ in range(len(data))]

            for prompt_i in range(prompt_num):
                # Cache for each sample in this prompt group (fixes variable scope bug)
                group_cache = {
                    "valid_response_lengths": [],
                    "data_sources": [],
                    "valid_prompt_indices": [],
                    "valid_response_indices": [],
                }
                group_pred_outputs = []
                group_labels = []
                group_extra_info = []
                task = None
                prompt_str = ""

                # === STAGE 1: Data extraction and caching ===
                for i in range(self.n_votes_per_prompt):
                    data_item = data[prompt_i * self.n_votes_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)

                    # Cache critical per-sample data
                    group_cache["valid_response_lengths"].append(int(valid_response_length))
                    group_cache["valid_prompt_indices"].append(valid_prompt_idx)
                    group_cache["valid_response_indices"].append(valid_response_idx)
                    
                    # Decode response for metric computation (only once in this stage)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    
                    group_cache["data_sources"].append(data_source)

                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_extra_info.append(extra_info)

                # === STAGE 2: Compute metrics (FP/FN, NLL, Diversity) ===
                # 1. Identify all online answers and calculate online consistency
                from verl.utils.reward_score.ttrl.auto_extract import auto_extract
                model_answers = auto_extract(task, group_pred_outputs, extra_info=group_extra_info)
                online_counter = Counter(model_answers)

                online_voted_answer, majority_count = online_counter.most_common(1)[0] if online_counter else (None, 0)
                online_consistency_rate = majority_count / self.n_votes_per_prompt if self.n_votes_per_prompt > 0 else 0.0

                # 2. Check for two-stage verified pseudo-label (highest priority)
                # This is injected by ray_trainer._run_two_stage_verification()
                two_stage_label = None
                two_stage_penalize = False
                first_sample_idx = prompt_i * self.n_votes_per_prompt
                if "verified_pseudo_label" in data[first_sample_idx].non_tensor_batch:
                    two_stage_raw = data[first_sample_idx].non_tensor_batch["verified_pseudo_label"]
                    if two_stage_raw is not None and str(two_stage_raw) != "None":
                        two_stage_label = str(two_stage_raw)
                    elif two_stage_raw is None:
                        # None means the "penalize" fallback was triggered
                        # (all candidate answers verified as False)
                        two_stage_penalize = True

                # 3. Extract offline pseudo-label (priority: direct lookup from JSONL)
                offline_voted_answer = self.offline_pseudo_labels.get(prompt_str.strip())

                # Fallback to metadata check if lookup failed
                if not offline_voted_answer:
                    for candidate in [data_item.non_tensor_batch.get("reward_model", {}).get("voted_answer"),
                                      data_item.non_tensor_batch.get("voted_answer")]:
                        if candidate:
                            offline_voted_answer = candidate
                            break

                # 4. Determine the label to use (priority: two-stage > hybrid offline > online majority)
                verified_label = None
                off_policy = 0.0
                label_source = "online_majority"

                if two_stage_label is not None:
                    # Two-stage verification succeeded
                    verified_label = two_stage_label
                    label_source = "two_stage_verified"
                elif two_stage_penalize:
                    # Two-stage verification says all candidates are wrong → penalize all
                    # We set verified_label to a deliberately unmatchable string
                    # so all rewards become 0, and we flag for -1 advantage later
                    verified_label = "__TWO_STAGE_PENALIZE_ALL__"
                    label_source = "two_stage_penalize"
                elif getattr(self, "enable_hybrid", False) and online_consistency_rate < 0.3:
                    if offline_voted_answer:
                        verified_label = offline_voted_answer
                        off_policy = 1.0
                        label_source = "offline_hybrid"
                    else:
                        if getattr(self, "pseudo_label_file", None):
                            print(f"Warning: Online SC {online_consistency_rate:.2f} < 0.3 but no label found for prompt in {self.pseudo_label_file}")

                # 5. Compute reward using chosen label
                rewards, ttrl_metrics, ttrl_details = test_time_train_metrics(
                    group_pred_outputs,
                    group_labels,
                    task=task,
                    extra_info=group_extra_info,
                    verified_label=verified_label,
                    model_answers=model_answers,
                    return_details=True,
                )
                
                # Accuracy comparison metrics
                ground_truth = group_labels[0]
                ttrl_metrics["label_accuracy_majority"] = float(ttrl_details.get("majority_hit", 0.0))
                ttrl_metrics["label_accuracy_two_stage"] = ttrl_metrics["label_accuracy"] # label_accuracy in ttt_metrics already uses verified_label if provided

                ttrl_metrics["off_policy_ratio"] = off_policy
                # Track label source as numeric flags for safe aggregation
                ttrl_metrics["label_source_two_stage"] = 1.0 if label_source == "two_stage_verified" else 0.0
                ttrl_metrics["label_source_penalize"] = 1.0 if label_source == "two_stage_penalize" else 0.0
                ttrl_metrics["label_source_offline"] = 1.0 if label_source == "offline_hybrid" else 0.0

                # If penalize mode triggered, override all rewards to -1
                if two_stage_penalize:
                    rewards = [-1.0] * len(rewards)
                    ttrl_metrics["two_stage_penalized"] = 1.0
                else:
                    ttrl_metrics["two_stage_penalized"] = 0.0

                # Compute FP/FN rates (skip if penalize mode set all rewards to -1)
                ground_truth = group_labels[0]
                true_rewards = ttrl_details.get("true_rewards", [])
                # For FP/FN, treat -1 rewards as negative (not a true positive)
                effective_rewards = [max(0, r) for r in rewards]
                n_pseudo_pos = sum(1 for r in effective_rewards if r > 0)
                n_false_pos = sum(1 for r, t in zip(effective_rewards, true_rewards) if r > 0 and t == 0)
                fp_rate = n_false_pos / n_pseudo_pos if n_pseudo_pos > 0 else 0.0
                n_pseudo_neg = sum(1 for r in effective_rewards if r == 0)
                n_false_neg = sum(1 for r, t in zip(effective_rewards, true_rewards) if r == 0 and t > 0)
                fn_rate = n_false_neg / n_pseudo_neg if n_pseudo_neg > 0 else 0.0
                ttrl_metrics["false_positive_rate"] = fp_rate
                ttrl_metrics["false_negative_rate"] = fn_rate

                # Calculate strategy entropy (silent, vectorized)
                current_group_data = data[prompt_i * self.n_votes_per_prompt:(prompt_i + 1) * self.n_votes_per_prompt]
                strategy_entropy = self._compute_strategy_entropy(current_group_data)
                ttrl_metrics["neg_log_likelihood"] = strategy_entropy

                # === COMPUTE DIVERSITY RATIO (NEW) ===
                final_answers = self._extract_final_answers(task, group_pred_outputs)
                unique_answers = len(set(final_answers))
                diversity_ratio = unique_answers / len(group_pred_outputs) if len(group_pred_outputs) > 0 else 0.0
                ttrl_metrics["diversity_ratio"] = diversity_ratio
                
                # Map answers to integers (0 = correct, other = incorrect) for ray_trainer GRPO consistency
                answer_to_id = {ans: hash(ans) for ans in set(final_answers)}
                group_answer_types = []
                for i in range(self.n_samples_per_prompt):
                    is_correct = rewards[i] > 0
                    if is_correct:
                        ans_type = 0
                    else:
                        ans_type = answer_to_id[final_answers[i]]
                        if ans_type == 0:
                            ans_type = 1
                    group_answer_types.append(ans_type)
                
                ttrl_metrics["_answer_types"] = group_answer_types
                ttrl_metrics["_consistency_rate"] = [online_consistency_rate] * self.n_samples_per_prompt

                for k, v in ttrl_metrics.items():
                    if k in ["_answer_types", "_consistency_rate"]:
                        all_ttrl_metrics[k].extend(v)
                    else:
                        all_ttrl_metrics[k].append(v)

                # === STAGE 3: Reward assignment with cached data (fixes variable scope bug) ===
                for i in range(self.n_votes_per_prompt):
                    # Use cached values instead of loop-final values
                    v_len = group_cache["valid_response_lengths"][i]
                    d_source = group_cache["data_sources"][i]
                    
                    if i < self.n_samples_per_prompt and v_len > 0:
                        reward_tensor[prompt_i * self.n_samples_per_prompt + i, v_len - 1] = rewards[i]
                    scores[prompt_i * self.n_votes_per_prompt + i] = rewards[i]

                    # === ON-DEMAND PRINT (only when num_examine condition is met) ===
                    if d_source not in already_print_data_sources:
                        already_print_data_sources[d_source] = 0

                    if already_print_data_sources[d_source] < self.num_examine:
                        already_print_data_sources[d_source] += 1
                        # Only decode for debug output (expensive operation)
                        prompt_str = self.tokenizer.decode(group_cache["valid_prompt_indices"][i], skip_special_tokens=False)
                        response_str = self.tokenizer.decode(group_cache["valid_response_indices"][i], skip_special_tokens=False)
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                        print("[score]", rewards[i])

            data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    # Only compute mean for numeric values; non-numeric fields (like _answer_types) are passed through
                    if k.startswith("_") or not all(isinstance(x, (int, float, np.number)) for x in v):
                        ttrl_info[k] = np.array(v) if k.startswith("_") else v
                    else:
                        v_mean = np.mean(v)
                        print(f"[{k}]", v_mean)
                        ttrl_info[k] = v_mean

            return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print_data_sources = {}

            # Group by task to avoid inconsistency errors from mixed tasks
            task_groups = {}
            # Record valid response length for each sample to facilitate reward backfill
            sample_valid_resp_len = {}

            for i in range(len(data)):
                data_item = data[i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]
                sample_valid_resp_len[i] = int(valid_response_length)

                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]

                # Print a few samples
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)

                task_key = self._data_source_to_task(data_source)
                if task_key not in task_groups:
                    task_groups[task_key] = {"indices": [], "outputs": [], "labels": [], "extra": []}
                task_groups[task_key]["indices"].append(i)
                task_groups[task_key]["outputs"].append(response_str)
                task_groups[task_key]["labels"].append(ground_truth)
                task_groups[task_key]["extra"].append(extra_info)

            # Call verification function separately by task and backfill results to corresponding sample positions
            for task_key, group in task_groups.items():
                rewards, verify_extra_info = auto_verify(task_key, group["outputs"], group["labels"], extra_info=group["extra"])
                # Aggregate extra information
                for k, v in verify_extra_info.items():
                    if isinstance(v, list):
                        reward_extra_info[k] += v
                # Backfill reward to corresponding sample's last token position
                for idx_in_group, sample_idx in enumerate(group["indices"]):
                    valid_len = sample_valid_resp_len[sample_idx]
                    reward_tensor[sample_idx, valid_len - 1] = rewards[idx_in_group]

            # Compute TTRL metrics
            all_ttrl_metrics = defaultdict(list)
            prompt_num = len(data) // self.eval_n_samples
            for prompt_i in range(prompt_num):
                group_pred_outputs_ttrl = []
                group_labels_ttrl = []
                group_extra_info_ttrl = []

                task = None

                for i in range(self.eval_n_samples):
                    data_item = data[prompt_i * self.eval_n_samples + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]



                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels_ttrl.append(ground_truth)
                    group_pred_outputs_ttrl.append(response_str)
                    group_extra_info_ttrl.append(extra_info)

                _, ttrl_metrics, _ = test_time_train_metrics(group_pred_outputs_ttrl, group_labels_ttrl, task=task, extra_info=group_extra_info_ttrl)
                
                # === Calculate strategy entropy ===
                current_group_data = data[prompt_i * self.eval_n_samples:(prompt_i + 1) * self.eval_n_samples]
                strategy_entropy = self._compute_strategy_entropy(current_group_data)
                ttrl_metrics["neg_log_likelihood"] = strategy_entropy
                if strategy_entropy > 0:
                    print(f"    Strategy entropy: H_ttrl={strategy_entropy:.3f} (normalized negative log-likelihood)")
                
                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v


            
            return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict=False):

        if self.mode == "train":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported for TTRLRewardManager")

        if return_dict:
            return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                    "ttrl_info": ttrl_info,
                }
        else:
            return reward_tensor