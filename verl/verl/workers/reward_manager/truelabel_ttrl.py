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
True Label TTRL Reward Manager

This manager uses ground truth labels directly for reward computation,
instead of pseudo labels from majority voting. It is designed for
supervised/oracle baseline experiments.

Reward computation:
    reward_i = 1.0 if model_answer_i matches ground_truth else 0.0

This manager also computes and passes answer_types for PASS_GRPO
advantage computation.
"""

from collections import Counter, defaultdict
from functools import partial
import random
import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex
from verl.utils.reward_score.ttrl.qwen.math_grade import grade_answer
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer


class TrueLabelTTRLRewardManager:
    """TTRL Reward Manager using ground truth labels (oracle baseline)."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        reward_fn_key: str = "data_source",
        compute_score=None,
        n_votes_per_prompt: int = 1,
        n_samples_per_prompt: int = 1,
        mode: str = "eval",
        eval_n_samples: int = 1,
        noise_rate: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        self.noise_rate = noise_rate
        self.debug_mode = num_examine > 0

        assert n_votes_per_prompt >= n_samples_per_prompt, (
            f"TTRL requirement: n_votes_per_prompt({n_votes_per_prompt}) >= "
            f"n_samples_per_prompt({n_samples_per_prompt})"
        )

        print(
            "TrueLabelTTRLRewardManager initialized with "
            f"n_votes_per_prompt {n_votes_per_prompt}, "
            f"n_samples_per_prompt {n_samples_per_prompt}, "
            f"eval_n_samples {eval_n_samples}, "
            f"noise_rate {noise_rate}"
        )

    def _data_source_to_task(self, data_source):
        ds = str(data_source)
        if ds in ["MATH-TTT", "AIME-TTT", "AMC-TTT", "AIME25"]:
            return "math"
        if ds in ["GPQA-TTT"]:
            return "gpqa"
        if ds in ["BBEH", "bbeh", "BigBench-Extra-Hard"]:
            return "bbeh"

        dsl = ds.lower()
        if any(key in dsl for key in ["gpqa"]):
            return "gpqa"
        if any(key in dsl for key in ["aime", "math", "amc", "aime25"]):
            return "math"
        if "bbeh" in dsl or "bigbench" in dsl:
            return "bbeh"

        raise NotImplementedError(
            f"Data source {data_source} is not supported for TrueLabelTTRLRewardManager"
        )

    def _extract_final_answers(self, task: str, outputs: list[str]) -> list[str]:
        """Extract final answers from outputs for answer_type computation."""
        extract_fn = partial(extract_answer, data_name=task)
        normalized_outputs = [normalize_latex(x) for x in outputs]
        final_answers = [extract_fn(text) or "<empty>" for text in normalized_outputs]
        return final_answers

    def _decode_data_item(self, data_item):
        prompt_idx = data_item.batch["prompts"]
        prompt_length = prompt_idx.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_idx = prompt_idx[-valid_prompt_length:]

        response_idx = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_idx = response_idx[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
        response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)

        return prompt_str, response_str, valid_response_length

    def _compute_strategy_entropy(self, data_items):
        """Compute normalized negative log-likelihood (strategy entropy) for a group."""
        if not data_items:
            return 0.0

        log_probs_list = []
        response_lengths = []

        for data_item in data_items:
            if not hasattr(data_item, "batch") or "old_log_probs" not in data_item.batch:
                continue

            batch = data_item.batch
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is None or len(attention_mask) <= batch["prompts"].shape[-1]:
                continue

            prompt_length = batch["prompts"].shape[-1]
            response_length_val = attention_mask[prompt_length:].sum().item()
            if response_length_val <= 0:
                continue

            old_log_probs = batch["old_log_probs"]
            if not isinstance(old_log_probs, torch.Tensor) or old_log_probs.numel() == 0:
                continue

            log_probs_length = old_log_probs.shape[-1]
            if log_probs_length == response_length_val:
                response_log_probs = old_log_probs
            elif log_probs_length == prompt_length + response_length_val:
                response_log_probs = old_log_probs[prompt_length:prompt_length + response_length_val]
            elif log_probs_length > response_length_val:
                response_log_probs = old_log_probs[-response_length_val:]
            else:
                continue

            log_probs_list.append(response_log_probs.sum().item())
            response_lengths.append(response_length_val)

        if not log_probs_list:
            return 0.0

        log_probs_array = np.array(log_probs_list)
        response_lengths_array = np.array(response_lengths)
        neg_log_likelihoods = -log_probs_array / response_lengths_array

        return float(np.mean(neg_log_likelihoods))

    def compute_post_ttrl_metrics(self, data: DataProto):
        """Compute post-TTRL training evaluation metrics.
        
        Optimized version: uses batch_decode and single batched auto_verify.
        """
        assert len(data) % self.n_samples_per_prompt == 0
        prompt_num = len(data) // self.n_samples_per_prompt
        total_samples = len(data)
        print(f"Computing post-TTRL metrics, {prompt_num} prompts in total...")

        # Batch decode all responses
        all_response_strs, _, _ = self._batch_decode_responses(data, total_samples)
        
        # Pre-extract metadata
        all_ground_truths = [data[i].non_tensor_batch["reward_model"]["ground_truth"] for i in range(total_samples)]
        all_data_sources = [data[i].non_tensor_batch[self.reward_fn_key] for i in range(total_samples)]
        all_extra_infos = [data[i].non_tensor_batch["extra_info"] for i in range(total_samples)]
        
        # Determine task
        task = self._data_source_to_task(all_data_sources[0])
        
        # Single batched auto_verify for all true_rewards
        all_true_rewards, _ = auto_verify(
            task, all_response_strs, all_ground_truths, extra_info=all_extra_infos
        )

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
            start = prompt_i * self.n_samples_per_prompt
            end = start + self.n_samples_per_prompt
            true_rewards = all_true_rewards[start:end]
            
            post_ttrl_metrics = {
                "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
                f"post_pass@{self.n_samples_per_prompt}": 1.0 if sum(true_rewards) > 0 else 0.0,
            }
            
            for k, v in post_ttrl_metrics.items():
                post_ttrl_metrics_list[k].append(v)

        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    def _batch_decode_responses(self, data: DataProto, total_samples: int):
        """Batch decode all response strings at once, much faster than per-item decode.
        
        Returns:
            Tuple of (response_strs, prompt_strs, valid_response_lengths)
        """
        all_response_ids = []
        all_prompt_ids = []
        valid_response_lengths = []
        
        for i in range(total_samples):
            data_item = data[i]
            prompt_idx = data_item.batch["prompts"]
            prompt_length = prompt_idx.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_idx = prompt_idx[-valid_prompt_length:]
            
            response_idx = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_idx = response_idx[:valid_response_length]
            
            all_response_ids.append(valid_response_idx)
            all_prompt_ids.append(valid_prompt_idx)
            valid_response_lengths.append(valid_response_length)
        
        # Batch decode all at once
        response_strs = self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=False)
        prompt_strs = self.tokenizer.batch_decode(all_prompt_ids, skip_special_tokens=False)
        
        return response_strs, prompt_strs, valid_response_lengths

    def _compute_ttrl_reward(self, data: DataProto, noised_prompt_indices: set = None):
        """Compute rewards using ground truth labels (not pseudo labels).
        
        Optimized version: uses batch_decode and minimizes redundant auto_verify calls.
        
        Args:
            noised_prompt_indices: If provided, for these prompt indices, use a random
                wrong answer as the label instead of true ground truth (test_noise mode).
        """
        if noised_prompt_indices is None:
            noised_prompt_indices = set()
        print("Starting TRUE LABEL reward calculation...")
        if noised_prompt_indices:
            print(f"[test_noise] {len(noised_prompt_indices)} prompts will use noised (wrong) labels")

        reward_extra_info = defaultdict(list)
        ttrl_info = {}

        if len(data) % self.n_votes_per_prompt != 0:
            print(f"WARNING: Data size {len(data)} is not divisible by n_votes_per_prompt {self.n_votes_per_prompt}")
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            ttrl_info["_answer_types"] = np.arange(len(data), dtype=np.int64)
            return reward_tensor, reward_extra_info, ttrl_info

        prompt_num = len(data) // self.n_votes_per_prompt
        total_samples = len(data)
        reward_tensor = torch.zeros_like(
            data.batch["responses"][: prompt_num * self.n_samples_per_prompt], dtype=torch.float32
        )

        # ========== OPTIMIZATION: Batch decode all responses upfront ==========
        all_response_strs, all_prompt_strs, all_valid_resp_lengths = self._batch_decode_responses(data, total_samples)
        
        # Pre-extract metadata (non-tensor fields are cheap to access)
        all_ground_truths = [data[i].non_tensor_batch["reward_model"]["ground_truth"] for i in range(total_samples)]
        all_data_sources = [data[i].non_tensor_batch[self.reward_fn_key] for i in range(total_samples)]
        all_extra_infos = [data[i].non_tensor_batch["extra_info"] for i in range(total_samples)]
        
        # Determine tasks per prompt
        tasks = [self._data_source_to_task(all_data_sources[p * self.n_votes_per_prompt]) for p in range(prompt_num)]

        # ========== OPTIMIZATION: Batch extract final answers for ALL samples ==========
        all_final_answers = self._extract_final_answers(tasks[0], all_response_strs) if prompt_num > 0 else []

        # ========== OPTIMIZATION: Build batched auto_verify inputs ==========
        # Collect all (output, effective_label) pairs for a single batched auto_verify call
        batch_outputs = []
        batch_labels = []
        batch_extra_infos = []
        effective_labels_per_prompt = []  # Track effective label for each prompt
        
        for prompt_i in range(prompt_num):
            start = prompt_i * self.n_votes_per_prompt
            end = start + self.n_votes_per_prompt
            ground_truth = all_ground_truths[start]
            task = tasks[prompt_i]
            
            # Verify all labels in this group match
            group_labels = all_ground_truths[start:end]
            if len(set(group_labels)) != 1:
                print(f"WARNING: Ground truth not unique in prompt group {prompt_i}, using first label")
            
            # For test_noise mode: replace ground truth with a random wrong answer
            effective_label = ground_truth
            if prompt_i in noised_prompt_indices:
                group_answers = all_final_answers[start:end]
                wrong_answers = [a for a in set(group_answers) if a != ground_truth and a is not None and a != ""]
                if wrong_answers:
                    effective_label = random.choice(wrong_answers)
                else:
                    effective_label = "NOISE_WRONG_ANSWER_" + str(random.randint(0, 99999))
                print(f"[test_noise] Prompt {prompt_i}: true='{ground_truth}' -> noised='{effective_label}'")
            
            effective_labels_per_prompt.append(effective_label)
            
            for i in range(self.n_votes_per_prompt):
                batch_outputs.append(all_response_strs[start + i])
                batch_labels.append(effective_label)
                batch_extra_infos.append(all_extra_infos[start + i])
        
        # ========== SINGLE batched auto_verify call for ALL true_rewards ==========
        all_true_rewards, _ = auto_verify(
            tasks[0], batch_outputs, batch_labels, extra_info=batch_extra_infos
        ) if prompt_num > 0 else ([], {})
        
        # ========== Process per-prompt metrics using pre-computed results ==========
        already_print_data_sources = {}
        all_ttrl_metrics = defaultdict(list)
        scores = [0.0] * total_samples
        all_answer_types = []
        all_consistency_rates = []
        all_accuracy_rates = []
        all_label_accuracies = []

        for prompt_i in range(prompt_num):
            start = prompt_i * self.n_votes_per_prompt
            end = start + self.n_votes_per_prompt
            task = tasks[prompt_i]
            ground_truth = all_ground_truths[start]
            effective_label = effective_labels_per_prompt[prompt_i]
            
            # Slice pre-computed results for this prompt group
            true_rewards = all_true_rewards[start:end]
            final_answers = all_final_answers[start:end]
            group_resp_lengths = all_valid_resp_lengths[start:end]
            
            # Compute strategy entropy for this group
            current_group_data = data[start:end]
            strategy_entropy = self._compute_strategy_entropy(current_group_data)
            if self.debug_mode and strategy_entropy > 0:
                print(f"    Strategy entropy: H_ttrl={strategy_entropy:.3f} (normalized negative log-likelihood)")

            ground_truth_ratio = sum(true_rewards) / len(true_rewards)

            # Compute answer types for PASS_GRPO
            for i in range(self.n_votes_per_prompt):
                if true_rewards[i] == 1.0:
                    all_answer_types.append(0)
                else:
                    all_answer_types.append(hash(final_answers[i]))

            # Compute majority vote accuracy (label_accuracy)
            # OPTIMIZATION: derive from final_answers directly instead of calling auto_verify
            counter = Counter(final_answers)
            majority_answer, majority_count = counter.most_common(1)[0]
            majority_ratio = majority_count / len(final_answers)
            
            # Check if majority answer matches ground truth using string comparison
            # via a single-item verify (unavoidable for math equivalence checking)
            label_accuracy = 1.0 if auto_verify(
                task, [majority_answer], [ground_truth],
                extra_info=[all_extra_infos[start]], num_workers=0
            )[0][0] else 0.0

            unique_answers = len(set(final_answers))
            diversity_ratio = unique_answers / len(final_answers) if len(final_answers) > 0 else 0.0

            # OPTIMIZATION: derive majority_rewards from final_answers using grade_answer
            # instead of calling auto_verify again on full response text.
            # grade_answer handles math equivalence (e.g. "0.5" == "\frac{1}{2}")
            majority_rewards = [1.0 if grade_answer(ans, majority_answer) else 0.0 for ans in final_answers]

            # false_positive_rate / false_negative_rate
            n_pseudo_pos = sum(1 for m in majority_rewards if m > 0)
            n_false_pos = sum(1 for m, t in zip(majority_rewards, true_rewards) if m > 0 and t == 0)
            fp_rate = n_false_pos / n_pseudo_pos if n_pseudo_pos > 0 else 0.0

            n_pseudo_neg = sum(1 for m in majority_rewards if m == 0)
            n_false_neg = sum(1 for m, t in zip(majority_rewards, true_rewards) if m == 0 and t > 0)
            fn_rate = n_false_neg / n_pseudo_neg if n_pseudo_neg > 0 else 0.0

            for i in range(self.n_votes_per_prompt):
                all_consistency_rates.append(majority_ratio)
                all_accuracy_rates.append(ground_truth_ratio)
                all_label_accuracies.append(label_accuracy)

            ttrl_metrics = {
                "ground_truth_ratio": ground_truth_ratio,
                f"pass@{self.n_votes_per_prompt}": 1.0 if sum(true_rewards) >= 1 else 0.0,
                "label_accuracy": label_accuracy,
                "majority_ratio": majority_ratio,
                "diversity_ratio": diversity_ratio,
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
                "neg_log_likelihood": strategy_entropy,
            }

            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

            # Assign rewards
            for i in range(self.n_votes_per_prompt):
                current_reward = float(true_rewards[i])
                vlen = group_resp_lengths[i]

                if i < self.n_samples_per_prompt and vlen > 0:
                    reward_tensor[prompt_i * self.n_samples_per_prompt + i, vlen - 1] = current_reward

                scores[start + i] = current_reward

                data_source = all_data_sources[start + i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("\n    === Sample Debug Output ===")
                    print("    [prompt]", all_prompt_strs[start + i])
                    print("    [response]", all_response_strs[start + i])
                    print(f"    [true_reward] {current_reward:.1f}")
                    print(f"    [ground_truth] {ground_truth}")

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
        
        # Store per-sample arrays for PASS_GRPO and diagnostics (only for training samples)
        training_answer_types = []
        training_consistency_rates = []
        training_accuracy_rates = []
        training_label_accuracies = []
        for prompt_i in range(prompt_num):
            for i in range(self.n_samples_per_prompt):
                global_idx = prompt_i * self.n_votes_per_prompt + i
                training_answer_types.append(all_answer_types[global_idx])
                training_consistency_rates.append(all_consistency_rates[global_idx])
                training_accuracy_rates.append(all_accuracy_rates[global_idx])
                training_label_accuracies.append(all_label_accuracies[global_idx])
        
        ttrl_info["_answer_types"] = np.array(training_answer_types)
        ttrl_info["_consistency_rate"] = np.array(training_consistency_rates)
        ttrl_info["_accuracy_rate"] = np.array(training_accuracy_rates)
        ttrl_info["_label_accuracy"] = np.array(training_label_accuracies)
        
        # Store per-prompt label_accuracy for test_minority mode
        ttrl_info["_per_prompt_label_accuracy"] = all_ttrl_metrics.get("label_accuracy", [])

        print("\n=== True Label TTRL Training Metrics Summary ===")
        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v

        return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):
        """Compute evaluation rewards using ground truth labels.
        
        Optimized version: uses batch_decode for all responses upfront.
        """
        print("Starting TRUE LABEL evaluation reward calculation...")

        reward_extra_info = defaultdict(list)
        ttrl_info = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        assert len(data) % self.eval_n_samples == 0
        prompt_num = len(data) // self.eval_n_samples
        total_samples = len(data)
        
        # ========== OPTIMIZATION: Batch decode all responses upfront ==========
        all_response_strs, all_prompt_strs, all_valid_resp_lengths = self._batch_decode_responses(data, total_samples)

        # Pre-extract metadata
        all_ground_truths = [data[i].non_tensor_batch["reward_model"]["ground_truth"] for i in range(total_samples)]
        all_data_sources = [data[i].non_tensor_batch[self.reward_fn_key] for i in range(total_samples)]
        all_extra_infos = [data[i].non_tensor_batch["extra_info"] for i in range(total_samples)]

        # Debug print (using pre-decoded strings)
        for i in range(total_samples):
            data_source = all_data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", all_prompt_strs[i])
                print("[response]", all_response_strs[i])

        # Group by task for batched verification
        task_groups = {}
        for i in range(total_samples):
            task_key = self._data_source_to_task(all_data_sources[i])
            if task_key not in task_groups:
                task_groups[task_key] = {"indices": [], "outputs": [], "labels": [], "extra": []}
            task_groups[task_key]["indices"].append(i)
            task_groups[task_key]["outputs"].append(all_response_strs[i])
            task_groups[task_key]["labels"].append(all_ground_truths[i])
            task_groups[task_key]["extra"].append(all_extra_infos[i])

        # Single batched auto_verify per task group
        all_rewards = [0.0] * total_samples
        for task_key, group in task_groups.items():
            rewards, verify_extra_info = auto_verify(
                task_key, group["outputs"], group["labels"], extra_info=group["extra"]
            )
            for k, v in verify_extra_info.items():
                if isinstance(v, list):
                    reward_extra_info[k] += v
            for idx_in_group, sample_idx in enumerate(group["indices"]):
                all_rewards[sample_idx] = float(rewards[idx_in_group])
                vlen = all_valid_resp_lengths[sample_idx]
                if vlen > 0:
                    reward_tensor[sample_idx, vlen - 1] = float(rewards[idx_in_group])

        # Compute TTRL metrics for logging (using pre-decoded and pre-computed rewards)
        all_ttrl_metrics = defaultdict(list)
        for prompt_i in range(prompt_num):
            start = prompt_i * self.eval_n_samples
            end = start + self.eval_n_samples
            
            true_rewards = all_rewards[start:end]
            
            # Compute strategy entropy for eval group
            current_group_data = data[start:end]
            strategy_entropy = self._compute_strategy_entropy(current_group_data)

            ttrl_metrics = {
                "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
                f"pass@{self.eval_n_samples}": 1.0 if sum(true_rewards) >= 1 else 0.0,
                "neg_log_likelihood": strategy_entropy,
            }

            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v

        return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict: bool = False):
        print("\n" + "=" * 50)
        print("TrueLabelTTRLRewardManager execution started")
        print(f"Mode: {self.mode}")
        print(f"Data size: {len(data)}")
        print("=" * 50)

        if self.mode in ("train", "test_minority"):
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
            if self.mode == "test_minority":
                # Build per-sample zero_advantage_mask:
                # For each prompt, if true label != majority answer (label_accuracy == 0),
                # set mask=1 for all responses of that prompt (advantage will be zeroed).
                prompt_num = len(data) // self.n_votes_per_prompt
                zero_mask = np.zeros(prompt_num * self.n_samples_per_prompt, dtype=np.float32)
                # Retrieve per-prompt label_accuracy from ttrl_info
                per_prompt_label_acc = ttrl_info.get("_per_prompt_label_accuracy", [])
                for prompt_i in range(prompt_num):
                    if prompt_i < len(per_prompt_label_acc) and per_prompt_label_acc[prompt_i] == 0.0:
                        for j in range(self.n_samples_per_prompt):
                            zero_mask[prompt_i * self.n_samples_per_prompt + j] = 1.0
                ttrl_info["_zero_advantage_mask"] = zero_mask
                n_zeroed_prompts = int(sum(1 for acc in per_prompt_label_acc if acc == 0.0))
                n_zeroed_samples = int(zero_mask.sum())
                print(f"[test_minority] Zeroing advantage for {n_zeroed_prompts}/{prompt_num} prompts "
                      f"({n_zeroed_samples}/{len(zero_mask)} samples) where true label != majority")
        elif self.mode == "test_noise":
            # Randomly select noise_rate fraction of prompts and use wrong labels
            prompt_num = len(data) // self.n_votes_per_prompt
            n_noised = max(1, int(prompt_num * self.noise_rate))
            noised_indices = set(random.sample(range(prompt_num), min(n_noised, prompt_num)))
            print(f"[test_noise] Noising {len(noised_indices)}/{prompt_num} prompts "
                  f"(noise_rate={self.noise_rate:.2f})")
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(
                data, noised_prompt_indices=noised_indices
            )
            ttrl_info["noise_rate_actual"] = len(noised_indices) / prompt_num
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "ttrl_info": ttrl_info,
            }
        return reward_tensor
