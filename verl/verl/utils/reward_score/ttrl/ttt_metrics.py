from collections import Counter
from typing import List

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import verify_many


def _get_majority_answer(model_answers):
    counter = Counter(model_answers)
    if counter:
        return counter.most_common(1)[0]
    return None, 0


def _verify_single_label(task, label, ground_truth):
    if label is None:
        return 0.0
    return 1.0 if verify_many(task, [label], [ground_truth], num_workers=0)[0] else 0.0


def _verify_batch(task, solutions, label):
    if label is None:
        return [0.0] * len(solutions)
    return verify_many(task, solutions, [label] * len(solutions))


def _match_rate(predictions, targets):
    if not predictions:
        return 0.0
    return sum(1 if pred == true else 0 for pred, true in zip(predictions, targets)) / len(predictions)


def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math", extra_info=None,
    verified_label=None,
    model_answers=None,
    return_details=False):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    if model_answers is None:
        model_answers = auto_extract(task, solutions, extra_info=extra_info)
    estimated_label, majority_count = _get_majority_answer(model_answers)
    majority_ratio = majority_count / len(solutions) if solutions else 0.0
    majority_hit = _verify_single_label(task, estimated_label, ground_truth)

    selected_label = verified_label if verified_label is not None else estimated_label
    hit_rate = _verify_single_label(task, selected_label, ground_truth)
    rewards = _verify_batch(task, solutions, selected_label)
    true_rewards = _verify_batch(task, solutions, ground_truth)
    rewards_hit_rate = _match_rate(rewards, true_rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards) if true_rewards else 0.0,
        "majority_voting_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        f"pass@{len(solutions)}": 1.0 if true_rewards and sum(true_rewards) >= 1 else 0.0,
        "neg_log_likelihood": 0.0,  # Policy entropy placeholder, will be calculated in semantic_novelty.py
    }
    details = {}
    if return_details:
        details = {
            "model_answers": model_answers,
            "majority_answer": estimated_label,
            "selected_label": selected_label,
            "majority_count": majority_count,
            "majority_hit": majority_hit,
            "true_rewards": true_rewards,
        }
    return rewards, ttrl_metrics, details

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]
    _ = extra_info

    true_rewards = verify_many(task, solutions, [ground_truth] * len(solutions))

    # Compare pred_rewards with true_rewards to calculate reward hit rate
    rewards_hit_rate = 0.0
    if pred_rewards:
        rewards_hit_rate = sum(
            1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
        ) / len(pred_rewards)

    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards) if true_rewards else 0.0,
        f"post_pass@{len(solutions)}": 1.0 if true_rewards and sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics
