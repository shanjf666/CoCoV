import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.qwen.qwen_eval import (qwen_reward_fn,
                                                         qwen_reward_fn_gpqa,
                                                         simplerl_reward_fn)
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex


# Module-level mapping so worker processes can look up verify functions by name
_TASK2VERIFY = {
    "math": qwen_reward_fn,
    "simplerl_math": simplerl_reward_fn,
    "gpqa": qwen_reward_fn_gpqa,
}

# Default number of workers for parallel verification
_DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 1)

# Minimum batch size to trigger multiprocessing (below this, serial is faster)
_MIN_PARALLEL_BATCH = 16


def _verify_single(args):
    """Worker function for parallel verification. Accepts a tuple to be picklable."""
    task, output_text, label = args
    cleaned = normalize_latex(output_text)
    verify_fn = _TASK2VERIFY[task]
    return verify_fn(cleaned, label)


def verify_many(task, all_outputs, all_labels, num_workers=None):
    """Verify model outputs against labels without collecting extra prediction metadata."""
    assert task in _TASK2VERIFY, f"{task} not in {list(_TASK2VERIFY.keys())}"

    if num_workers is None:
        num_workers = _DEFAULT_NUM_WORKERS

    n = len(all_outputs)

    # --- Parallel path ---
    if num_workers > 1 and n >= _MIN_PARALLEL_BATCH:
        work_items = [(task, output, label) for output, label in zip(all_outputs, all_labels)]
        rewards = [0.0] * n
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_verify_single, item): idx for idx, item in enumerate(work_items)}
                for future in as_completed(futures):
                    idx = futures[future]
                    rewards[idx] = future.result()
        except Exception as e:
            # Fallback to serial if multiprocessing fails (e.g. pickling issues)
            print(f"[auto_verify] Parallel execution failed ({e}), falling back to serial")
            rewards = _verify_serial(task, all_outputs, all_labels)
    else:
        # --- Serial path (small batches or explicitly requested) ---
        rewards = _verify_serial(task, all_outputs, all_labels)

    return rewards


def auto_verify(task, all_outputs, all_labels, extra_info=None, num_workers=None):
    """Verify model outputs against labels, optionally using multiprocessing.
    
    Args:
        task: Task name (e.g. "math", "gpqa")
        all_outputs: List of model output strings
        all_labels: List of ground truth label strings
        extra_info: Optional extra info for extraction
        num_workers: Number of parallel workers. 0 or 1 means serial.
                     None means auto-detect.
    
    Returns:
        Tuple of (rewards_list, verify_extra_info_dict)
    """
    verify_extra_info = defaultdict(list)
    rewards = verify_many(task, all_outputs, all_labels, num_workers=num_workers)

    verify_extra_info["acc"] = rewards
    verify_extra_info["pred"] = auto_extract(task, all_outputs, extra_info=extra_info)

    return rewards, verify_extra_info


def _verify_serial(task, all_outputs, all_labels):
    """Serial verification fallback."""
    verify_fn = _TASK2VERIFY[task]
    cleaned_outputs = [normalize_latex(x) for x in all_outputs]
    return [verify_fn(output, label) for output, label in zip(cleaned_outputs, all_labels)]