from functools import partial

from latex2sympy2 import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm

from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex


import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Default number of workers for parallel extraction
_DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 1)
_MIN_PARALLEL_BATCH = 16

def _extract_single(args):
    """Worker function for parallel extraction."""
    task, output_text = args
    task2extract_fn = {
        "math": partial(extract_answer, data_name="math"),
        "gpqa": partial(extract_answer, data_name="gpqa"),
    }
    extract_fn = task2extract_fn[task]
    cleaned = normalize_latex(output_text)
    return extract_fn(cleaned)

def auto_extract(task, all_outputs, extra_info=None, num_workers=None):
    if num_workers is None:
        num_workers = _DEFAULT_NUM_WORKERS

    n = len(all_outputs)
    model_answers = [None] * n

    # --- Parallel path ---
    if num_workers > 1 and n >= _MIN_PARALLEL_BATCH:
        work_items = [(task, output) for output in all_outputs]
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_extract_single, item): idx for idx, item in enumerate(work_items)}
                for future in as_completed(futures):
                    idx = futures[future]
                    model_answers[idx] = future.result()
        except Exception as e:
            print(f"[auto_extract] Parallel execution failed ({e}), falling back to serial")
            model_answers = _extract_serial(task, all_outputs)
    else:
        model_answers = _extract_serial(task, all_outputs)

    return [answer for answer in model_answers if answer is not None]

def _extract_serial(task, all_outputs):
    task2extract_fn = {
        "math": partial(extract_answer, data_name=task),
        "gpqa": partial(extract_answer, data_name=task),
    }
    assert task in task2extract_fn, f"{task} not in {list(task2extract_fn.keys())}"
    extract_fn = task2extract_fn[task]

    cleaned_outputs = [normalize_latex(x) for x in all_outputs]
    return [extract_fn(generated_text) for generated_text in cleaned_outputs]