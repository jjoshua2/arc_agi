import json
import random
import os
import time
import traceback
import typing as T
from copy import deepcopy
from enum import Enum
import asyncio
from types import SimpleNamespace
import pickle
import hashlib

import httpx
import numpy as np
import redis.asyncio as redis
from devtools import debug
from pydantic import BaseModel, TypeAdapter
from tqdm.asyncio import tqdm_asyncio
import jax.numpy as jnp
import jax
import chex
import numpy as onp

from src import PLOT, logfire
from src.models import (
    GRID,
    Attempt,
    AttemptEdge,
    Challenge,
    FixAttemptConfig,
    Prompt,
    RootAttemptConfig,
    prompt_map,
    random_string,
    Library,
    Primitive,
)
from src.prompts.examples import (
    GRID_CHANGE_PROMPT_EXAMPLE_1,
    GRID_SAME_PROMPT_EXAMPLE_1,
    example_1_grid_change_challenge_id,
    example_1_reasoning_grid_change,
    example_1_same_grid_challenge_id,
    example_1_same_grid_reasoning,
    example_2_challenge_id,
    example_2_reasoning_grid_same,
    example_3_challenge_id,
    example_3_reasoning_grid_same,
    example_7_grid_change_challenge_id,
    example_7_reasoning_grid_change_bad_colors,
)
from src.render_legacy import grid_to_base64_png_oai_content
from src.reps import array_to_str, grid_diffs_to_ascii, grid_to_ascii
from src.run_python import run_python_transform_sync
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from src.perf import perf_add, perf_get, perf_clear

# Track top two attempts chosen per challenge for later export
_TOP_TWO_ATTEMPTS: dict[str, list["Attempt"]] = {}

def set_top_two_attempts_for(challenge_id: str, attempts: list["Attempt"]) -> None:
    try:
        if attempts is None:
            return
        _TOP_TWO_ATTEMPTS[challenge_id] = attempts[:2]
    except Exception:
        pass

def get_top_two_attempts_for(challenge_id: str) -> list["Attempt"] | None:
    return _TOP_TWO_ATTEMPTS.get(challenge_id)

# ---- Transform result cache (opt-in) ----
_TRANSFORM_CACHE_ENABLED = os.environ.get("SUBMISSION_TRANSFORM_CACHE", "1") == "1"
_TRANSFORM_CACHE_PATH = os.environ.get("TRANSFORM_CACHE_PATH", "transforms_cache.pkl")
_TRANSFORM_CACHE_CLEAR = os.environ.get("SUBMISSION_TRANSFORM_CACHE_CLEAR", "0") == "1"
_transform_cache: dict[str, list] = {}


def _hash_obj(obj: T.Any) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _transform_cache_key(*, code: str, grid_lists: list) -> str:
    h = hashlib.sha256()
    h.update(code.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(_hash_obj(grid_lists).encode("utf-8"))
    return h.hexdigest()


def load_transform_cache() -> None:
    global _transform_cache
    if not _TRANSFORM_CACHE_ENABLED:
        return
    if _TRANSFORM_CACHE_CLEAR:
        _transform_cache = {}
        return
    try:
        if os.path.exists(_TRANSFORM_CACHE_PATH):
            with open(_TRANSFORM_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                _transform_cache = data
                logfire.debug(f"TRANSFORM_CACHE: loaded {len(_transform_cache)} entries from {_TRANSFORM_CACHE_PATH}")
    except Exception as e:
        logfire.debug(f"TRANSFORM_CACHE: failed to load: {e}")
        _transform_cache = {}


def save_transform_cache() -> None:
    if not _TRANSFORM_CACHE_ENABLED:
        return
    try:
        with open(_TRANSFORM_CACHE_PATH, "wb") as f:
            pickle.dump(_transform_cache, f)
        logfire.debug(f"TRANSFORM_CACHE: saved {len(_transform_cache)} entries to {_TRANSFORM_CACHE_PATH}")
    except Exception as e:
        logfire.debug(f"TRANSFORM_CACHE: failed to save: {e}")


def run_python_transform_sync_cached(*, code: str, grid_lists: list, timeout: int, raise_exception: bool):
    if not _TRANSFORM_CACHE_ENABLED:
        return run_python_transform_sync(code=code, grid_lists=grid_lists, timeout=timeout, raise_exception=raise_exception)
    key = _transform_cache_key(code=code, grid_lists=grid_lists)
    if key in _transform_cache:
        res = _transform_cache[key]
        return SimpleNamespace(transform_results=res, latency_ms=0.0)
    out = run_python_transform_sync(code=code, grid_lists=grid_lists, timeout=timeout, raise_exception=raise_exception)
    try:
        if out and getattr(out, "transform_results", None):
            _transform_cache[key] = out.transform_results
    except Exception:
        pass
    return out

# ---- Bounded, non-blocking wrappers for synchronous transform execution ----
# Default to 6 concurrent threads (balanced for GIL-limited CPU work)
try:
    _TRANSFORM_EXEC_CONCURRENCY = max(1, int(os.environ.get("SUBMISSION_TRANSFORM_CONCURRENCY", "6")))
except Exception:
    _TRANSFORM_EXEC_CONCURRENCY = 6

_transform_exec_sem = asyncio.Semaphore(_TRANSFORM_EXEC_CONCURRENCY)

async def run_transform_cached_async(*, code: str, grid_lists: list, timeout: int, raise_exception: bool):
    async with _transform_exec_sem:
        return await asyncio.to_thread(
            run_python_transform_sync_cached,
            code=code,
            grid_lists=grid_lists,
            timeout=timeout,
            raise_exception=raise_exception,
        )

async def run_transform_sync_async(*, code: str, grid_lists: list, timeout: int, raise_exception: bool):
    async with _transform_exec_sem:
        return await asyncio.to_thread(
            run_python_transform_sync,
            code=code,
            grid_lists=grid_lists,
            timeout=timeout,
            raise_exception=raise_exception,
        )

# auto-load cache on import if enabled
try:
    load_transform_cache()
except Exception:
    pass
from lpn.src.evaluator import Evaluator
from lpn.src.models.lpn import LPN


class TqdmLogfire:
    """File-like class redirecting tqdm progress bar to given logging logger."""

    def __init__(self):
        pass

    def write(self, msg: str) -> None:
        logfire.debug(msg.lstrip("\r"))

    def flush(self) -> None:
        pass


def chunk_list(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def content_blocks_from_matrix(
    *,
    matrix: GRID,
    _label: str,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, str]]:
    matrix = deepcopy(matrix)
    grid = np.array(matrix)
    x, y = grid.shape
    messages = [
        {"type": "text", "text": _label},
        {"type": "text", "text": f"Shape: {x} by {y}\n\n"},
    ]
    if include_image:
        messages.append(grid_to_base64_png_oai_content(grid=grid))
    if use_ascii:
        messages.append(
            {
                "type": "text",
                "text": f"ASCII representation:\n\n{grid_to_ascii(grid=grid, separator='|', spreadsheet_ascii=False)}\n\n",
            }
        )
    if use_array:
        messages.append({"type": "text", "text": array_to_str(grid=matrix)})
    return messages


def content_from_challenge(
    challenge: Challenge,
    include_diffs: bool,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, T.Any]]:
    content = []
    for i, train in enumerate(challenge.train):
        example_number = i + 1
        # add input blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.input,
                _label=f"# Example {example_number}\n\n## Input {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        # add output blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.output,
                _label=f"## Output {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        if not does_grid_change_shape(challenge=challenge) and include_diffs:
            content.append(
                {
                    "type": "text",
                    "text": f"## Color changes between the Input and Output ASCII representation:\n\n"
                    f"{grid_diffs_to_ascii(grid_input=np.array(train.input), grid_output=np.array(train.output), separator='|')}\n\n",
                }
            )

    # TODO for now, only do the first test... Will have to treat these multi tests as multiple examples later
    # assert len(challenge.test) == 1
    content.extend(
        content_blocks_from_matrix(
            matrix=challenge.test[0].input,
            _label="# Additional input\n\n",
            include_image=include_image,
            use_ascii=use_ascii,
            use_array=use_array,
        )
    )

    return content


def does_grid_change_shape(challenge: Challenge) -> bool:
    for train in challenge.train:
        if np.array(train.input).shape != np.array(train.output).shape:
            return True
    return False


def challenge_to_messages(
    *,
    challenge: Challenge,
    add_examples: bool,
    use_cache_control: bool = True,
    include_diffs: bool,
    prompt: Prompt,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, str]]:
    # first, is example same grid size?
    grid_change_shape = does_grid_change_shape(challenge)
    # debug(grid_change_shape)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_map[prompt]}]}
    ]
    if add_examples:
        # Lazy import to avoid circular dependency during module import time.
        from src.data import training_challenges
        if grid_change_shape:
            # messages.extend(GRID_CHANGE_PROMPT_EXAMPLE_1)
            example_1_grid_change_prompt = content_from_challenge(
                challenge=training_challenges[example_1_grid_change_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            example_7_grid_change_prompt = content_from_challenge(
                challenge=training_challenges[example_7_grid_change_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": example_1_grid_change_prompt,
                    },
                    {"role": "assistant", "content": [ {"type": "text", "text": example_1_reasoning_grid_change} ]},
                ]
            )
        else:
            example_1_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_1_same_grid_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )

            # ADDING OTHER EXAMPLE!
            example_2_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_2_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            example_3_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_3_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": example_1_grid_same_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": [ {"type": "text", "text": example_1_same_grid_reasoning} ],
                    },
                ]
            )

        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Great work! Now I will give you another puzzle to solve just like that one.",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Great, please give me the next puzzle.",
                        }
                    ],
                },
            ]
        )
    if use_cache_control:
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    content = content_from_challenge(
        challenge=challenge,
        include_diffs=include_diffs,
        include_image=include_image,
        use_ascii=use_ascii,
        use_array=use_array,
    )
    if use_cache_control:
        content[-1]["cache_control"] = {"type": "ephemeral"}
    messages.append({"role": "user", "content": content})
    return messages


def eval_attempts(
    attempts: list[Attempt],
    config: RootAttemptConfig | FixAttemptConfig,
    plot: bool,
    time_took_ms: float,
) -> None:
    if not attempts:
        return None

    for attempt in attempts:
        # debug(attempt.train_accuracy, attempt.test_accuracy)
        if plot:
            try:
                start = time.time()
                attempt.plot(ignore_fixing=True)
                took = time.time() - start
                if took > 0.5:
                    logfire.debug(f"TOOK {took} SECONDS TO PLOT")
            except Exception as e:
                logfire.debug(f"FAILED TO PLOT: {e}")

    # get total accuracies
    avg_train_accuracy = sum(attempt.train_accuracy for attempt in attempts) / len(
        attempts
    )
    avg_test_accuracy = sum(attempt.test_accuracy for attempt in attempts) / len(
        attempts
    )
    total_cost = sum(attempt.cost_cents for attempt in attempts)
    total_runs = len(attempts)
    total_correct = len(
        [a for a in attempts if a.test_accuracy == 1 and a.train_accuracy == 1]
    )
    debug_d = {
        "challenge_id": attempts[0].challenge.id,
        "total_runs": total_runs,
        "total_correct": total_correct,
        "avg_train_accuracy": avg_train_accuracy,
        "avg_test_accuracy": avg_test_accuracy,
        "total_cost": total_cost,
        "prompt_config": config.prompt_config,
        "llm_config": config.llm_config,
        "time_took_ms": round(time_took_ms, 2),
    }

    logfire.debug(f"[{attempts[0].challenge.id}] eval", **debug_d)
    print(
        f"[{attempts[0].challenge.id}] finished processing node [{attempts[0].config.llm_config.model.value}]: {total_runs} attempts, {round(avg_train_accuracy * 100, 2)}% train accuracy, {round(avg_test_accuracy * 100, 2)}% test accuracy, ${round(total_cost / 100, 2)}, {round(time_took_ms / 1000, 2)} secs",
    )
    logfire.debug(f"[{attempts[0].challenge.id}] finished processing node [{attempts[0].config.llm_config.model.value}]: {total_runs} attempts, {round(avg_train_accuracy * 100, 2)}% train accuracy, {round(avg_test_accuracy * 100, 2)}% test accuracy, ${round(total_cost / 100, 2)}, {round(time_took_ms / 1000, 2)} secs")

def percent_right_from_grids(train_output: GRID, train_attempt: GRID) -> float:
    try:
        if len(train_output) != len(train_attempt):
            return 0
        if len(train_output[0]) != len(train_attempt[0]):
            return 0

        num_right = 0
        rows = len(train_output)
        cols = len(train_output[0])

        for row in range(rows):
            for col in range(cols):
                if train_output[row][col] == train_attempt[row][col]:
                    num_right += 1
        return num_right / (rows * cols)
    except Exception as e:
        logfire.debug(f"in percent right from grids: {e=}")
        return 0

# ---- Process-backed scoring for CPU-heavy loops (enabled by default to bypass GIL) ----
try:
    _SCORE_USE_PROCESS = os.environ.get("SUBMISSION_SCORE_PROCESS", "1") == "1"
except Exception:
    _SCORE_USE_PROCESS = True

try:
    _SCORE_WORKERS = max(1, int(os.environ.get("SUBMISSION_SCORE_WORKERS", "4")))
except Exception:
    _SCORE_WORKERS = 4

# Separate thread-pool sizing for scoring (defaults to 6, but processes used by default)
try:
    _SCORE_THREADS = max(1, int(os.environ.get("SUBMISSION_SCORE_THREADS", "6")))
except Exception:
    _SCORE_THREADS = 6

_score_executor: ProcessPoolExecutor | None = None
_score_thread_executor: ThreadPoolExecutor | None = None
_score_sem = asyncio.Semaphore(_SCORE_THREADS)

def _get_score_executor() -> ProcessPoolExecutor:
    global _score_executor
    if _score_executor is None:
        # Create lazily to avoid Windows spawn overhead unless requested
        _score_executor = ProcessPoolExecutor(max_workers=_SCORE_WORKERS)
    return _score_executor

def _get_score_thread_executor() -> ThreadPoolExecutor:
    global _score_thread_executor
    if _score_thread_executor is None:
        _score_thread_executor = ThreadPoolExecutor(max_workers=_SCORE_THREADS)
    return _score_thread_executor

def _compute_primitive_score_sync(challenge_train: list, transformed_grids: list) -> tuple[float, float]:
    # Compute num_correct and average percent_right across training examples
    num_correct = 0
    avg_right_lst: list[float] = []
    for idx, train in enumerate(challenge_train):
        if train.output == transformed_grids[idx]:
            num_correct += 1
        train_accuracy = percent_right_from_grids(train.output, transformed_grids[idx])
        avg_right_lst.append(train_accuracy)
    secondary_score = sum(avg_right_lst) / len(avg_right_lst) if avg_right_lst else 0.0
    return float(num_correct), float(secondary_score)

async def compute_primitive_score_async(challenge_train: list, transformed_grids: list) -> tuple[float, float]:
    if not transformed_grids:
        return 0.0, 0.0
    if _SCORE_USE_PROCESS:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_get_score_executor(), _compute_primitive_score_sync, challenge_train, transformed_grids)
    # Default to bounded thread offload; avoids event-loop blocking and caps concurrency to 4 by default
    loop = asyncio.get_running_loop()
    async with _score_sem:
        return await loop.run_in_executor(_get_score_thread_executor(), _compute_primitive_score_sync, challenge_train, transformed_grids)

# -------- Two-pass primitive scoring (first-pass single-example, second-pass full) --------
_FIRST_PASS_SCORE_CACHE: dict[str, tuple[float, float]] = {}
_first_pass_executor: ProcessPoolExecutor | None = None

def _fp_cache_key(challenge_id: str, primitive_id: str, example_index: int) -> str:
    return f"{challenge_id}:{primitive_id}:{example_index}"

def _get_first_pass_executor() -> ProcessPoolExecutor:
    global _first_pass_executor
    if _first_pass_executor is None:
        # Use 4 processes for true parallelism on primitive evaluation
        _first_pass_executor = ProcessPoolExecutor(max_workers=4)
    return _first_pass_executor

def _evaluate_primitives_chunk_sync(primitives_chunk: list, challenge_dict: dict, example_index: int) -> list[tuple[int, float, float]]:
    """Process a chunk of primitives in a separate process (bypasses GIL).
    Returns list of (original_index, num_correct, accuracy) for each primitive.
    """
    from src.run_python import run_python_transform_sync
    from copy import deepcopy
    import os
    
    results = []
    train_example = challenge_dict['train'][example_index]
    
    for orig_idx, primitive_code in primitives_chunk:
        try:
            tr = run_python_transform_sync(
                code=primitive_code,
                grid_lists=[deepcopy(train_example['input'])],
                timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
                raise_exception=False,
            )
            if tr and tr.transform_results:
                pred = tr.transform_results[0]
                # Simple grid comparison
                expected = train_example['output']
                num_correct = 1.0 if pred == expected else 0.0
                
                # Calculate accuracy
                if len(expected) != len(pred) or (expected and len(expected[0]) != len(pred[0])):
                    acc = 0.0
                else:
                    total_cells = len(expected) * (len(expected[0]) if expected else 0)
                    if total_cells == 0:
                        acc = 0.0
                    else:
                        correct_cells = sum(
                            1 for r in range(len(expected)) for c in range(len(expected[0]))
                            if expected[r][c] == pred[r][c]
                        )
                        acc = correct_cells / total_cells
                        
                results.append((orig_idx, num_correct, float(acc)))
            else:
                results.append((orig_idx, 0.0, 0.0))
        except Exception:
            results.append((orig_idx, 0.0, 0.0))
    
    return results

async def _evaluate_primitive_single_example_async(
    primitive: Primitive,
    challenge: Challenge,
    example_index: int,
) -> tuple[float, float]:
    # Returns (num_correct_on_example, percent_right_on_example)
    key = _fp_cache_key(challenge.id, getattr(primitive, "id", "?"), example_index)
    if key in _FIRST_PASS_SCORE_CACHE:
        return _FIRST_PASS_SCORE_CACHE[key]
    try:
        train_ex = challenge.train[example_index]
        tr = await run_transform_cached_async(
            code=primitive.python_code_str,
            grid_lists=[deepcopy(train_ex.input)],
            timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
            raise_exception=False,
        )
        if tr and tr.transform_results:
            pred = tr.transform_results[0]
            acc = percent_right_from_grids(train_ex.output, pred)
            num_correct = 1.0 if pred == train_ex.output else 0.0
            res = (num_correct, float(acc))
            _FIRST_PASS_SCORE_CACHE[key] = res
            return res
        _FIRST_PASS_SCORE_CACHE[key] = (0.0, 0.0)
        return 0.0, 0.0
    except Exception:
        _FIRST_PASS_SCORE_CACHE[key] = (0.0, 0.0)
        return 0.0, 0.0

async def _two_pass_select_primitives_async(
    library: Library,
    challenge: Challenge,
    k_top: int,
    challenge_primitive_scores: dict[str, dict[str, tuple[float, float]]] | None,
) -> list[Primitive]:
    # Config
    try:
        fp_top_k = max(1, int(os.environ.get("SUBMISSION_FIRST_PASS_TOP_K", "50")))
    except Exception:
        fp_top_k = 50
    try:
        sp_batch = max(1, int(os.environ.get("SUBMISSION_SECOND_PASS_BATCH_SIZE", "150")))
    except Exception:
        sp_batch = 150

    if not library or not library.primitives:
        return []

    # First pass on example_index=0 across all primitives (process-based for true parallelism)
    example_index = 0
    prims = list(library.primitives)
    
    # Check cache first to avoid re-evaluation
    cached_results: dict[int, tuple[float, float]] = {}
    uncached_indices: list[int] = []
    for i, p in enumerate(prims):
        cache_key = _fp_cache_key(challenge.id, getattr(p, "id", "?"), example_index)
        if cache_key in _FIRST_PASS_SCORE_CACHE:
            cached_results[i] = _FIRST_PASS_SCORE_CACHE[cache_key]
        else:
            uncached_indices.append(i)
    
    first_pass_scores: list[tuple[int, float, float]] = []
    
    # Add cached results
    for i, (nc, acc) in cached_results.items():
        first_pass_scores.append((i, float(nc), float(acc)))
    
    if uncached_indices:
        print(f"[{challenge.id}] First pass: evaluating {len(uncached_indices)} primitives across 4 processes ({len(cached_results)} cached)")
        logfire.debug(f"[{challenge.id}] Process-based first pass: {len(uncached_indices)} uncached, {len(cached_results)} cached")
        
        # Divide uncached primitives across 4 processes (~500 each for 2000 total)
        num_processes = 4
        chunk_size = max(1, len(uncached_indices) // num_processes)
        chunks = []
        
        # Serialize challenge for process communication
        challenge_dict = {
            'train': [{'input': t.input, 'output': t.output} for t in challenge.train],
            'id': challenge.id
        }
        
        # Create chunks of (original_index, primitive_code)
        for i in range(0, len(uncached_indices), chunk_size):
            chunk_indices = uncached_indices[i:i+chunk_size]
            chunk_data = [(idx, prims[idx].python_code_str) for idx in chunk_indices]
            chunks.append(chunk_data)
        
        # Process chunks in parallel using ProcessPoolExecutor
        loop = asyncio.get_running_loop()
        executor = _get_first_pass_executor()
        
        chunk_futures = [
            loop.run_in_executor(
                executor, 
                _evaluate_primitives_chunk_sync, 
                chunk, 
                challenge_dict, 
                example_index
            ) for chunk in chunks
        ]
        
        # Wait for all processes to complete
        chunk_results = await asyncio.gather(*chunk_futures, return_exceptions=True)
        
        # Collect results and update cache
        for chunk_result in chunk_results:
            if isinstance(chunk_result, Exception):
                print(f"[{challenge.id}] Process chunk failed: {chunk_result}")
                continue
            for orig_idx, nc, acc in chunk_result:
                first_pass_scores.append((orig_idx, float(nc), float(acc)))
                # Cache the result
                cache_key = _fp_cache_key(challenge.id, getattr(prims[orig_idx], "id", "?"), example_index)
                _FIRST_PASS_SCORE_CACHE[cache_key] = (float(nc), float(acc))

    # Identify perfect-on-first-example primitives
    perfect_on_first: list[int] = [idx for idx, nc, acc in first_pass_scores if nc >= 1.0 and acc >= 1.0]
    
    # Early check: if any perfect-on-first, immediately test them on ALL examples
    if perfect_on_first:
        print(f"[{challenge.id}] Found {len(perfect_on_first)} perfect-on-first primitives; checking if any solve all examples...")
        logfire.debug(f"[{challenge.id}] Found {len(perfect_on_first)} perfect-on-first primitives")
        perfect_candidates = [prims[idx] for idx in perfect_on_first]
        
        # Evaluate perfect candidates on all training examples
        perfect_full_scores: list[tuple[int, float, float]] = []  # (idx_in_prims, num_correct, secondary_score)
        total_tr = len(challenge.train)
        
        for idx in perfect_on_first:
            p = prims[idx]
            pid = getattr(p, "id", "?")
            # Check cache first
            if challenge_primitive_scores is not None and challenge.id in challenge_primitive_scores and pid in challenge_primitive_scores[challenge.id]:
                nc, sc = challenge_primitive_scores[challenge.id][pid]
                perfect_full_scores.append((idx, float(nc), float(sc)))
            else:
                # Evaluate
                try:
                    nc, sc = await evaluate_primitive_weighed_async(p, challenge, challenge_primitive_scores)
                    perfect_full_scores.append((idx, float(nc), float(sc)))
                except Exception:
                    perfect_full_scores.append((idx, 0.0, 0.0))
        
        # Check if any achieved perfect on ALL training examples
        perfect_all = [(idx, nc, sc) for idx, nc, sc in perfect_full_scores if nc >= float(total_tr)]
        if perfect_all:
            print(f"[{challenge.id}] Found {len(perfect_all)} primitives that solve all training examples!")
            logfire.debug(f"[{challenge.id}] Early-exit: {len(perfect_all)} primitives solve all training examples")
            # Return the best perfect ones (softmax sample from them)
            scores = [nc + sc for _, nc, sc in perfect_all]
            if not scores:
                # Shouldn't happen, but fallback
                return [prims[perfect_all[0][0]]]
            scores_arr = np.array(scores)
            exp_scores = np.exp(scores_arr - np.max(scores_arr))
            probabilities = exp_scores / np.clip(exp_scores.sum(), 1e-12, None)
            selected_positions = _safe_sample_indices(len(perfect_all), min(k_top, len(perfect_all)), probabilities, ctx=f"two_pass_perfect:{challenge.id}")
            return [prims[perfect_all[pos][0]] for pos in selected_positions]

    # No perfect solution found; proceed with top-K selection from all primitives
    # Rank by (num_correct + acc)
    def _score_tuple(tup: tuple[int, float, float]) -> tuple[float, float]:
        _, nc, acc = tup
        return (nc + acc, acc)
    
    first_pass_scores.sort(key=_score_tuple, reverse=True)
    top_indices: list[int] = [idx for idx, _, _ in first_pass_scores[:fp_top_k]]
    
    # Deduplicate
    seen = set()
    candidates: list[Primitive] = []
    for idx in top_indices:
        if idx in seen:
            continue
        seen.add(idx)
        candidates.append(prims[idx])
    
    if not candidates:
        return []

    # Second pass: evaluate shortlisted candidates across all training examples
    # NOTE: if any perfect-on-first primitives were already evaluated above, they are already in cache
    full_scores: list[tuple[int, float, float]] = []  # (idx_in_prims, num_correct_total, secondary_score)

    total_tr = len(challenge.train)
    def _has_cache(pid: str) -> bool:
        try:
            return challenge_primitive_scores is not None and pid in challenge_primitive_scores.get(challenge.id, {})
        except Exception:
            return False

    # Build list of (idx, primitive) for candidates
    cand_list = [(prims.index(p), p) for p in candidates]

    # Evaluate in batches
    for start in range(0, len(cand_list), sp_batch):
        batch = cand_list[start:start+sp_batch]
        # Split into cached and to-eval
        to_eval: list[tuple[int, Primitive]] = []
        for idx, p in batch:
            pid = getattr(p, "id", "?")
            if _has_cache(pid):
                try:
                    nc, sc = challenge_primitive_scores[challenge.id][pid]
                    full_scores.append((idx, float(nc), float(sc)))
                except Exception:
                    to_eval.append((idx, p))
            else:
                to_eval.append((idx, p))

        if to_eval:
            tasks = [
                evaluate_primitive_weighed_async(p, challenge, challenge_primitive_scores)
                for _, p in to_eval
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (idx, p), r in zip(to_eval, results):
                if isinstance(r, Exception):
                    full_scores.append((idx, 0.0, 0.0))
                else:
                    nc, sc = r
                    full_scores.append((idx, float(nc), float(sc)))

    # Compute selection probabilities via softmax over (num_correct + secondary_score)
    scores = []
    idxs = []
    for idx, nc, sc in full_scores:
        scores.append(nc + sc)
        idxs.append(idx)
    if not scores:
        return []

    scores_arr = np.array(scores)
    exp_scores = np.exp(scores_arr - np.max(scores_arr))
    probabilities = exp_scores / np.clip(exp_scores.sum(), 1e-12, None)

    # Sample k_top distinct primitives based on probabilities
    selected_idx_positions = _safe_sample_indices(len(scores), min(k_top, len(scores)), probabilities, ctx=f"two_pass:{challenge.id}")
    final_indices = [idxs[pos] for pos in selected_idx_positions]
    return [prims[i] for i in final_indices]


def get_best_primitives(
    library: Library, challenge: Challenge, k_top: int
) -> list[Primitive]:
    # first, order primitives by how many examples they got right
    # then, order by the diff in cells
    # TODO: use a better metric later
    example_correct: list[Primitive] = []
    secondary_score_lst: list[Primitive] = []
    for primitive in library.primitives:
        t0 = time.perf_counter()
        transform_results = run_python_transform_sync_cached(
            code=primitive.python_code_str,
            grid_lists=[deepcopy(train.input) for train in challenge.train],
            timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
            raise_exception=False, # since we are applying all primitives to all tasks
        )
        perf_add(challenge.id, "scoring_transform_ms", (time.perf_counter() - t0) * 1000.0)
        if transform_results.transform_results:
            transformed_grids = transform_results.transform_results
            t1 = time.perf_counter()
            num_correct = 0
            avg_right_lst = []
            for idx, train in enumerate(challenge.train):
                if train.output == transformed_grids[idx]:
                    num_correct += 1
                train_accuracy = percent_right_from_grids(train.output, transformed_grids[idx])
                avg_right_lst.append(train_accuracy)
            perf_add(challenge.id, "scoring_compute_ms", (time.perf_counter() - t1) * 1000.0)
            example_correct.append(num_correct)
            secondary_score_lst.append(sum(avg_right_lst) / len(avg_right_lst))
        else:
            example_correct.append(0)
            secondary_score_lst.append(0)
    # Get the sorted indices in descending order
    sorted_indices = sorted(range(len(example_correct)), key=lambda i: (example_correct[i], secondary_score_lst[i]), reverse=True)

    k_top = min(k_top, len(sorted_indices))
    return [library.primitives[i] for i in sorted_indices[:k_top]]

def get_latents_from_lpn(
    lpn_model: LPN,
    evaluator: Evaluator,
    key: chex.PRNGKey,
    input_list: list[GRID],
    output_list: list[GRID],
) -> tuple[chex.Array, chex.Array]:
    pair_list, shape_list = [], []
    for input, output in zip(input_list, output_list):
        input = jnp.array(input)
        input_shape = input.shape
        input, input_shape = evaluator.pad_and_crop_json(input, input_shape)
        output = jnp.array(output)
        output_shape = output.shape
        output, output_shape = evaluator.pad_and_crop_json(output, output_shape)
        pair_list.append(jnp.stack([input, output], axis=-1))
        shape_list.append(jnp.stack(jnp.array([input_shape, output_shape]), axis=-1))
    pairs = jnp.stack(pair_list)
    grid_shapes = jnp.stack(shape_list)

    latents_mu, latents_logvar = lpn_model.encoder(pairs, grid_shapes, True)

    if latents_logvar is not None:
        # TODO: try generate a key everytime to encourage exploration
        key, key_latents = jax.random.split(key)
        latents, *_ = lpn_model._sample_latents(latents_mu, latents_logvar, key_latents)
    else:
        latents = latents_mu

    mode_kwargs = {
        "num_steps": 9,
        "lr": 0.01,
    }

    first_context, second_context = lpn_model._get_gradient_ascent_context(
        latents, pairs, grid_shapes, key, **mode_kwargs
    )

    return first_context, second_context


def _safe_sample_indices(n: int, k_top: int, probabilities: onp.ndarray | list[float], ctx: str = "") -> list[int]:
    if n <= 0 or k_top <= 0:
        return []
    k = min(k_top, n)
    used_uniform = False
    try:
        p = onp.asarray(probabilities, dtype=float).reshape(-1)
    except Exception:
        p = None
    if p is None or p.shape[0] != n or not onp.isfinite(p).all() or p.sum() <= 0:
        # fallback to uniform
        p = onp.ones(n, dtype=float) / n
        used_uniform = True
    else:
        p = onp.maximum(p, 0.0)
        s = p.sum()
        if s <= 0 or not onp.isfinite(s):
            p = onp.ones(n, dtype=float) / n
            used_uniform = True
        else:
            p = p / s
    if used_uniform:
        try:
            logfire.debug(f"SAFE_SAMPLE: uniform fallback (ctx={ctx}, n={n}, k={k})")
        except Exception:
            pass
    try:
        return onp.random.choice(n, size=k, replace=False, p=p).tolist()
    except Exception as e:
        try:
            logfire.debug(f"SAFE_SAMPLE: deterministic fallback due to np.random.choice error (ctx={ctx}, n={n}, k={k}, err={e})")
        except Exception:
            pass
        # Final fallback: deterministic first k indices
        return list(range(k))


def get_best_primitives_by_lpn_vmap(
    library: Library, 
    challenge: Challenge, 
    k_top: int, 
    lpn_model: LPN, 
    evaluator: Evaluator, 
    key: chex.PRNGKey,
    challenge_primitive_scores: dict[str, dict[str, float]] = None,
    batch_size: int = 50,  # Process primitives in batches to avoid memory issues
) -> list[Primitive]:
    """Vectorized version using jax.vmap for parallel LPN inference."""
    if len(library.primitives) == 0:
        return []
    
    example_input_list = [example.input for example in challenge.train]
    example_output_list = [example.output for example in challenge.train]
    
    expected_latents, _ = get_latents_from_lpn(lpn_model, evaluator, key, example_input_list, example_output_list)

    # Filter out already computed primitives
    primitives_to_compute = []
    cosine_similarity_lst = []
    
    for primitive in library.primitives:
        if challenge.id in challenge_primitive_scores and primitive.id in challenge_primitive_scores[challenge.id]:
            cosine_similarity_lst.append(challenge_primitive_scores[challenge.id][primitive.id])
        else:
            primitives_to_compute.append(primitive)
            cosine_similarity_lst.append(None)  # Placeholder
    
    if primitives_to_compute:
        # Process primitives in batches to avoid memory issues
        primitive_batches = [primitives_to_compute[i:i + batch_size] for i in range(0, len(primitives_to_compute), batch_size)]
        
        for batch in primitive_batches:
            batch_results = process_primitive_batch_parallel(
                batch, challenge, lpn_model, evaluator, key, example_input_list, expected_latents
            )
            
            # Update results
            for i, (primitive, result) in enumerate(zip(batch, batch_results)):
                global_idx = library.primitives.index(primitive)
                cosine_similarity_lst[global_idx] = result
                
                if challenge_primitive_scores is not None:
                    if challenge.id not in challenge_primitive_scores:
                        challenge_primitive_scores[challenge.id] = {}
                    challenge_primitive_scores[challenge.id][primitive.id] = result

    # Convert scores to probabilities using softmax
    scores = np.array(cosine_similarity_lst)
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    probabilities = exp_scores / exp_scores.sum()

    # Sample k_top primitives based on probabilities
    k_top = min(k_top, len(library.primitives))
    selected_indices = _safe_sample_indices(
        len(library.primitives), k_top, probabilities, ctx=f"lpn_vmap:{challenge.id}"
    )
    return [library.primitives[i] for i in selected_indices]

def process_primitive_batch_parallel(
    primitives: list[Primitive], 
    challenge: Challenge, 
    lpn_model: LPN, 
    evaluator: Evaluator,
    key: chex.PRNGKey,
    example_input_list: list[GRID], 
    expected_latents: chex.Array
) -> list[float]:
    """Process a batch of primitives using jax.lax.map for parallel LPN inference."""
    
    # First, run Python transforms for all primitives in the batch
    transform_results_list = []
    
    for primitive in primitives:
        try:
            transform_results = run_python_transform_sync_cached(
                code=primitive.python_code_str,
                grid_lists=[deepcopy(train.input) for train in challenge.train],
                timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
                raise_exception=False,
            )
            if transform_results and transform_results.transform_results:
                transform_results_list.append(transform_results.transform_results)
            else:
                # Add None for invalid primitives
                transform_results_list.append(None)
        except Exception as e:
            logfire.debug(f"error running python transform: {e=} for primitive {primitive.python_code_str} and challenge {challenge.id}")
            transform_results_list.append(None)
    
    # Filter out None results
    valid_transforms = [t for t in transform_results_list if t is not None]
    valid_indices = [i for i, t in enumerate(transform_results_list) if t is not None]
    
    if not valid_transforms:
        return [0.0] * len(primitives)
    
    # Use Python for loop with JIT compilation for parallel processing
    try:
        # JIT compile the processing function for better performance
        @jax.jit
        def jitted_process_single_primitive(transform):
            primitive_latents, _ = get_latents_from_lpn(lpn_model, evaluator, key, example_input_list, transform)
            
            # Calculate cosine similarity
            expected_norm = jnp.linalg.norm(expected_latents, axis=-1, keepdims=True)
            primitive_norm = jnp.linalg.norm(primitive_latents, axis=-1, keepdims=True)
            
            expected_normalized = expected_latents / (expected_norm + 1e-8)
            primitive_normalized = primitive_latents / (primitive_norm + 1e-8)
            
            cosine_similarity = jnp.sum(expected_normalized * primitive_normalized, axis=-1)
            avg_cosine_similarity = jnp.mean(cosine_similarity)
            
            return avg_cosine_similarity
        
        # Process each primitive individually with JIT compilation
        similarities = []
        for transform in valid_transforms:
            try:
                similarity = jitted_process_single_primitive(transform)
                similarities.append(float(similarity))
            except Exception as e:
                logfire.debug(f"Error processing primitive: {e}")
                similarities.append(0.0)
        
        # Convert back to list and handle invalid primitives
        results = [0.0] * len(primitives)
        for valid_idx, similarity in zip(valid_indices, similarities):
            results[valid_idx] = similarity
        
        return results
        
    except Exception as e:
        logfire.debug(f"Error in JIT processing: {e}. Falling back to sequential processing.")
        # Fall back to sequential processing
        return process_primitive_batch_sequential(
            primitives, challenge, lpn_model, evaluator, key, example_input_list, expected_latents
        )

def process_primitive_batch_sequential(
    primitives: list[Primitive], 
    challenge: Challenge, 
    lpn_model: LPN, 
    evaluator: Evaluator,
    key: chex.PRNGKey,
    example_input_list: list[GRID], 
    expected_latents: chex.Array
) -> list[float]:
    """Fallback sequential processing when vmap fails."""
    results = []
    for primitive in primitives:
        try:
            transform_results = run_python_transform_sync(
                code=primitive.python_code_str,
                grid_lists=[deepcopy(train.input) for train in challenge.train],
                timeout=5,
                raise_exception=False,
            )
            
            if transform_results and transform_results.transform_results:
                transformed_grids = transform_results.transform_results
                try:
                    primitive_latents, _ = get_latents_from_lpn(lpn_model, evaluator, key, example_input_list, transformed_grids)
                except Exception as e:
                    logfire.debug(f"error getting latents from lpn: {e=} for primitive {primitive.python_code_str} and challenge {challenge.id}")
                    results.append(0.0)
                    continue

                # Calculate cosine similarity
                expected_norm = jnp.linalg.norm(expected_latents, axis=-1, keepdims=True)
                primitive_norm = jnp.linalg.norm(primitive_latents, axis=-1, keepdims=True)
                
                expected_normalized = expected_latents / (expected_norm + 1e-8)
                primitive_normalized = primitive_latents / (primitive_norm + 1e-8)
                
                cosine_similarity = jnp.sum(expected_normalized * primitive_normalized, axis=-1)
                avg_cosine_similarity = jnp.mean(cosine_similarity)
                
                results.append(float(avg_cosine_similarity))
            else:
                results.append(0.0)
                
        except Exception as e:
            logfire.debug(f"error running python transform: {e=} for primitive {primitive.python_code_str} and challenge {challenge.id}")
            results.append(0.0)
    
    return results

async def evaluate_primitive_weighed_async(
        primitive: Primitive, 
        challenge: Challenge, 
        challenge_primitive_scores: dict[str, dict[str, tuple[float, float]]] = None
    ) -> tuple[float, float]:
    """Evaluate a single primitive asynchronously for weighed scoring."""
    if challenge_primitive_scores is not None:
        if challenge.id in challenge_primitive_scores and primitive.id in challenge_primitive_scores[challenge.id]:
            return challenge_primitive_scores[challenge.id][primitive.id]

    try:
        t0 = time.perf_counter()
        transform_results = await run_transform_cached_async(
            code=primitive.python_code_str,
            grid_lists=[deepcopy(train.input) for train in challenge.train],
            timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
            raise_exception=False,  # since we are applying all primitives to all tasks
        )
        perf_add(challenge.id, "scoring_transform_ms", (time.perf_counter() - t0) * 1000.0)
        if transform_results.transform_results:
            transformed_grids = transform_results.transform_results
            t1 = time.perf_counter()
            num_correct, secondary_score = await compute_primitive_score_async(challenge.train, transformed_grids)
            perf_add(challenge.id, "scoring_compute_ms", (time.perf_counter() - t1) * 1000.0)
            if challenge_primitive_scores is not None:
                challenge_primitive_scores.setdefault(challenge.id, {})
                challenge_primitive_scores[challenge.id][primitive.id] = (float(num_correct), float(secondary_score))
            return float(num_correct), float(secondary_score)
        else:
            return 0.0, 0.0
    except Exception as e:
        logfire.debug(f"Error evaluating primitive {primitive.id}: {e}")
        return 0.0, 0.0

async def get_best_primitives_weighed_by_score_async(
    library: Library, 
    challenge: Challenge, 
    k_top: int, 
    challenge_primitive_scores: dict[str, dict[str, tuple[float, float]]] = None
) -> list[Primitive]:
    """Parallel version of get_best_primitives_weighed_by_score using asyncio.
    Optionally uses a two-pass strategy controlled via env vars to reduce full scoring.
    """
    if len(library.primitives) == 0:
        return []

    # Two-pass gate (default ON): first-pass single-example across all, second-pass full on top-K
    two_pass_enabled = os.environ.get("SUBMISSION_TWO_PASS_ENABLED", "1") != "0"
    if two_pass_enabled:
        try:
            return await _two_pass_select_primitives_async(library, challenge, k_top, challenge_primitive_scores)
        except Exception as e:
            # Fall back to legacy path on any error
            logfire.debug(f"Two-pass selection failed for {challenge.id}, falling back. err={e}")

    # Legacy: evaluate all primitives fully, then softmax sample k_top
    # Create tasks for all primitives
    tasks = [
        evaluate_primitive_weighed_async(primitive, challenge, challenge_primitive_scores) 
        for primitive in library.primitives
    ]

    # Execute all tasks in parallel (bounded inside evaluate_primitive_weighed_async)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    example_correct: list[float] = []
    secondary_score_lst: list[float] = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logfire.debug(f"Exception in primitive {i}: {result}")
            print(f"Exception in primitive {i}: {result}")
            example_correct.append(0.0)
            secondary_score_lst.append(0.0)
        else:
            num_correct, secondary_score = result
            example_correct.append(num_correct)
            secondary_score_lst.append(secondary_score)

    # secondary score is the average of the train accuracy across examples
    # we are essentially biasing towards primitives which get 100% on an
    # example instead of 99% by adding primary score to secondary score
    scores = [p + s for p, s in zip(example_correct, secondary_score_lst)]

    # Convert scores to probabilities using softmax
    scores = np.array(scores)
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    probabilities = exp_scores / exp_scores.sum()

    # Sample k_top primitives based on probabilities
    k_top = min(k_top, len(library.primitives))
    selected_indices = _safe_sample_indices(
        len(library.primitives), k_top, probabilities, ctx=f"weighed_async:{challenge.id}"
    )
    return [library.primitives[i] for i in selected_indices]

def get_best_primitives_weighed_by_score(
    library: Library, challenge: Challenge, k_top: int
) -> list[Primitive]:
    """Synchronous version for backward compatibility."""
    if len(library.primitives) == 0:
        return []
    
    # first, calculate scores for each primitive
    example_correct: list[float] = []
    secondary_score_lst: list[float] = []
    for primitive in library.primitives:
        transform_results = run_python_transform_sync(
            code=primitive.python_code_str,
            grid_lists=[deepcopy(train.input) for train in challenge.train],
            timeout=5,
            raise_exception=False, # since we are applying all primitives to all tasks
        )
        if transform_results.transform_results:
            transformed_grids = transform_results.transform_results
            num_correct = 0
            avg_right_lst = []
            for idx, train in enumerate(challenge.train):
                if train.output == transformed_grids[idx]:
                    num_correct += 1
                train_accuracy = percent_right_from_grids(train.output, transformed_grids[idx])
                avg_right_lst.append(train_accuracy)
            example_correct.append(num_correct)
            secondary_score_lst.append(sum(avg_right_lst) / len(avg_right_lst))
        else:
            example_correct.append(0)
            secondary_score_lst.append(0)

    # secondary score is the average of the train accuracy across examples
    # we are essentially biasing towards primitives which get 100% on an
    # example instead of 99% by adding primary score to secondary score
    scores = [p + s for p, s in zip(example_correct, secondary_score_lst)]
    
    # Convert scores to probabilities using softmax
    scores = np.array(scores)
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    probabilities = exp_scores / exp_scores.sum()

    # Sample k_top primitives based on probabilities
    k_top = min(k_top, len(library.primitives))
    selected_indices = _safe_sample_indices(
        len(library.primitives), k_top, probabilities, ctx=f"weighed_sync:{challenge.id}"
    )
    return [library.primitives[i] for i in selected_indices]

def get_best_attempts(
    attempts: list[Attempt], k_top: int, unique_code: bool, unique_output: bool
) -> list[Attempt]:
    # first, order attempts by how many examples they got right
    # then, order by the diff in cells
    # use a better metric later
    example_correct: list[Attempt] = []
    example_wrong: list[Attempt] = []
    for a in attempts:
        if a.train_accuracy > 0:
            example_correct.append(a)
        else:
            example_wrong.append(a)
    sorted_correct = sorted(
        example_correct, key=lambda a: a.train_accuracy, reverse=True
    )
    sorted_wrong = sorted(
        example_wrong,
        key=lambda a: a.avg_cell_diff_percent,
        reverse=True,
    )
    all_sorted: list[Attempt] = [*sorted_correct, *sorted_wrong]

    if unique_code:
        has_seen_python: set[str] = set()
        unique_sorted = []
        for item in all_sorted:
            code_str = item.python_code_str
            if code_str not in has_seen_python:
                unique_sorted.append(item)
                has_seen_python.add(code_str)
        all_sorted = unique_sorted
    if unique_output:
        has_seen_grid: set[str] = set()
        unique_sorted = []
        for item in all_sorted:
            output_grid = str(item.test_attempt)
            if output_grid not in has_seen_grid:
                unique_sorted.append(item)
                has_seen_grid.add(output_grid)
        all_sorted = unique_sorted

    return all_sorted[:k_top]


def get_diverse_attempts(
    root_attempt: Attempt, sorted_attempts: list[Attempt], limit: int
) -> list[Attempt]:
    if root_attempt in sorted_attempts:
        sorted_attempts.remove(root_attempt)
    attempts_by_correct_examples: dict[int, list[Attempt]] = {}
    correct_examples_by_attempt: dict[Attempt, set[int]] = {}
    for a in [root_attempt, *sorted_attempts]:
        for i, train_example in enumerate(a.challenge.train):
            if a.train_attempts[i] == train_example.output:
                if i not in attempts_by_correct_examples:
                    attempts_by_correct_examples[i] = []
                attempts_by_correct_examples[i].append(a)
                if a not in correct_examples_by_attempt:
                    correct_examples_by_attempt[a] = set()
                correct_examples_by_attempt[a].add(i)
    # make sure you have at least one attempt for each correct example
    final_attempts: list[Attempt] = [root_attempt, *sorted_attempts][0:limit]
    count: dict[int, int] = {}
    for a in final_attempts:
        for ii in correct_examples_by_attempt.get(a, set()):
            count[ii] = 1
    # find the missing ones
    missing = attempts_by_correct_examples.keys() - count.keys()
    for miss in missing:
        final_attempts.append(attempts_by_correct_examples[miss][0])
    return final_attempts


def has_perfect_attempts(attempts: list[Attempt]) -> bool:
    attempts_perfect = [a for a in attempts if a.train_accuracy == 1]
    n_perfect = len(attempts_perfect)
    if n_perfect >= 2:
        message = f"[{attempts_perfect[0].challenge.id}] found {n_perfect} solutions from {len(attempts)} attempts"
        logfire.debug(message)
        print(message)
        return True
    return False


async def run_fixes_tree(
    parent_attempts: list[Attempt],
    edges: list[AttemptEdge],
    warm_cache: bool,  # too complex rn w speed
) -> list[Attempt]:
    # DFS fixes
    all_attempts: list[Attempt] = []
    if not parent_attempts:
        challenge_id = ""
    else:
        challenge_id = parent_attempts[0].challenge.id
    if not edges:
        return all_attempts
    for edge in edges:
        best_k = get_best_attempts(
            attempts=parent_attempts,
            k_top=edge.k_top_config.k_top,
            unique_code=edge.k_top_config.unique_code,
            unique_output=edge.k_top_config.unique_output,
        )
        if not best_k:
            continue
        for fix_attempt_config in edge.configs:
            start_level = time.time()
            message = f"[{best_k[0].challenge.id}] running fix node with {fix_attempt_config.attempts * len(best_k)} total attempts."
            print(message)
            logfire.debug(message)
            if fix_attempt_config.attempts == 0:
                continue
            local_attempts: list[Attempt] = []
            tasks = []
            for parent_attempt in best_k:
                if not edge.pooling:
                    tasks.append(
                        parent_attempt.fix_many(
                            attempt_config=fix_attempt_config.model_copy(deep=True),
                            raise_exception=False,
                            n_times=fix_attempt_config.attempts,
                        )
                    )
                else:
                    attempts_to_use = get_diverse_attempts(
                        root_attempt=parent_attempt,
                        sorted_attempts=get_best_attempts(
                            attempts=parent_attempts,
                            k_top=100_000,
                            unique_code=edge.k_top_config.unique_code,
                            unique_output=edge.k_top_config.unique_output,
                        ),
                        limit=edge.pooling.size,
                    )
                    tasks.append(
                        Attempt.run_many(
                            challenge=parent_attempt.challenge,
                            attempt_config=fix_attempt_config.model_copy(deep=True),
                            raise_exception=False,
                            fixing=attempts_to_use,
                            n_times=fix_attempt_config.attempts,
                        )
                    )

            responses = await tqdm_asyncio.gather(
                *tasks,
                desc=f"[{challenge_id}] Processing fix attempts",
                file=TqdmLogfire(),
            )
            for r in responses:
                local_attempts.extend(r)

            start_eval = time.time()
            took_level = time.time() - start_level
            eval_attempts(
                attempts=local_attempts,
                config=fix_attempt_config,
                plot=PLOT,
                time_took_ms=(took_level * 1000),
            )
            logfire.debug(
                f"[{challenge_id}] eval took {(time.time() - start_eval)} secs"
            )
            all_attempts.extend(local_attempts)
            # now see if you have a solution
            if has_perfect_attempts(all_attempts):
                return all_attempts

            if fix_attempt_config.include_all_attempts_in_fixes:
                parent_attempts = all_attempts
            else:
                parent_attempts = local_attempts

            # now run the fixes
            all_attempts.extend(
                await run_fixes_tree(
                    parent_attempts=parent_attempts,
                    edges=fix_attempt_config.fixes,
                    warm_cache=warm_cache,
                )
            )

            dedup_attempts(all_attempts)
            if has_perfect_attempts(all_attempts):
                return all_attempts

    logfire.debug(f"ALL ATTEMPTS LEN: {len(all_attempts)}")
    return all_attempts


def dedup_attempts(attempts: list[Attempt]) -> list[Attempt]:
    has_seen: set[str] = set()
    _all_attempts = []
    for a in attempts:
        if a.id not in has_seen:
            _all_attempts.append(a)
        has_seen.add(a.id)

    return _all_attempts


async def run_tree(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    warm_cache_root: bool,
    warm_cache_fix: bool,
    library: Library = None,
    use_primitives_weighed_by_score: bool = False,
    lpn_model: LPN = None,
    evaluator: Evaluator = None,
    key: chex.PRNGKey = None,
    challenge_primitive_lpn_scores: dict[str, dict[str, float]] = None,
    challenge_primitive_accuracy_scores: dict[str, dict[str, tuple[float, float]]] = None,
    on_llm_dispatch: T.Callable[[], None] | None = None,
) -> list[Attempt]:
    assert not(use_primitives_weighed_by_score and lpn_model), "Cannot use both use_primitives_weighed_by_score and lpn_model"

    # Determine how many top primitives to use from the library (configurable via env)
    top_k_env = os.environ.get("SUBMISSION_PRIMITIVES_TOP_K")
    try:
        top_k = max(1, int(top_k_env)) if top_k_env is not None else 2
    except Exception:
        top_k = 2
    print(f"[{challenge.id}] Using top K primitives = {top_k}")
    logfire.debug(f"[{challenge.id}] Using top K primitives = {top_k}")

    # find the best functions in the library for this challenge
    if use_primitives_weighed_by_score:
        primitives = await get_best_primitives_weighed_by_score_async(
            library=library,
            challenge=challenge,
            k_top=top_k,
            challenge_primitive_scores=challenge_primitive_accuracy_scores,
        )
    elif lpn_model and evaluator:
        primitives = get_best_primitives_by_lpn_vmap(
            library=library,
            challenge=challenge,
            k_top=top_k,
            lpn_model=lpn_model,
            evaluator=evaluator,
            key=key,
            challenge_primitive_scores=challenge_primitive_lpn_scores,
        )
    else:
        primitives = get_best_primitives(library=library, challenge=challenge, k_top=top_k)
    if primitives:
        def _summ(p):
            try:
                code = getattr(p, "python_code_str", "") or ""
                short = (code[:20] + "...") if len(code) > 20 else code
                return {"id": getattr(p, "id", "?"), "code": short}
            except Exception:
                return {"id": getattr(p, "id", "?"), "code": "<unavailable>"}
        _display = [_summ(p) for p in primitives]
        print(f"[{challenge.id}] Found primitives: {_display}")
        logfire.debug(f"[{challenge.id}] Found primitives: {_display}")
    else:
        primitives = None

    rate_delay_env = os.environ.get("SUBMISSION_RATE_DELAY_MAX_SECONDS")
    try:
        rate_delay_max = max(0.0, float(rate_delay_env)) if rate_delay_env is not None else 0.0
    except Exception:
        rate_delay_max = 0.0
    if rate_delay_max > 0.0:
        await asyncio.sleep(random.random() * rate_delay_max)

    all_attempts: list[Attempt] = []
    dispatch_signalled = False
    for root_attempt_config in tree:
        start_level = time.time()
        message = f"[{challenge.id}] running root node with {root_attempt_config.attempts} attempts."
        print(message)
        logfire.debug(message)
        # Signal once just before we begin LLM calls for this challenge
        if on_llm_dispatch and not dispatch_signalled:
            try:
                on_llm_dispatch()
            except Exception:
                pass
            dispatch_signalled = True
        local_attempts = await Attempt.run_many(
            challenge=challenge,
            attempt_config=root_attempt_config,
            raise_exception=False,
            fixing=[],
            n_times=root_attempt_config.attempts,
            primitives=primitives,
        )
        start_eval = time.time()
        took_level = time.time() - start_level
        eval_attempts(
            attempts=local_attempts,
            config=root_attempt_config,
            plot=PLOT,
            time_took_ms=(took_level * 1000),
        )
        logfire.debug(f"[{challenge.id}] eval took {(time.time() - start_eval)} secs")
        print(f"[{challenge.id}] eval took {(time.time() - start_eval)} secs")
        all_attempts.extend(local_attempts)
        all_attempts = dedup_attempts(all_attempts)

        # now see if you have a solution
        if has_perfect_attempts(all_attempts):
            return all_attempts

        # now run the fixes
        """
        if root_attempt_config.include_all_attempts_in_fixes:
            parent_attempts = all_attempts
        else:
            parent_attempts = local_attempts
        all_attempts.extend(
            await run_fixes_tree(
                parent_attempts=parent_attempts,
                edges=root_attempt_config.fixes,
                warm_cache=warm_cache_fix,
            )
        )
        all_attempts = dedup_attempts(all_attempts)

        # now see if you have a solution
        if has_perfect_attempts(all_attempts):
            return all_attempts
        """
    return dedup_attempts(all_attempts)


def get_grids_from_attempt(attempt: Attempt) -> list[GRID]:
    challenge = attempt.challenge
    if len(challenge.test) == 1:
        return [attempt.test_attempt]
    transform_results = run_python_transform_sync_cached(
        code=attempt.python_code_str,
        grid_lists=[deepcopy(test.input) for test in challenge.test],
        timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
        raise_exception=True,
    )
    logfire.debug(
        f"[{challenge.id}] FINAL: Transform results took {transform_results.latency_ms:.2f} ms"
    )
    return transform_results.transform_results


async def can_primitive_solve_challenge_async(
    primitive: Primitive,
    challenge: Challenge,
    challenge_primitive_scores: dict[str, dict[float]] = None,
) -> bool:
    try:
        transform_train_results = await run_transform_sync_async(
            code=primitive.python_code_str,
            grid_lists=[deepcopy(train.input) for train in challenge.train],
            timeout=5,
            raise_exception=False, # since we are applying all primitives to all tasks
        )
        transform_eval_results = await run_transform_sync_async(
            code=primitive.python_code_str,
            grid_lists=[deepcopy(test.input) for test in challenge.test],
            timeout=5,
            raise_exception=False, # since we are applying all primitives to all tasks
        )
        if transform_train_results.transform_results and transform_eval_results.transform_results:
            transformed_train_grids = transform_train_results.transform_results
            transformed_eval_grids = transform_eval_results.transform_results
            num_train_correct, num_eval_correct = 0, 0
            avg_right_lst: list[float] = []
            for idx, train in enumerate(challenge.train):
                if train.output == transformed_train_grids[idx]:
                    num_train_correct += 1
                train_accuracy = percent_right_from_grids(train.output, transformed_train_grids[idx])
                avg_right_lst.append(train_accuracy)
            for idx, test in enumerate(challenge.test):
                if test.output == transformed_eval_grids[idx]:
                    num_eval_correct += 1

            secondary_score = sum(avg_right_lst) / len(avg_right_lst)
            challenge_primitive_scores[challenge.id][primitive.id] = (float(num_train_correct), secondary_score)

            if num_train_correct == len(challenge.train) and num_eval_correct == len(challenge.test):
                return True
        return False
    except Exception as e:
        logfire.debug(f"[{challenge.id}] Error applying primitive {primitive.id}: {e}")
        return False
    

async def can_library_solve_challenge(
    library: Library,
    challenge: Challenge,
    challenge_primitive_scores: dict[str, dict[float]] = None,
) -> bool:
    # Create tasks for all primitives
    tasks = [
        can_primitive_solve_challenge_async(primitive, challenge, challenge_primitive_scores) 
        for primitive in library.primitives
    ]
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return any(results)
    

async def solve_challenge(
    tree: list[RootAttemptConfig], 
    challenge: Challenge, 
    library: Library = None, 
    url: str = None, 
    use_primitives_weighed_by_score: bool = False,
    lpn_model: LPN = None,
    evaluator: Evaluator = None,
    key: chex.PRNGKey = None,
    challenge_primitive_lpn_scores: dict[str, dict[str, float]] = None,
    challenge_primitive_accuracy_scores: dict[str, dict[str, tuple[float, float]]] = None,
    aggregate_cost_in_cents: list[float] = [ 0.0 ],
    on_llm_dispatch: T.Callable[[], None] | None = None,
) -> tuple[list[GRID], list[GRID]]:
    if url:
        env_vars = {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "LOGFIRE_TOKEN": os.environ.get("LOGFIRE_TOKEN"),
            "NEON_DB_DSN": os.environ.get("NEON_DB_DSN"),
            "XAI_API_KEY": os.environ.get("XAI_API_KEY"),
        }
        if "KAGGLE" in os.environ:
            # RECORD 0 LOGS
            env_vars["KAGGLE"] = os.environ["KAGGLE"]
        try:
            async with httpx.AsyncClient(timeout=3000) as client:
                r = await client.post(
                    url,
                    json={
                        "tree": TypeAdapter(list[RootAttemptConfig]).dump_python(
                            tree, mode="json"
                        ),
                        "challenge": Challenge.model_dump(challenge, mode="json"),
                        "env_vars": env_vars,
                    },
                )
                j = r.json()
            print(f"[{challenge.id}] solved")
            return j
        except Exception as e:
            logfire.debug(f"ERROR RUNNING PYTHON: {e}")
            pass

    run_id = f"run_{random_string(10)}"
    started_at_ms = time.time() * 1000

    attempts = await run_tree(
        tree=tree, 
        challenge=challenge, 
        library=library, 
        warm_cache_root=True, 
        warm_cache_fix=False, 
        use_primitives_weighed_by_score=use_primitives_weighed_by_score,
        lpn_model=lpn_model,
        evaluator=evaluator,
        key=key,
        challenge_primitive_lpn_scores=challenge_primitive_lpn_scores,
        challenge_primitive_accuracy_scores=challenge_primitive_accuracy_scores,
        on_llm_dispatch=on_llm_dispatch,
    )
    attempts = dedup_attempts(attempts)

    # Report perf metrics
    _perf = perf_get(challenge.id)
    if _perf:
        print(f"[{challenge.id}] perf: scoring_transform_ms={_perf.get('scoring_transform_ms', 0):.1f}, scoring_compute_ms={_perf.get('scoring_compute_ms', 0):.1f}, llm_ms={_perf.get('llm_ms', 0):.1f}, post_llm_transform_ms={_perf.get('post_llm_transform_ms', 0):.1f}")
        logfire.debug(f"[{challenge.id}] perf: scoring_transform_ms={_perf.get('scoring_transform_ms', 0):.1f}, scoring_compute_ms={_perf.get('scoring_compute_ms', 0):.1f}, llm_ms={_perf.get('llm_ms', 0):.1f}, post_llm_transform_ms={_perf.get('post_llm_transform_ms', 0):.1f}")
        perf_clear(challenge.id)

    # get the number and cost from all of these attempts
    total_cost_cents = sum(a.cost_cents for a in attempts)
    aggregate_cost_in_cents[0] += total_cost_cents
    logfire.debug(
        f"[{challenge.id}] DONE: n attempts: {len(attempts)}, total cost cents: {total_cost_cents}, aggregate cost in cents: {aggregate_cost_in_cents[0]}"
    )
    print(f"[{challenge.id}] DONE: n attempts: {len(attempts)}, total cost cents: {total_cost_cents}, aggregate cost in cents: {aggregate_cost_in_cents[0]}")


    ended_at_ms = time.time() * 1000

    if os.environ.get("NEON_DB_DSN"):
        await Attempt.insert_run(
            run_id=run_id, started_at_ms=started_at_ms, ended_at_ms=ended_at_ms
        )
        await Attempt.insert_many(attempts=attempts, run_id=run_id)

    top_two = get_best_attempts(
        attempts=attempts, k_top=2, unique_code=True, unique_output=True
    )
    
    if len(top_two) == 0:
        # TODO: LLM call failed. Return empty solutions
        return [], []

    if len(top_two) == 1:
        top_two.append(top_two[0])

    # TODO: only add primitive if top one scores better than best primitive on this task
    if library and top_two:
        current_primitive_count = len(library.primitives)
        library.add_primitive(Primitive(id=f"{current_primitive_count}", python_code_str=top_two[0].python_code_str))

    first_solution = top_two[0]
    second_solution = top_two[1]

    if PLOT:
        first_solution.plot(ignore_fixing=True)
        second_solution.plot(ignore_fixing=True)

    return get_grids_from_attempt(first_solution), get_grids_from_attempt(
        second_solution
    )


async def solve_challenge_with_accuracy(
    tree: list[RootAttemptConfig], 
    challenge: Challenge, 
    library: Library = None, 
    url: str = None, 
    use_primitives_weighed_by_score: bool = False,
    lpn_model: LPN = None,
    evaluator: Evaluator = None,
    key: chex.PRNGKey = None,
    challenge_primitive_lpn_scores: dict[str, dict[str, float]] = None,
    challenge_primitive_accuracy_scores: dict[str, dict[str, tuple[float, float]]] = None,
    aggregate_cost_in_cents: list[float] = [ 0.0 ],
    on_llm_dispatch: T.Callable[[], None] | None = None,
) -> list[tuple[list[GRID], float]]:
    run_id = f"run_{random_string(10)}"
    started_at_ms = time.time() * 1000

    attempts = await run_tree(
        tree=tree, 
        challenge=challenge, 
        library=library, 
        warm_cache_root=True, 
        warm_cache_fix=False, 
        use_primitives_weighed_by_score=use_primitives_weighed_by_score,
        lpn_model=lpn_model,
        evaluator=evaluator,
        key=key,
        challenge_primitive_lpn_scores=challenge_primitive_lpn_scores,
        challenge_primitive_accuracy_scores=challenge_primitive_accuracy_scores,
        on_llm_dispatch=on_llm_dispatch,
    )
    attempts = dedup_attempts(attempts)

    # get the number and cost from all of these attempts
    total_cost_cents = sum(a.cost_cents for a in attempts)
    aggregate_cost_in_cents[0] += total_cost_cents
    logfire.debug(
        f"[{challenge.id}] DONE: n attempts: {len(attempts)}, total cost cents: {total_cost_cents}, aggregate cost in cents: {aggregate_cost_in_cents[0]}"
    )
    print(f"[{challenge.id}] DONE: n attempts: {len(attempts)}, total cost cents: {total_cost_cents}, aggregate cost in cents: {aggregate_cost_in_cents[0]}")


    ended_at_ms = time.time() * 1000

    if os.environ.get("NEON_DB_DSN"):
        await Attempt.insert_run(
            run_id=run_id, started_at_ms=started_at_ms, ended_at_ms=ended_at_ms
        )
        await Attempt.insert_many(attempts=attempts, run_id=run_id)

    top_two = get_best_attempts(
        attempts=attempts, k_top=2, unique_code=True, unique_output=True
    )
    # record for export later
    try:
        set_top_two_attempts_for(challenge.id, top_two)
    except Exception:
        pass
    
    if len(top_two) == 0:
        # TODO: LLM call failed. Return empty solutions
        return [ ([], 0.0), ([], 0.0) ]

    if len(top_two) == 1:
        top_two.append(top_two[0])

    # TODO: only add primitive if top one scores better than best primitive on this task
    if library and top_two:
        current_primitive_count = len(library.primitives)
        library.add_primitive(Primitive(id=f"{current_primitive_count}", python_code_str=top_two[0].python_code_str))

    first_solution = top_two[0]
    second_solution = top_two[1]

    if PLOT:
        first_solution.plot(ignore_fixing=True)
        second_solution.plot(ignore_fixing=True)

    return [ ( get_grids_from_attempt(first_solution), first_solution.train_accuracy), (get_grids_from_attempt(
        second_solution
    ), second_solution.train_accuracy) ]


async def solve_challenge_server(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    env_vars: dict[str, str],
) -> tuple[list[GRID], list[GRID]]:
    for k, v in env_vars.items():
        os.environ[k] = v
    res = await solve_challenge(tree=tree, challenge=challenge)
    for k in env_vars.keys():
        del os.environ[k]
    return res


class CacheData(BaseModel):
    redis_dsn: str
    run_id: str


class ChallengeStatus(str, Enum):
    queued = "queued"
    running = "running"
    errored = "errored"
    done = "done"


class ChallengeItem(BaseModel):
    status: ChallengeStatus
    queued_at_ms: float
    started_at_ms: float | None
    errored_at_ms: float | None
    done_at_ms: float | None
    solution_attempts: tuple[list[GRID], list[GRID]] | None
    last_ping_at_ms: float | None


async def solve_challenge_background(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    cache_data: CacheData,
    environ_data: dict[str, str],
    url: str = None,
) -> tuple[list[GRID], list[GRID]]:
    if url:
        async with httpx.AsyncClient(timeout=3600) as client:
            r = await client.post(
                url,
                json={
                    "tree": TypeAdapter(list[RootAttemptConfig]).dump_python(
                        tree, mode="json"
                    ),
                    "challenge": Challenge.model_dump(challenge, mode="json"),
                    "cache_data": cache_data.model_dump(mode="json"),
                    "environ_data": environ_data,
                },
            )
            j = r.json()
            debug(j)
            # TODO run retry logic here?
        return j

    for k, v in environ_data.items():
        os.environ[k] = v

    redis_client = redis.Redis.from_url(cache_data.redis_dsn)

    # confirm it hasn't already been done or solved
    key = f"{cache_data.run_id}:{challenge.id}"
    challenge_item = await redis_client.get(key)
    if not challenge_item:
        now = time.time() * 1000
        challenge_item = ChallengeItem(
            status=ChallengeStatus.running,
            queued_at_ms=now,
            started_at_ms=now,
            errored_at_ms=None,
            done_at_ms=None,
            solution_attempts=None,
            last_ping_at_ms=now,
        )
    else:
        challenge_item = ChallengeItem.model_validate_json(challenge_item)
        if challenge_item.status not in [
            ChallengeStatus.queued,
            ChallengeStatus.errored,
        ]:
            raise Exception(f"Invalid challenge status: {challenge_item.status.value}")
        now = time.time() * 1000
        challenge_item = ChallengeItem(
            status=ChallengeStatus.running,
            queued_at_ms=challenge_item.queued_at_ms,
            started_at_ms=now,
            errored_at_ms=challenge_item.errored_at_ms,
            done_at_ms=challenge_item.done_at_ms,
            solution_attempts=challenge_item.solution_attempts,
            last_ping_at_ms=now,
        )

    await redis_client.set(key, challenge_item.model_dump_json())
    try:
        solution_attempts = await solve_challenge(tree=tree, challenge=challenge)
        challenge_item = ChallengeItem.model_validate_json(await redis_client.get(key))
        now = time.time() * 1000
        await redis_client.set(
            key,
            ChallengeItem(
                status=ChallengeStatus.done,
                queued_at_ms=challenge_item.queued_at_ms,
                started_at_ms=challenge_item.started_at_ms,
                errored_at_ms=challenge_item.errored_at_ms,
                done_at_ms=now,
                solution_attempts=solution_attempts,
                last_ping_at_ms=now,
            ).model_dump_json(),
        )
        return solution_attempts
    except Exception as e:
        logfire.debug(f"ERROR CATCHING ATTEMPTS: {e=}, {traceback.format_exc()}")
        now = time.time() * 1000
        await redis_client.set(
            key,
            ChallengeItem(
                status=ChallengeStatus.errored,
                queued_at_ms=challenge_item.queued_at_ms,
                started_at_ms=challenge_item.started_at_ms,
                errored_at_ms=now,
                done_at_ms=challenge_item.done_at_ms,
                solution_attempts=challenge_item.solution_attempts,
                last_ping_at_ms=now,
            ).model_dump_json(),
        )
        raise e
