import asyncio
import random
import pickle
import os
import argparse
import json
import time
from collections import defaultdict
import typing as T

from devtools import debug

from pathlib import Path

from src.data import eval_challenges, training_challenges, v2_training_challenges, v2_eval_challenges, build_challenges_v2, build_challenges
from src.logic import solve_challenge_with_accuracy
from src.models import GRID
from src import logfire

from pydantic import BaseModel, TypeAdapter

def solve_challenge_in_process(challenge_id: str, challenge_dict: dict, library_dict: dict, env_vars: dict) -> tuple[str, bool, dict]:
    """Solve a single challenge in a separate process for true parallelism.
    Returns (challenge_id, solved, solution_data)
    """
    import asyncio
    import os
    
    # Set environment variables in the process
    for key, value in env_vars.items():
        if value is not None:
            os.environ[key] = str(value)
    
    # Import after setting env vars to avoid JAX issues
    from src.logic import solve_challenge_with_accuracy
    from src.trees.experiments import grokfast_dreamcoder_tree
    from src.models import Library, Primitive, Challenge, Example
    
    try:
        # Reconstruct objects from serialized data
        challenge = Challenge(
            id=challenge_dict['id'],
            train=[Example(input=t['input'], output=t['output']) for t in challenge_dict['train']],
            test=[Example(input=t['input'], output=t.get('output')) for t in challenge_dict['test']]
        )
        
        library = Library(primitives=[
            Primitive(id=p['id'], python_code_str=p['code']) 
            for p in library_dict['primitives']
        ]) if library_dict else None
        
        async def _async_solve():
            try:
                total_cost = [0.0]  # Mutable list for cost tracking
                solutions_and_accuracies = await solve_challenge_with_accuracy(
                    challenge=challenge,
                    tree=grokfast_dreamcoder_tree,
                    library=library,
                    use_primitives_weighed_by_score=True,  # Use the two-pass approach
                    lpn_model=None,
                    evaluator=None,
                    key=None,
                    challenge_primitive_lpn_scores={},
                    challenge_primitive_accuracy_scores={},
                    aggregate_cost_in_cents=total_cost,
                )
                
                first_solutions_and_accuracy, second_solutions_and_accuracy = solutions_and_accuracies[0], solutions_and_accuracies[1]
                first_solutions, first_accuracy = first_solutions_and_accuracy
                second_solutions, second_accuracy = second_solutions_and_accuracy
                
                # Check if solved (both attempts have 100% accuracy)
                solved = (first_accuracy == 1.0 and second_accuracy == 1.0)
                
                solution_data = {
                    'first_solutions': first_solutions,
                    'second_solutions': second_solutions, 
                    'first_accuracy': first_accuracy,
                    'second_accuracy': second_accuracy,
                    'cost': total_cost[0]
                }
                
                return challenge_id, solved, solution_data
                
            except Exception as e:
                print(f"[{challenge_id}] Process async error: {e}")
                import traceback
                traceback.print_exc()
                return challenge_id, False, {'error': str(e)}
        
        # Run async function in process
        return asyncio.run(_async_solve())
        
    except Exception as e:
        print(f"[{challenge_id}] Process setup error: {e}")
        import traceback
        traceback.print_exc()
        return challenge_id, False, {'error': str(e)}

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.run_python import run_python_transform_sync
from src.logic import run_python_transform_sync_cached
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Use spawn method to avoid JAX/fork issues
multiprocessing.set_start_method('spawn', force=True)

class ChallengeSolutionWithAccuracy(BaseModel):
    attempt_1: list[GRID]
    attempt_2: list[GRID]
    accuracy_1: float
    accuracy_2: float

class ChallengeSolution(BaseModel):
    attempt_1: GRID
    attempt_2: GRID

def load_lpn_model(artifact_path: str) -> tuple[T.Any, T.Any, T.Any]:
    # Lazy imports so baseline runs don’t require JAX/Flax/etc.
    import wandb
    import omegaconf
    import hydra
    from flax.serialization import from_bytes
    import jax
    import optax
    from flax.training.train_state import TrainState
    from lpn.src.models.transformer import EncoderTransformer, DecoderTransformer
    from lpn.src.models.lpn import LPN
    from lpn.src.evaluator import Evaluator

    run = wandb.init()
    artifact = run.use_artifact(artifact_path, type='model')
    run.finish()
    cfg = omegaconf.OmegaConf.create(artifact.metadata)
    artifact_dir = artifact.download()
    omegaconf.OmegaConf.save(config=cfg, f=os.path.join(artifact_dir, "config.yaml"))

    cfg.encoder_transformer['_target_'] = 'lpn.src.models.utils.EncoderTransformerConfig'
    cfg.encoder_transformer.transformer_layer['_target_'] = 'lpn.src.models.utils.TransformerLayerConfig'

    cfg.decoder_transformer['_target_'] = 'lpn.src.models.utils.DecoderTransformerConfig'
    cfg.decoder_transformer.transformer_layer['_target_'] = 'lpn.src.models.utils.TransformerLayerConfig'

    encoder = EncoderTransformer(hydra.utils.instantiate(cfg.encoder_transformer))
    decoder = DecoderTransformer(hydra.utils.instantiate(cfg.decoder_transformer))

    lpn_model = LPN(encoder=encoder, decoder=decoder)

    key = jax.random.PRNGKey(0)
    grids = jax.random.randint(
        key, (1, 3, decoder.config.max_rows, decoder.config.max_cols, 2), minval=0, maxval=decoder.config.vocab_size,
    )
    shapes = jax.random.randint(
        key, (1, 3, 2, 2), minval=1, maxval=min(decoder.config.max_rows, decoder.config.max_cols) + 1,
    )
    variables = lpn_model.init(key, grids, shapes, dropout_eval=False, prior_kl_coeff=0.0, pairwise_kl_coeff=0.0, mode="mean")
    evaluator = Evaluator(
        lpn_model,
        inference_mode="gradient_ascent",
        inference_mode_kwargs={
            "num_steps": 200,
            "lr": 1.0,
            "lr_schedule": True,
            "optimizer": "adam",
            "optimizer_kwargs": {"b2": 0.9},
            "accumulate_gradients_decoder_pairs": True,
        },
    )

    learning_rate, linear_warmup_steps = 0, 0
    linear_warmup_scheduler = optax.warmup_exponential_decay_schedule(
        init_value=learning_rate / (linear_warmup_steps + 1),
        peak_value=learning_rate,
        warmup_steps=linear_warmup_steps,
        transition_steps=1,
        end_value=learning_rate,
        decay_rate=1.0,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(linear_warmup_scheduler))
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)
    train_state = TrainState.create(apply_fn=lpn_model.apply, tx=optimizer, params=variables["params"])

    ckpt_path = "state.msgpack"

    with open(os.path.join(artifact_dir, ckpt_path), "rb") as data_file:
        byte_data = data_file.read()
    loaded_state = from_bytes(train_state, byte_data)

    # Explicitly bind the model to the parameters
    bound_lpn_model = lpn_model.bind({'params': loaded_state.params})
    return bound_lpn_model, evaluator, key


async def main() -> None:
    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval", action="store_true", help="Use saved library")
    parser.add_argument("-l", "--lpn", type=str, help="Use LPN model")
    parser.add_argument("-v1", "--version1", action="store_true", help="Test on ARC-AGI-1 public eval set")
    parser.add_argument("-p", "--path", type=str, help="Input dataset path (JSON file or directory)")
    parser.add_argument("--precheck-only", action="store_true", help="Only run library precheck; no LLM calls (requires -e)")
    args = parser.parse_args()

    if args.precheck_only and not args.eval:
        print("Error: --precheck-only requires -e/--eval (resume from a saved library).")
        return

    if args.lpn:
        artifact_path = os.getenv("LPN_ARTIFACT_PATH")
        lpn_model, evaluator, key = load_lpn_model(artifact_path)
    else:
        lpn_model, evaluator, key = None, None, None

    num_correct: int = 0
    num_tested: int = 0
    total_cost_in_cents: list[float] = [ 0.0 ]

    start_time = time.perf_counter()
    grace_seconds = (11 * 3600) + (45 * 60)
    hard_stop_seconds = (11 * 3600) + (50 * 60)
    grace_triggered = False
    hard_stop_time = start_time + hard_stop_seconds

    if args.version1:
        challenges = eval_challenges
    elif args.path:
        print(f"Building challenges from {args.path}")
        path = Path(args.path)
        if path.is_dir():
            # Load all *.json files recursively from the directory
            challenges = build_challenges_v2(
                challenges_path=path,
            )
        else:
            # Load a single JSON file
            challenges = build_challenges(
                challenges_path=path,
                solutions_path=None,
            )
    else:
        challenges = v2_eval_challenges

    eval_ids_to_test = list(challenges.keys())

    # Optional: slice tasks via env (supports START+COUNT)
    start_env = os.environ.get("SUBMISSION_TASKS_START")
    count_env = os.environ.get("SUBMISSION_TASKS_COUNT")
    try:
        start_i = max(0, int(start_env)) if start_env is not None else 0
    except Exception:
        start_i = 0
    try:
        count_i = max(0, int(count_env)) if count_env is not None else None
    except Exception:
        count_i = None
    if start_i or count_i is not None:
        end_i = start_i + count_i if count_i is not None else None
        eval_ids_to_test = eval_ids_to_test[start_i:end_i]
        print(f"Applying task slice: start={start_i}, count={count_i if count_i is not None else 'ALL'}")
        logfire.debug(f"Applying task slice: start={start_i}, count={count_i}")

    print(f"eval set size: {len(eval_ids_to_test)}")
    logfire.debug(f"eval set size: {len(eval_ids_to_test)}")
    debug(eval_ids_to_test)

    from src.trees.experiments import grokfast_dreamcoder_tree
    from src.models import Library

    # Function to load library
    def load_library(filename="saved_library.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            print(f"No library file found at {filename}, creating new library")
            return Library(primitives=[])

    # Function to save library
    def save_library(library, filename="saved_library.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(library, f)
        print(f"Library saved to {filename}")

    def load_challenge_primitive_accuracy_scores(filename="challenge_primitive_accuracy_scores.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            return defaultdict(dict)

    def save_challenge_primitive_accuracy_scores(d, filename="challenge_primitive_accuracy_scores.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump(d, f)
            print(f"Saved challenge_primitive_accuracy_scores to {filename}")
        except Exception as e:
            print(f"WARNING: failed to save challenge_primitive_accuracy_scores: {e}")

    # Only use library if -e flag is provided
    library = None
    if args.eval:
        # library of primitives that have been obtained from arc-agi 2 train set (1000 challenges)
        library_path = os.environ.get("SUBMISSION_LIBRARY_PATH", "saved_library_1000.pkl")
        print(f"Loading library from {library_path}")
        library = load_library(library_path)
    else:
        library = Library(primitives=[])

    print(f"library size: {len(library.primitives)}")
    logfire.debug(f"library size: {len(library.primitives)}")

    solved_challenges = set()
    # Dictionary to store primitive lpn scores for each challenge (scores don't change across runs)
    challenge_primitive_lpn_scores = defaultdict(dict)
    # Dictionary to store primitive naive accuracy scores (how many squares it gets correct)
    challenge_primitive_accuracy_scores = load_challenge_primitive_accuracy_scores()
    #print(f"challenge_primitive_accuracy_scores length: {len(challenge_primitive_accuracy_scores)}")
    intermediate_solutions_d: dict[str, ChallengeSolutionWithAccuracy] = {}
    solutions_d: dict[str, list[ChallengeSolution]] = {}

    def _load_previous_submission_state() -> tuple[dict[str, list[ChallengeSolution]], set[str]]:
        prior_solutions: dict[str, list[ChallengeSolution]] = {}
        solved_ids: set[str] = set()
        solved_ids_file = Path("solved_ids.json")
        if solved_ids_file.exists():
            try:
                raw_ids = json.loads(solved_ids_file.read_text(encoding="utf-8"))
                if isinstance(raw_ids, list):
                    solved_ids.update(str(cid) for cid in raw_ids)
            except Exception as e:
                print(f"WARNING: failed to read solved_ids.json: {e}")
        submission_path = Path("submission.json")
        if submission_path.exists():
            try:
                adapter = TypeAdapter(dict[str, list[ChallengeSolution]])
                stored = adapter.validate_json(submission_path.read_text(encoding="utf-8"))
                for cid, entries in stored.items():
                    if cid not in challenges:
                        continue
                    prior_solutions[cid] = entries
                    if cid in solved_ids:
                        continue
                    challenge = challenges[cid]
                    expected = [t.output for t in challenge.test]
                    if not expected or any(e is None for e in expected):
                        continue
                    for entry in entries:
                        attempt1_ok = len(entry.attempt_1) == len(expected) and all(
                            pred == exp for pred, exp in zip(entry.attempt_1, expected)
                        )
                        attempt2_ok = len(entry.attempt_2) == len(expected) and all(
                            pred == exp for pred, exp in zip(entry.attempt_2, expected)
                        )
                        if attempt1_ok or attempt2_ok:
                            solved_ids.add(cid)
                            break
            except Exception as e:
                print(f"WARNING: failed to inspect submission.json for solved challenges: {e}")
        return prior_solutions, solved_ids

    previous_solutions, preexisting_solved_ids = _load_previous_submission_state()
    if previous_solutions:
        for cid, entries in previous_solutions.items():
            solutions_d[cid] = entries
    if preexisting_solved_ids:
        solved_challenges.update(preexisting_solved_ids)
        print(f"Loaded {len(preexisting_solved_ids)} previously solved challenges.")

    # Process-based challenge execution
    _challenge_executor: ProcessPoolExecutor | None = None
    
    def _get_challenge_executor() -> ProcessPoolExecutor:
        nonlocal _challenge_executor
        if _challenge_executor is None:
            # Keep primitive evaluation to 5 processes, but allow more challenges in pipeline
            max_workers = min(5, max(4, round1_batch_size))
            _challenge_executor = ProcessPoolExecutor(max_workers=max_workers)
        return _challenge_executor
    

    async def try_solve_challenge(challenge_id: str, solved_challenges: set[str], total_cost_in_cents: float) -> bool:
        if challenge_id in solved_challenges:
            print(f"Challenge {challenge_id} already solved")
            logfire.debug(f"Challenge {challenge_id} already solved")
            return True
        debug(challenge_id)
        print(f"value length: {len(challenge_primitive_accuracy_scores[challenge_id])}")
        challenge = challenges[challenge_id]

        # Library-only precheck: if -e (eval) and LIBRARY_PRECHECK != '0', try to solve using existing primitives without any LLM calls.
        solutions_and_accuracies = None
        if args.eval and os.environ.get("LIBRARY_PRECHECK", "1") != "0" and library and library.primitives:
            # Evaluate each primitive on training examples; if one achieves perfect train accuracy, use it to generate test outputs.
            for primitive in library.primitives:
                try:
                    tr = run_python_transform_sync_cached(
                        code=primitive.python_code_str,
                        grid_lists=[train.input for train in challenge.train],
                        timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
                        raise_exception=False,
                    )
                    if tr and tr.transform_results and len(tr.transform_results) == len(challenge.train):
                        train_ok = True
                        for idx, train in enumerate(challenge.train):
                            if tr.transform_results[idx] != train.output:
                                train_ok = False
                                break
                        if train_ok:
                            # Generate test predictions with the same primitive
                            test_predictions: list[GRID] = []
                            for test in challenge.test:
                                tr_test = run_python_transform_sync_cached(
                                    code=primitive.python_code_str,
                                    grid_lists=[test.input],
                                    timeout=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
                                    raise_exception=False,
                                )
                                if tr_test and tr_test.transform_results:
                                    test_predictions.append(tr_test.transform_results[0])
                                else:
                                    test_predictions.append([[0]])
                            solutions_and_accuracies = [ (test_predictions, 1.0), (test_predictions, 1.0) ]
                            print(f"[{challenge.id}] solved via library precheck using primitive {getattr(primitive, 'id', '?')}")
                            logfire.debug(f"[{challenge.id}] solved via library precheck using primitive {getattr(primitive, 'id', '?')}")
                            break
                except Exception as e:
                    logfire.debug(f"[{challenge.id}] library precheck primitive {getattr(primitive, 'id', '?')} error: {e}")
                    continue

        if solutions_and_accuracies is None:
            if args.precheck_only:
                # Skip LLM-backed solving; synthesize placeholders of correct length
                num_tests = len(challenge.test)
                placeholder = [[0]]
                test_predictions = [placeholder for _ in range(num_tests)]
                solutions_and_accuracies = [ (test_predictions, 0.0), (test_predictions, 0.0) ]
                print(f"[{challenge.id}] precheck-only: no library match; skipping LLM and using placeholders")
                logfire.debug(f"[{challenge.id}] precheck-only: no library match; skipping LLM and using placeholders")
            else:
                solutions_and_accuracies = await solve_challenge_with_accuracy(
                    challenge=challenge,
                    tree=grokfast_dreamcoder_tree,
                    library=library,
                    use_primitives_weighed_by_score=not lpn_model,
                    lpn_model=lpn_model,
                    evaluator=evaluator,
                    key=key,
                    challenge_primitive_lpn_scores=challenge_primitive_lpn_scores,
                    challenge_primitive_accuracy_scores=challenge_primitive_accuracy_scores,
                    aggregate_cost_in_cents=total_cost_in_cents,
                )

        logfire.debug(f"[{challenge.id}] solutions_and_accuracies: {solutions_and_accuracies}")
        print(f"[{challenge.id}] solutions_and_accuracies: {solutions_and_accuracies}")

        first_solutions_and_accuracy, second_solutions_and_accuracy = solutions_and_accuracies[0], solutions_and_accuracies[1]

        logfire.debug(f"[{challenge.id}] tuple 1: {first_solutions_and_accuracy}, tuple 2: {second_solutions_and_accuracy}")
        print(f"[{challenge.id}] tuple 1: {first_solutions_and_accuracy}, tuple 2: {second_solutions_and_accuracy}")

        first_solutions, first_accuracy = first_solutions_and_accuracy
        second_solutions, second_accuracy = second_solutions_and_accuracy

        logfire.debug(f"[{challenge.id}] first_solutions: {first_solutions}, first_accuracy: {first_accuracy}")
        logfire.debug(f"[{challenge.id}] second_solutions: {second_solutions}, second_accuracy: {second_accuracy}")
        print(f"[{challenge.id}] first_solutions: {first_solutions}, first_accuracy: {first_accuracy}")
        print(f"[{challenge.id}] second_solutions: {second_solutions}, second_accuracy: {second_accuracy}")

        first_solution_correct_length, second_solution_correct_length = True, True
        if len(challenge.test) != len(first_solutions):
            print(f"[{challenge.id}] first_solutions have len {len(first_solutions)} but challenge.test has len {len(challenge.test)}")
            first_solution_correct_length = False
        if len(challenge.test) != len(second_solutions):
            print(f"[{challenge.id}] second_solutions have len {len(second_solutions)} but challenge.test has len {len(challenge.test)}")
            second_solution_correct_length = False

        if challenge.id not in intermediate_solutions_d:
            if first_solution_correct_length and second_solution_correct_length:
                intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                    attempt_1=first_solutions,
                    attempt_2=second_solutions,
                    accuracy_1=first_accuracy,
                    accuracy_2=second_accuracy,
                )
            elif first_solution_correct_length:
                intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                    attempt_1=first_solutions,
                    attempt_2=first_solutions,
                    accuracy_1=first_accuracy,
                    accuracy_2=first_accuracy,
                )
            elif second_solution_correct_length:
                intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                    attempt_1=second_solutions,
                    attempt_2=second_solutions,
                    accuracy_1=second_accuracy,
                    accuracy_2=second_accuracy,
                )
            else:
                raise ValueError(f"[{challenge.id}] first_solution_correct_length: {first_solution_correct_length}, second_solution_correct_length: {second_solution_correct_length}. Challenge test length: {len(challenge.test)}")
        else:
            old_solutions = intermediate_solutions_d[challenge.id]
            lst = [ [old_solutions.accuracy_1, old_solutions.attempt_1], 
                   [old_solutions.accuracy_2, old_solutions.attempt_2] ]
            if first_solution_correct_length:
                lst.append([first_accuracy, first_solutions])
            if second_solution_correct_length:
                lst.append([second_accuracy, second_solutions])
            lst.sort(key=lambda x: x[0], reverse=True)
            intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                attempt_1=lst[0][1],
                attempt_2=lst[1][1],
                accuracy_1=lst[0][0],
                accuracy_2=lst[1][0],
            )

        solutions_d[challenge.id] = []
        for i in range(len(challenge.test)):
            solutions_d[challenge.id].append(
                ChallengeSolution(
                    attempt_1=intermediate_solutions_d[challenge.id].attempt_1[i],
                    attempt_2=intermediate_solutions_d[challenge.id].attempt_2[i],
                )
            )

        if first_accuracy == 1.0 and second_accuracy == 1.0:
            return True
        else:
            return False

    # Allow controlling rounds via env var (defaults to 2)
    rounds_env = os.environ.get("SUBMISSION_ROUNDS", "2")
    try:
        total_rounds = max(1, int(rounds_env))
    except Exception:
        total_rounds = 2
    if args.precheck_only:
        total_rounds = 1
        print("Precheck-only mode enabled: skipping LLM-backed solving and forcing SUBMISSION_ROUNDS=1")
        logfire.debug("Precheck-only mode enabled: skipping LLM-backed solving and forcing SUBMISSION_ROUNDS=1")
    print(f"Using SUBMISSION_ROUNDS={total_rounds}")
    logfire.debug(f"Using SUBMISSION_ROUNDS={total_rounds}")

    # Optional: override attempts per task (affects per-task concurrency)
    attempts_env = os.environ.get("SUBMISSION_ATTEMPTS")
    if attempts_env:
        try:
            attempts_override = max(1, int(attempts_env))
            from src.trees.experiments import grokfast_dreamcoder_tree as _tree_ref
            for node in _tree_ref:
                node.attempts = attempts_override
            print(f"Using SUBMISSION_ATTEMPTS={attempts_override}")
            logfire.debug(f"Using SUBMISSION_ATTEMPTS={attempts_override}")
        except Exception as _:
            print("WARNING: invalid SUBMISSION_ATTEMPTS; ignoring")
            logfire.debug("WARNING: invalid SUBMISSION_ATTEMPTS; ignoring")

    # Dynamic batch sizing strategy:
    # Round 1: Small batches (5 at a time) through ALL challenges to start LLM calls quickly
    # Round 2+: Large batches (all at once) for maximum parallelism
    bs_env = os.environ.get("SUBMISSION_BATCH_SIZE")
    try:
        round1_batch_size = max(1, int(bs_env)) if bs_env else 20  # Increased from 5 to keep pipeline full
    except Exception:
        round1_batch_size = 20
    
    # Larger batch size for rounds 2+ (can be overridden)
    large_bs_env = os.environ.get("SUBMISSION_LARGE_BATCH_SIZE")
    try:
        later_rounds_batch_size = max(round1_batch_size, int(large_bs_env)) if large_bs_env else len(eval_ids_to_test)
    except Exception:
        later_rounds_batch_size = len(eval_ids_to_test)  # Process all at once in later rounds
    
    print(f"Using dynamic batch sizing: round 1 = {round1_batch_size} at a time, round 2+ = {later_rounds_batch_size} all at once")
    logfire.debug(f"Dynamic batch sizing: round 1 = {round1_batch_size} at a time, round 2+ = {later_rounds_batch_size} all at once")

    disable_low_solve_stop_env = os.environ.get("SUBMISSION_DISABLE_LOW_SOLVE_STOP", "0").lower()
    low_solve_stop_enabled = disable_low_solve_stop_env not in {"1", "true", "yes"}
    low_solve_stop_triggered = False

    time_exhausted = False

    for i in range(total_rounds):
        if time.perf_counter() >= hard_stop_time:
            print("Hard stop already reached before starting round; ending early.")
            logfire.debug("Hard stop already reached before starting round; ending early.")
            time_exhausted = True
            break
        # Track per-round count of cases that are perfect on train but wrong on test
        train_perfect_but_test_wrong: set[str] = set()
        solved_before_round_snapshot = set(solved_challenges)
        round_new_solved_ids: set[str] = set()

        # Dynamic batch sizing: small batches for round 1, large batches for rounds 2+
        current_batch_size = round1_batch_size if i == 0 else later_rounds_batch_size
        print(f"Round {i+1}: using batch size {current_batch_size}")
        logfire.debug(f"Round {i+1}: using batch size {current_batch_size}")
        
        # Sliding window concurrency: keep up to current_batch_size tasks in-flight
        sem = asyncio.Semaphore(current_batch_size)
        processed = 0

        async def bounded_try(challenge_id: str):
            async with sem:
                try:
                    if challenge_id in solved_challenges:
                        return (challenge_id, True, None)
                    
                    # Serialize challenge and library data for process communication
                    challenge_dict = {
                        'id': challenge_id,
                        'train': [{'input': t.input, 'output': t.output} for t in challenges[challenge_id].train],
                        'test': [{'input': t.input, 'output': t.output} for t in challenges[challenge_id].test]
                    }
                    
                    library_dict = {
                        'primitives': [{
                            'id': getattr(p, 'id', f'prim_{i}'),
                            'code': p.python_code_str
                        } for i, p in enumerate(library.primitives)] if library else []
                    } if library else None
                    
                    # Environment variables to pass to process
                    env_vars = {
                        'SUBMISSION_TWO_PASS_ENABLED': os.environ.get('SUBMISSION_TWO_PASS_ENABLED', '1'),
                        'SUBMISSION_FIRST_PASS_TOP_K': os.environ.get('SUBMISSION_FIRST_PASS_TOP_K', '50'),
                        'SUBMISSION_TRANSFORM_TIMEOUT': os.environ.get('SUBMISSION_TRANSFORM_TIMEOUT', '5'),
                        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
                        'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY'),
                        'LOGFIRE_TOKEN': os.environ.get('LOGFIRE_TOKEN')
                    }
                    
                    # Execute in separate process
                    loop = asyncio.get_running_loop()
                    executor = _get_challenge_executor()
                    challenge_id, solved, solution_data = await loop.run_in_executor(
                        executor,
                        solve_challenge_in_process,
                        challenge_id, 
                        challenge_dict,
                        library_dict,
                        env_vars
                    )
                    
                    if 'error' not in solution_data:
                        # Update solutions_d with results
                        first_solutions = solution_data['first_solutions']
                        second_solutions = solution_data['second_solutions']
                        first_accuracy = solution_data['first_accuracy']
                        second_accuracy = solution_data['second_accuracy']
                        total_cost_in_cents[0] += solution_data['cost']
                        
                        # Store solution data (similar to original try_solve_challenge logic)
                        intermediate_solutions_d[challenge_id] = ChallengeSolutionWithAccuracy(
                            attempt_1=first_solutions,
                            attempt_2=second_solutions,
                            accuracy_1=first_accuracy,
                            accuracy_2=second_accuracy,
                        )
                        
                        solutions_d[challenge_id] = []
                        for i in range(len(challenges[challenge_id].test)):
                            solutions_d[challenge_id].append(
                                ChallengeSolution(
                                    attempt_1=first_solutions[i] if i < len(first_solutions) else [[0]],
                                    attempt_2=second_solutions[i] if i < len(second_solutions) else [[0]],
                                )
                            )
                    
                    return (challenge_id, solved, None)
                    
                except Exception as e:
                    return (challenge_id, None, e)

        # Helper to compute whether train was perfect but test answers were wrong
        def _is_train_perfect_but_test_wrong(cid: str) -> bool:
            try:
                ch = challenges[cid]
                inter = intermediate_solutions_d.get(cid)
                if not inter:
                    return False
                # Train-perfect if any chosen attempt had accuracy == 1.0
                train_perfect = (inter.accuracy_1 == 1.0) or (inter.accuracy_2 == 1.0)
                if not train_perfect:
                    return False
                # Compute test correctness for either attempt matching fully
                # Build expected outputs list
                expected = [t.output for t in ch.test]
                # If test outputs are unknown/withheld, we cannot evaluate this metric
                if any(e is None for e in expected):
                    return False
                # Check attempt 1
                a1_ok = len(inter.attempt_1) == len(expected) and all(
                    pred == exp for pred, exp in zip(inter.attempt_1, expected)
                )
                # Check attempt 2
                a2_ok = len(inter.attempt_2) == len(expected) and all(
                    pred == exp for pred, exp in zip(inter.attempt_2, expected)
                )
                return (not (a1_ok or a2_ok))
            except Exception as _e:
                try:
                    logfire.debug(f"Error computing train-perfect-but-test-wrong for {cid}: {_e}")
                except Exception:
                    pass
                return False

        eval_iter = iter(eval_ids_to_test)
        active_tasks: dict[asyncio.Task, str] = {}
        ids_exhausted = False
        hard_timeout_reached = False

        while active_tasks or not ids_exhausted:
            now = time.perf_counter()
            elapsed = now - start_time

            if (not grace_triggered) and elapsed >= grace_seconds:
                grace_triggered = True
                print("Run has reached the 11h45m mark. Pausing new challenge launches and allowing in-flight work up to the hard stop.")
                logfire.debug("Grace period trigger reached at %.2fh" % (elapsed / 3600.0))

            if grace_triggered and now >= hard_stop_time:
                print("Hard stop reached at 11h50m. Cancelling remaining work and finalizing submission.")
                logfire.debug("Hard stop reached with %d active tasks" % len(active_tasks))
                hard_timeout_reached = True
                break

            while (not grace_triggered) and (len(active_tasks) < current_batch_size) and (not ids_exhausted):
                try:
                    cid = next(eval_iter)
                except StopIteration:
                    ids_exhausted = True
                    break
                task = asyncio.create_task(bounded_try(cid))
                active_tasks[task] = cid

            if not active_tasks:
                if ids_exhausted:
                    break
                if grace_triggered:
                    # No work in flight and we are in grace mode; exit round.
                    break
                continue

            wait_timeout = None
            if grace_triggered:
                wait_timeout = max(0.0, hard_stop_time - now)

            wait_set = set(active_tasks.keys())
            done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED, timeout=wait_timeout)

            if not done:
                hard_timeout_reached = True
                break

            for task in done:
                cid = active_tasks.pop(task, None)
                if cid is None:
                    continue
                try:
                    challenge_id, solved, err = task.result()
                except asyncio.CancelledError:
                    continue
                processed += 1
                if err is not None:
                    print(f"Error solving challenge {challenge_id}: {err}")
                    logfire.debug(f"Error solving challenge {challenge_id}: {err}")
                elif solved:
                    was_previously_solved = challenge_id in solved_challenges
                    if not was_previously_solved:
                        solved_challenges.add(challenge_id)
                    if (
                        challenge_id not in solved_before_round_snapshot
                        and challenge_id not in round_new_solved_ids
                    ):
                        round_new_solved_ids.add(challenge_id)
                else:
                    print(f"Challenge {challenge_id} not solved")
                    logfire.debug(f"Challenge {challenge_id} not solved")

                try:
                    if _is_train_perfect_but_test_wrong(challenge_id):
                        train_perfect_but_test_wrong.add(challenge_id)
                except Exception:
                    pass

                if processed % max(1, current_batch_size // 2) == 0 or processed == len(eval_ids_to_test):
                    print(f"Round {i+1} progress: {processed}/{len(eval_ids_to_test)} processed, {len(solved_challenges)} solved so far")
                    logfire.debug(f"Round {i+1} progress: {processed}/{len(eval_ids_to_test)} processed, {len(solved_challenges)} solved so far")

        if hard_timeout_reached:
            to_cancel = list(active_tasks.keys())
            if to_cancel:
                for task in to_cancel:
                    task.cancel()
                await asyncio.gather(*to_cancel, return_exceptions=True)
            time_exhausted = True
        elif grace_triggered and not ids_exhausted:
            to_cancel = list(active_tasks.keys())
            if to_cancel:
                for task in to_cancel:
                    task.cancel()
                await asyncio.gather(*to_cancel, return_exceptions=True)
            time_exhausted = True

        if time_exhausted:
            remaining = []
            try:
                while True:
                    remaining.append(next(eval_iter))
            except StopIteration:
                pass
            if remaining:
                print(f"Timing shutdown skipped {len(remaining)} remaining challenges in this round.")
                logfire.debug(f"Timing shutdown skipped challenges: {remaining}")

        if time_exhausted:
            # Still produce per-round summary before exiting outer loop
            pass

        # End-of-round summary
        logfire.debug(f"After {i+1} rounds, Solved Challenges: {solved_challenges}")
        print(f"After {i+1} rounds, Solved Challenges: {solved_challenges}")
        logfire.debug(f"After {i+1} rounds, Correct Percent SO FAR: {len(solved_challenges) / len(eval_ids_to_test)}")
        print(f"After {i+1} rounds, Correct Percent SO FAR: {len(solved_challenges) / len(eval_ids_to_test)}")
        print(f"After {i+1} rounds, Train-perfect-on-train but wrong-on-test: {len(train_perfect_but_test_wrong)}")
        logfire.debug(f"After {i+1} rounds, Train-perfect-on-train but wrong-on-test: {len(train_perfect_but_test_wrong)}")
        # Also list their IDs for inspection
        try:
            ids_list = sorted(train_perfect_but_test_wrong)
        except Exception:
            ids_list = list(train_perfect_but_test_wrong)
        print(f"After {i+1} rounds, Train-perfect-on-train but wrong-on-test IDs: {ids_list}")
        logfire.debug(f"After {i+1} rounds, Train-perfect-on-train but wrong-on-test IDs: {ids_list}")

        print("Saving library...")
        save_library(library, f"saved_library_eval_round_{i}.pkl")

        logfire.debug(f"After {i+1} rounds, Total cost in cents: {total_cost_in_cents[0]}")
        print(f"After {i+1} rounds, Total cost in cents: {total_cost_in_cents[0]}")

        print(
            f"After {i+1} rounds, New challenges solved this round: {len(round_new_solved_ids)}"
        )
        logfire.debug(
            f"After {i+1} rounds, New challenges solved this round: {len(round_new_solved_ids)}"
        )


        if time_exhausted:
            print("Timing limit reached; exiting remaining rounds early.")
            logfire.debug("Timing limit reached; exiting remaining rounds early.")
            break


    print(f"FINAL: Solved Challenges: {solved_challenges}")
    print(f"FINAL: Correct Percent: {len(solved_challenges) / len(eval_ids_to_test)}")

    print(f"FINAL: Total cost in cents: {total_cost_in_cents[0]}")

    total_runtime_seconds = time.perf_counter() - start_time
    print(
        "FINAL: Total runtime: "
        f"{total_runtime_seconds / 3600:.2f} hours ({total_runtime_seconds / 60:.1f} minutes)"
    )
    logfire.debug(f"FINAL: Total runtime seconds: {total_runtime_seconds}")

    try:
        Path("solved_ids.json").write_text(
            json.dumps(sorted(solved_challenges)),
            encoding="utf-8",
        )
        print("Saved solved challenge IDs to solved_ids.json")
    except Exception as e:
        print(f"WARNING: failed to persist solved_ids.json: {e}")

    # -------- Export unsolved best functions and summary (timestamped folder) --------
    from datetime import datetime, timezone
    from src.logic import get_top_two_attempts_for

    def _safe_ts() -> str:
        # Windows-safe timestamp like 2025-09-30T22-53-45Z
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    def _count_cell_mismatches(expected: list[list[int]], pred: list[list[int]]) -> int:
        try:
            if expected is None:
                return 0
            if pred is None:
                # treat as full mismatch if no prediction
                return len(expected) * (len(expected[0]) if expected else 0)
            if len(expected) != len(pred) or (expected and pred and len(expected[0]) != len(pred[0])):
                return len(expected) * (len(expected[0]) if expected else 0)
            mism = 0
            for i in range(len(expected)):
                row_e, row_p = expected[i], pred[i]
                for j in range(len(row_e)):
                    if row_e[j] != row_p[j]:
                        mism += 1
            return mism
        except Exception:
            # be conservative
            try:
                return len(expected) * (len(expected[0]) if expected else 0)
            except Exception:
                return 0

    def _total_mismatches_train(ch, attempt) -> int:
        try:
            total = 0
            for idx, train in enumerate(ch.train):
                pred = attempt.train_attempts[idx] if idx < len(attempt.train_attempts) else None
                total += _count_cell_mismatches(train.output, pred)
            return total
        except Exception:
            return 0

    def _total_mismatches_test(ch, preds: list[list[list[int]]]) -> tuple[int, bool]:
        # returns (total_mismatches, used_test)
        try:
            expected = [t.output for t in ch.test]
            if any(e is None for e in expected):
                return 0, False
            total = 0
            for idx, exp in enumerate(expected):
                pr = preds[idx] if idx < len(preds) else None
                total += _count_cell_mismatches(exp, pr)
            return total, True
        except Exception:
            return 0, False

    try:
        # Build timestamped directory
        ts_dir = os.path.join("unsolved", _safe_ts())
        os.makedirs(ts_dir, exist_ok=True)

        # Helpers
        def _fallback_library_code(cid: str):
            try:
                # Use saved accuracy scores to pick best primitive for this challenge
                scores_d = challenge_primitive_accuracy_scores.get(cid, {})
                if not scores_d:
                    return None, None
                # Pick by (num_train_correct, avg_train_accuracy)
                best_pid, (num_correct, avg_acc) = max(
                    scores_d.items(), key=lambda kv: (kv[1][0], kv[1][1])
                )
                for p in getattr(library, "primitives", []):
                    if getattr(p, "id", None) == best_pid:
                        return (p.python_code_str or ""), {
                            "primitive_id": best_pid,
                            "num_train_correct": num_correct,
                            "avg_train_accuracy": avg_acc,
                        }
                return None, None
            except Exception:
                return None, None

        # Build per-challenge test predictions mapping from solutions_d
        def _collect_test_preds(cid: str) -> tuple[list[GRID], list[GRID]]:
            lst = solutions_d.get(cid, [])
            attempt1_preds: list[GRID] = []
            attempt2_preds: list[GRID] = []
            for item in lst:
                try:
                    attempt1_preds.append(item.attempt_1)
                    attempt2_preds.append(item.attempt_2)
                except Exception:
                    pass
            return attempt1_preds, attempt2_preds

        unsolved_ids = sorted(list(set(eval_ids_to_test) - set(solved_challenges)))

        summary_rows = []  # for CSV
        best_min_list = []  # for average of min mismatches across unsolved

        for cid in unsolved_ids:
            ch = challenges[cid]
            top_two = get_top_two_attempts_for(cid) or []

            # align test predictions (may be empty)
            a1_preds, a2_preds = _collect_test_preds(cid)

            stats = []
            for idx, att in enumerate(top_two[:2]):
                try:
                    train_m = _total_mismatches_train(ch, att)
                    test_preds = a1_preds if idx == 0 else a2_preds
                    test_m, used_test = _total_mismatches_test(ch, test_preds)
                    combined = train_m + (test_m if used_test else 0)
                    stats.append({
                        "attempt_index": idx + 1,
                        "train_mismatches": train_m,
                        "test_mismatches": test_m,
                        "used_test": used_test,
                        "combined_mismatches": combined,
                        "code": att.python_code_str or "",
                    })
                except Exception:
                    pass

            # If no attempts captured, still write metadata so it’s visible; mark unknowns
            if not stats:
                # detect test availability
                used_test_any = 0
                try:
                    exp = [t.output for t in ch.test]
                    used_test_any = 0 if any(e is None for e in exp) else 1
                except Exception:
                    used_test_any = 0

                # Fallback: choose best from saved library by cached accuracies
                fb_code, fb_meta = _fallback_library_code(cid)
                if fb_code:
                    # Write best1 from library
                    base_fn1 = f"{cid}_best1.py"
                    with open(os.path.join(ts_dir, base_fn1), "w", encoding="utf-8") as f:
                        f.write(fb_code)
                    meta = {
                        "challenge_id": cid,
                        "fallback_library": True,
                        "library_selection": fb_meta,
                        "note": "no_llm_attempts_recorded_this_round",
                    }
                    with open(os.path.join(ts_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    summary_rows.append([cid, "", "", used_test_any])
                    continue

                # Otherwise write minimal metadata so it’s visible
                meta = {
                    "challenge_id": cid,
                    "note": "no_attempts_recorded_for_export",
                }
                with open(os.path.join(ts_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                summary_rows.append([cid, -1, -1, used_test_any])
                continue

            # Deduplicate by code (skip empty code strings)
            seen_codes: set[str] = set()
            deduped = []
            for s in sorted(stats, key=lambda s: (s["combined_mismatches"], s["train_mismatches"], len(s["code"]))):
                code_key = s["code"].strip()
                if not code_key:
                    continue
                if code_key in seen_codes:
                    continue
                seen_codes.add(code_key)
                deduped.append(s)
                if len(deduped) >= 2:
                    break

            # Ensure at least one candidate is written if any code exists; else write metadata only
            if not deduped:
                # Fallback to best library primitive if available
                fb_code, fb_meta = _fallback_library_code(cid)
                if fb_code:
                    base_fn1 = f"{cid}_best1.py"
                    with open(os.path.join(ts_dir, base_fn1), "w", encoding="utf-8") as f:
                        f.write(fb_code)
                    meta = {
                        "challenge_id": cid,
                        "fallback_library": True,
                        "library_selection": fb_meta,
                        "note": "no_nonempty_code_in_llm_attempts",
                        "attempts_count": len(stats),
                    }
                    with open(os.path.join(ts_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    summary_rows.append([cid, "", "", 1 if any(s.get("used_test") for s in stats) else 0])
                    continue

                meta = {
                    "challenge_id": cid,
                    "note": "no_nonempty_code_in_attempts",
                    "attempts_count": len(stats),
                }
                with open(os.path.join(ts_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                summary_rows.append([cid, -1, -1, 1 if any(s.get("used_test") for s in stats) else 0])
                continue

            best1 = deduped[0]
            best_min_list.append(best1["combined_mismatches"])  # for summary average

            # Write best1 code
            base_fn1 = f"{cid}_best1.py"
            with open(os.path.join(ts_dir, base_fn1), "w", encoding="utf-8") as f:
                f.write(best1["code"])  # code only, as requested

            # Optionally write best2 if present and different
            best2 = deduped[1] if len(deduped) > 1 else None
            if best2 is not None:
                base_fn2 = f"{cid}_best2.py"
                with open(os.path.join(ts_dir, base_fn2), "w", encoding="utf-8") as f:
                    f.write(best2["code"])  # code only

            # Write per-challenge JSON metadata
            meta = {
                "challenge_id": cid,
                "best1": {
                    "train_mismatches": best1["train_mismatches"],
                    "test_mismatches": best1["test_mismatches"],
                    "combined_mismatches": best1["combined_mismatches"],
                    "used_test": best1["used_test"],
                },
            }
            if best2 is not None:
                meta["best2"] = {
                    "train_mismatches": best2["train_mismatches"],
                    "test_mismatches": best2["test_mismatches"],
                    "combined_mismatches": best2["combined_mismatches"],
                    "used_test": best2["used_test"],
                }
            with open(os.path.join(ts_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            summary_rows.append([
                cid,
                best1["combined_mismatches"],
                (best2["combined_mismatches"] if best2 is not None else ""),
                1 if (best1["used_test"] or (best2 is not None and best2["used_test"])) else 0,
            ])

        # Write run-level summary
        summary = {
            "num_challenges": len(eval_ids_to_test),
            "num_solved": len(solved_challenges),
            "num_unsolved": len(unsolved_ids),
            "avg_min_mismatches_across_unsolved": (
                (sum(best_min_list) / len(best_min_list)) if best_min_list else 0.0
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(ts_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # CSV summary
        import csv
        with open(os.path.join(ts_dir, "summary.csv"), "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["challenge_id", "best1_combined_mismatches", "best2_combined_mismatches", "used_test_labels"])
            writer.writerows(summary_rows)

        print(f"Exported unsolved artifacts to {ts_dir}")
    except Exception as e:
        print(f"WARNING: failed to export unsolved artifacts: {e}")

    # Final summary of 'train-perfect but wrong on test' and remaining unsolved IDs not in that set
    try:
        final_train_perfect_but_test_wrong: set[str] = set()
        for cid in eval_ids_to_test:
            ch = challenges[cid]
            inter = intermediate_solutions_d.get(cid)
            if not inter:
                continue
            train_perfect = (inter.accuracy_1 == 1.0) or (inter.accuracy_2 == 1.0)
            if not train_perfect:
                continue
            expected = [t.output for t in ch.test]
            # If any expected output is unavailable/withheld, skip classification
            if any(e is None for e in expected):
                continue
            a1_ok = len(inter.attempt_1) == len(expected) and all(
                pred == exp for pred, exp in zip(inter.attempt_1, expected)
            )
            a2_ok = len(inter.attempt_2) == len(expected) and all(
                pred == exp for pred, exp in zip(inter.attempt_2, expected)
            )
            if not (a1_ok or a2_ok):
                final_train_perfect_but_test_wrong.add(cid)
        unsolved = set(eval_ids_to_test) - set(solved_challenges)
        remaining_not_in_fp = sorted(list(unsolved - final_train_perfect_but_test_wrong))
        print(f"FINAL: Unsolved and not train-perfect-but-test-wrong count: {len(remaining_not_in_fp)}")
        print(f"FINAL: Unsolved and not train-perfect-but-test-wrong IDs: {remaining_not_in_fp}")
        logfire.debug(f"FINAL: Unsolved and not train-perfect-but-test-wrong count: {len(remaining_not_in_fp)}")
        logfire.debug(f"FINAL: Unsolved and not train-perfect-but-test-wrong IDs: {remaining_not_in_fp}")
    except Exception as _e:
        try:
            logfire.debug(f"Error computing final unsolved-not-trainperfect list: {_e}")
        except Exception:
            pass

    # Persist primitive accuracy scores for reuse next runs
    save_challenge_primitive_accuracy_scores(challenge_primitive_accuracy_scores, "challenge_primitive_accuracy_scores.pkl")

    # Persist transform cache (if enabled)
    try:
        from src.logic import save_transform_cache
        save_transform_cache()
    except Exception:
        pass


    # finally, check if there are solutions to all challenges
    for challenge_id in eval_ids_to_test:
        if challenge_id not in solutions_d:
            print(f"Challenge {challenge_id} does not have a solution")
            logfire.debug(f"Challenge {challenge_id} does not have a solution")
            challenge = challenges[challenge_id]
            solutions_d[challenge_id] = []
            for _ in range(len(challenge.test)):
                solutions_d[challenge_id].append(
                    ChallengeSolution(
                        attempt_1=[[0]],
                        attempt_2=[[0]],
                    )
                )

    solutions_path = "submission.json"
    print(f"Saving solutions to {solutions_path}")
    open(solutions_path, "w").write(
        TypeAdapter(dict[str, list[ChallengeSolution]])
        .dump_json(solutions_d)
        .decode("utf-8")
    )


if __name__ == "__main__":
    asyncio.run(main())
