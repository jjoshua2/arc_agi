import asyncio
import random
import pickle
import os
import argparse
from collections import defaultdict
import typing as T

from devtools import debug

from pathlib import Path

from src.data import eval_challenges, training_challenges, v2_training_challenges, v2_eval_challenges, build_challenges_v2, build_challenges
from src.logic import solve_challenge_with_accuracy
from src.models import GRID
from src import logfire

from pydantic import BaseModel, TypeAdapter

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.run_python import run_python_transform_sync
from src.logic import run_python_transform_sync_cached

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
    parser.add_argument("-p", "--path", type=str, help="Input dataset file path")
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

    if args.version1:
        challenges = eval_challenges
    elif args.path:
        print(f"Building challenges from {args.path}")
        challenges = build_challenges(
            challenges_path=Path(args.path),
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

    from src.trees.experiments import grok_dreamcoder_tree
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
            try:
                # Evaluate each primitive on training examples; if one achieves perfect train accuracy, use it to generate test outputs.
                for primitive in library.primitives:
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
                            print(f"[{challenge.id}] solved via library precheck using primitive {primitive.id}")
                            logfire.debug(f"[{challenge.id}] solved via library precheck using primitive {primitive.id}")
                            break
            except Exception as e:
                logfire.debug(f"[{challenge.id}] library precheck error: {e}")

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
                    tree=grok_dreamcoder_tree,
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
            from src.trees.experiments import grok_dreamcoder_tree as _tree_ref
            for node in _tree_ref:
                node.attempts = attempts_override
            print(f"Using SUBMISSION_ATTEMPTS={attempts_override}")
            logfire.debug(f"Using SUBMISSION_ATTEMPTS={attempts_override}")
        except Exception as _:
            print("WARNING: invalid SUBMISSION_ATTEMPTS; ignoring")
            logfire.debug("WARNING: invalid SUBMISSION_ATTEMPTS; ignoring")

    # Optional: override batch size (affects number of concurrent challenges)
    bs_env = os.environ.get("SUBMISSION_BATCH_SIZE")
    try:
        batch_size = max(1, int(bs_env)) if bs_env else 60
    except Exception:
        batch_size = 60
    print(f"Using SUBMISSION_BATCH_SIZE={batch_size}")
    logfire.debug(f"Using SUBMISSION_BATCH_SIZE={batch_size}")

    for i in range(total_rounds):
        # Track per-round count of cases that are perfect on train but wrong on test
        train_perfect_but_test_wrong: set[str] = set()

        # Sliding window concurrency: keep up to batch_size tasks in-flight
        sem = asyncio.Semaphore(batch_size)
        processed = 0

        async def bounded_try(challenge_id: str):
            async with sem:
                try:
                    res = await try_solve_challenge(challenge_id, solved_challenges, total_cost_in_cents)
                    return (challenge_id, res, None)
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

        tasks = [asyncio.create_task(bounded_try(cid)) for cid in eval_ids_to_test]

        for fut in asyncio.as_completed(tasks):
            challenge_id, solved, err = await fut
            processed += 1
            if err is not None:
                print(f"Error solving challenge {challenge_id}: {err}")
                logfire.debug(f"Error solving challenge {challenge_id}: {err}")
            elif solved:
                solved_challenges.add(challenge_id)
            else:
                print(f"Challenge {challenge_id} not solved")
                logfire.debug(f"Challenge {challenge_id} not solved")

            # Maintain per-round FP count
            try:
                if _is_train_perfect_but_test_wrong(challenge_id):
                    train_perfect_but_test_wrong.add(challenge_id)
            except Exception:
                pass

            # Light progress signal without batch head-of-line blocking
            if processed % max(1, batch_size // 2) == 0 or processed == len(eval_ids_to_test):
                print(f"Round {i+1} progress: {processed}/{len(eval_ids_to_test)} processed, {len(solved_challenges)} solved so far")
                logfire.debug(f"Round {i+1} progress: {processed}/{len(eval_ids_to_test)} processed, {len(solved_challenges)} solved so far")

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


    print(f"FINAL: Solved Challenges: {solved_challenges}")
    print(f"FINAL: Correct Percent: {len(solved_challenges) / len(eval_ids_to_test)}")

    print(f"FINAL: Total cost in cents: {total_cost_in_cents[0]}")

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
