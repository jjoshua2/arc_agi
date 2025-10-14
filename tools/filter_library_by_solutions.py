import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import Library, Primitive  # noqa: E402
from src.run_python import run_python_transform_sync  # noqa: E402


def load_library(path: str) -> Library:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Library file not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} does not contain a Library-like object")
    return Library(primitives=list(primitives))


def collect_challenge_dicts(inputs: Iterable[str]) -> list[dict]:
    collected: list[dict] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for file_path in sorted(path.rglob("*.json")):
                collected.extend(parse_challenge_file(file_path))
        else:
            collected.extend(parse_challenge_file(path))
    return collected


def parse_challenge_file(path: Path) -> list[dict]:
    if not path.exists() or not path.is_file():
        print(f"Skipping missing challenge file: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return []

    challenges: list[dict] = []
    if isinstance(payload, dict) and "train" in payload:
        challenge_id = payload.get("id") or path.stem
        payload = payload.copy()
        payload["id"] = challenge_id
        challenges.append(payload)
    elif isinstance(payload, dict):
        for cid, challenge in payload.items():
            if not isinstance(challenge, dict) or "train" not in challenge:
                continue
            entry = challenge.copy()
            entry["id"] = challenge.get("id") or cid
            challenges.append(entry)
    else:
        print(f"Unrecognized challenge structure in {path}; skipping")
    return challenges


def grids_from_examples(examples: list[dict], key: str) -> list:
    grids: list = []
    for example in examples:
        if key not in example:
            return []
        grids.append(example[key])
    return grids


def primitive_solves_challenge(primitive: Primitive, challenge: dict, *, timeout: int) -> bool:
    train_examples = challenge.get("train", [])
    if not train_examples:
        return False

    expected_train = grids_from_examples(train_examples, "output")
    if not expected_train:
        return False
    train_inputs = grids_from_examples(train_examples, "input")
    if not train_inputs:
        return False

    transform = run_python_transform_sync(
        code=primitive.python_code_str,
        grid_lists=train_inputs,
        timeout=timeout,
        raise_exception=False,
    )
    if not transform or not getattr(transform, "transform_results", None):
        return False
    results = transform.transform_results
    if len(results) != len(expected_train):
        return False
    if any(result != expected for result, expected in zip(results, expected_train)):
        return False

    test_examples = challenge.get("test", [])
    expected_test = grids_from_examples(test_examples, "output") if test_examples else []
    if expected_test and all(example is not None for example in expected_test):
        test_inputs = grids_from_examples(test_examples, "input")
        if not test_inputs:
            return False
        transform_test = run_python_transform_sync(
            code=primitive.python_code_str,
            grid_lists=test_inputs,
            timeout=timeout,
            raise_exception=False,
        )
        if not transform_test or not getattr(transform_test, "transform_results", None):
            return False
        test_results = transform_test.transform_results
        if len(test_results) != len(expected_test):
            return False
        if any(result != expected for result, expected in zip(test_results, expected_test)):
            return False

    return True


def diff_primitives_by_solving(
    *,
    primitives: Iterable[Primitive],
    challenges: list[dict],
    timeout: int,
) -> list[Primitive]:
    keep: list[Primitive] = []
    for primitive in primitives:
        solved_any = False
        for challenge in challenges:
            try:
                if primitive_solves_challenge(primitive, challenge, timeout=timeout):
                    solved_any = True
                    break
            except Exception as exc:
                print(f"Primitive {getattr(primitive, 'id', '?')} raised {exc}; skipping this challenge")
        if solved_any:
            keep.append(primitive)
    return keep


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a library to primitives that perfectly solve at least one provided challenge."
        )
    )
    parser.add_argument("--library", required=True, help="Path to input library pickle")
    parser.add_argument(
        "--challenges",
        nargs="+",
        required=True,
        help="One or more JSON files or directories containing ARC-style challenges",
    )
    parser.add_argument("--out", required=True, help="Path to write the filtered library pickle")
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
        help="Seconds to allow each primitive transform execution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        library = load_library(args.library)
    except Exception as exc:
        print(f"Failed to load library {args.library}: {exc}")
        return

    challenge_dicts = collect_challenge_dicts(args.challenges)
    if not challenge_dicts:
        print("No valid challenges found; aborting")
        return

    kept_primitives = diff_primitives_by_solving(
        primitives=library.primitives,
        challenges=challenge_dicts,
        timeout=args.timeout,
    )

    diff_library = Library(primitives=kept_primitives)
    ensure_parent_dir(args.out)
    with open(args.out, "wb") as fh:
        pickle.dump(diff_library, fh)

    print(f"Input primitives: {len(library.primitives)}")
    print(f"Challenges evaluated: {len(challenge_dicts)}")
    print(f"Primitives retained: {len(kept_primitives)}")
    print(f"Filtered library saved to {args.out}")


main()
