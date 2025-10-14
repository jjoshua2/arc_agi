import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Dict, Iterable

# Stub optional dependencies for smooth imports
if "devtools" not in sys.modules:
    sys.modules["devtools"] = types.SimpleNamespace(debug=lambda *_, **__: None)
if "asyncpg" not in sys.modules:
    asyncpg_stub = types.SimpleNamespace(
        pool=types.SimpleNamespace(Pool=object),
        create_pool=lambda *_, **__: None,
    )
    sys.modules["asyncpg"] = asyncpg_stub
if "asyncer" not in sys.modules:
    asyncer_stub = types.SimpleNamespace(asyncify=lambda func: func)
    sys.modules["asyncer"] = asyncer_stub

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import Library  # noqa: E402
from src.run_python import run_python_transform_sync  # noqa: E402
from src.data import (  # noqa: E402
    ChallengeAdapter,
    training_challenges,
    eval_challenges,
)


def load_conceptarc_challenges(root: str) -> dict[str, object]:
    base_path = Path(root).resolve()
    collected: dict[str, dict] = {}
    for file_path in sorted(base_path.rglob("*.json")):
        if not file_path.is_file():
            continue
        try:
            challenge = json.loads(file_path.read_text("utf-8"))
        except Exception:
            continue
        rel_parts = file_path.relative_to(base_path).with_suffix("").parts
        cid = "__".join(rel_parts)
        challenge = dict(challenge)
        challenge["id"] = cid
        collected[cid] = challenge
    if not collected:
        return {}
    return ChallengeAdapter.validate_python(collected)

CONCEPTARC_DIR = os.path.join("test_data", "ConceptARC", "data")
conceptarc_challenges = load_conceptarc_challenges(CONCEPTARC_DIR)


CHALLENGE_MAPS: list[Dict[str, object]] = [
    training_challenges,
    eval_challenges,
    conceptarc_challenges,
]


def load_library(path: str) -> Library:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} lacks a 'primitives' list")
    return Library(primitives=list(primitives))


def load_accuracy_scores(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def get_challenge(cid: str):
    for mapping in CHALLENGE_MAPS:
        challenge = mapping.get(cid)
        if challenge is not None:
            return challenge
    return None


def grids_from_examples(examples):
    return [example.input for example in examples], [example.output for example in examples]


def run_primitive(code: str, grid_lists: list, timeout: int) -> list | None:
    result = run_python_transform_sync(
        code=code,
        grid_lists=grid_lists,
        timeout=timeout,
        raise_exception=False,
    )
    if not result or not getattr(result, "transform_results", None):
        return None
    return result.transform_results


def compare_outputs(preds: list, expected: list) -> bool | str:
    if any(out is None for out in expected):
        return "unknown"
    if preds is None:
        return False
    if len(preds) != len(expected):
        return False
    for p, e in zip(preds, expected):
        if p != e:
            return False
    return True


def filter_primitives(library: Library, ids: Iterable[str]) -> dict[str, str]:
    ids_set = set(ids)
    mapping = {}
    for primitive in library.primitives:
        pid = str(primitive.id)
        if pid in ids_set:
            mapping[pid] = primitive.python_code_str or ""
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether primitives that ace training also solve test cases."
    )
    parser.add_argument("library", help="Path to library pickle (e.g., saved_library_eval_round_78.pkl)")
    parser.add_argument(
        "scores",
        help="Path to challenge_primitive_accuracy_scores.pkl",
    )
    parser.add_argument(
        "--primitive",
        dest="primitive_ids",
        action="append",
        required=True,
        help="Primitive ID to inspect (repeat flag for multiple IDs)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5")),
        help="Transform timeout per execution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_library(args.library)
    scores = load_accuracy_scores(args.scores)
    code_map = filter_primitives(library, args.primitive_ids)
    timeout = args.timeout

    if not code_map:
        print("No matching primitives found in library for requested IDs.")
        return

    for pid, code in code_map.items():
        print("=" * 80)
        print(f"Primitive {pid}")
        solved_any = False
        for cid, score_map in scores.items():
            entry = score_map.get(pid)
            if entry is None:
                continue
            challenge = get_challenge(cid)
            if challenge is None:
                continue
            train_inputs, train_expected = grids_from_examples(challenge.train)
            train_result = run_primitive(code, train_inputs, timeout)
            train_status = compare_outputs(train_result, train_expected)
            if train_status is True:
                solved_any = True
            test_inputs, test_expected = grids_from_examples(challenge.test)
            test_result = run_primitive(code, test_inputs, timeout)
            test_status = compare_outputs(test_result, test_expected)
            print(f"Challenge {cid}: train_status={train_status}, test_status={test_status}")
        if not solved_any:
            print("No challenges matched with perfect training accuracy in score file.")


main()
