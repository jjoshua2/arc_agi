import argparse
import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Iterable, NamedTuple

# Stub optional dependencies for smooth imports
if "devtools" not in sys.modules:
    sys.modules["devtools"] = types.SimpleNamespace(debug=lambda *args, **kwargs: None)
if "asyncpg" not in sys.modules:
    asyncpg_stub = types.SimpleNamespace(
        pool=types.SimpleNamespace(Pool=object),
        create_pool=lambda *args, **kwargs: None,
    )
    sys.modules["asyncpg"] = asyncpg_stub
if "asyncer" not in sys.modules:
    asyncer_stub = types.SimpleNamespace(asyncify=lambda func: func)
    sys.modules["asyncer"] = asyncer_stub

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import Challenge, Library, Primitive  # noqa: E402
from src.run_python import run_python_transform_sync  # noqa: E402
from src.data import ChallengeAdapter, training_challenges  # noqa: E402

DEFAULT_TIMEOUT = int(os.environ.get("SUBMISSION_TRANSFORM_TIMEOUT", "5"))

_TRANSFORM_CACHE_ENABLED = os.environ.get("SUBMISSION_TRANSFORM_CACHE", "1") == "1"
_TRANSFORM_CACHE_PATH = os.environ.get("TRANSFORM_CACHE_PATH", "transforms_cache.pkl")
_TRANSFORM_CACHE_CLEAR = os.environ.get("SUBMISSION_TRANSFORM_CACHE_CLEAR", "0") == "1"
_transform_cache: dict[str, list] = {}


def _hash_obj(obj: Any) -> str:
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
            with open(_TRANSFORM_CACHE_PATH, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                _transform_cache = data
    except Exception:
        _transform_cache = {}


def save_transform_cache() -> None:
    if not _TRANSFORM_CACHE_ENABLED:
        return
    try:
        with open(_TRANSFORM_CACHE_PATH, "wb") as fh:
            pickle.dump(_transform_cache, fh)
    except Exception:
        pass


def run_python_transform_sync_cached(
    *, code: str, grid_lists: list, timeout: int, raise_exception: bool
):
    if not _TRANSFORM_CACHE_ENABLED:
        return run_python_transform_sync(
            code=code, grid_lists=grid_lists, timeout=timeout, raise_exception=raise_exception
        )
    key = _transform_cache_key(code=code, grid_lists=grid_lists)
    if key in _transform_cache:
        return types.SimpleNamespace(transform_results=_transform_cache[key], latency_ms=0.0)
    result = run_python_transform_sync(
        code=code, grid_lists=grid_lists, timeout=timeout, raise_exception=raise_exception
    )
    if result and getattr(result, "transform_results", None):
        _transform_cache[key] = result.transform_results
    return result


class ChallengeIO(NamedTuple):
    challenge: Challenge
    inputs: list[list[list[int]]]
    outputs: list[list[list[int]]]


def load_library(path: str) -> Library:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} lacks a 'primitives' list")
    return Library(primitives=list(primitives))


def load_conceptarc_challenges(root: str) -> dict[str, Challenge]:
    base_path = Path(root).resolve()
    collected: dict[str, dict] = {}
    for file_path in sorted(base_path.rglob("*.json")):
        if not file_path.is_file():
            continue
        try:
            payload = json.loads(file_path.read_text("utf-8"))
        except Exception:
            continue
        rel_parts = file_path.relative_to(base_path).with_suffix("").parts
        cid = "__".join(rel_parts)
        payload = dict(payload)
        payload["id"] = cid
        collected[cid] = payload
    if not collected:
        return {}
    return ChallengeAdapter.validate_python(collected)


def prepare_challenge_ios(challenges: Iterable[Challenge]) -> list[ChallengeIO]:
    bundles: list[ChallengeIO] = []
    for challenge in challenges:
        if not challenge.test:
            continue
        inputs = [example.input for example in challenge.test]
        outputs = [example.output for example in challenge.test]
        if any(output is None for output in outputs):
            continue
        bundles.append(ChallengeIO(challenge=challenge, inputs=inputs, outputs=outputs))
    return bundles


def primitive_solves_test(code: str, bundle: ChallengeIO, timeout: int) -> bool:
    result = run_python_transform_sync_cached(
        code=code,
        grid_lists=bundle.inputs,
        timeout=timeout,
        raise_exception=False,
    )
    if not result or not getattr(result, "transform_results", None):
        return False
    outputs = result.transform_results
    if len(outputs) != len(bundle.outputs):
        return False
    for predicted, expected in zip(outputs, bundle.outputs):
        if predicted != expected:
            return False
    return True


def collect_test_solving_primitives(
    library: Library,
    bundles: list[ChallengeIO],
    *,
    timeout: int,
    progress_interval: int,
) -> tuple[list[Primitive], dict[str, list[str]]]:
    keep: list[Primitive] = []
    details: dict[str, list[str]] = {}
    total = len(library.primitives)
    for idx, primitive in enumerate(library.primitives, start=1):
        code = primitive.python_code_str or ""
        if not code:
            continue
        solved_ids: list[str] = []
        for bundle in bundles:
            if primitive_solves_test(code, bundle, timeout=timeout):
                solved_ids.append(bundle.challenge.id)
        # Skip primitives that do not solve any test grids
        if solved_ids:
            keep.append(primitive)
            details[str(primitive.id)] = solved_ids
        if progress_interval > 0 and idx % progress_interval == 0:
            percent = (idx / total) * 100
            print(f"Progress: evaluated {idx}/{total} primitives ({percent:.1f}%)", flush=True)
    return keep, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract primitives that perfectly solve test grids for ConceptARC or ARC v1 training challenges."
        )
    )
    parser.add_argument("--library", required=True, help="Path to the source library pickle")
    parser.add_argument("--out", required=True, help="Path to write the filtered library pickle")
    parser.add_argument(
        "--conceptarc",
        default=os.path.join("test_data", "ConceptARC", "data"),
        help="Path to ConceptARC data directory (default: test_data/ConceptARC/data)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Seconds to allow each transform execution (default uses SUBMISSION_TRANSFORM_TIMEOUT or 5)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print progress every N primitives (default 25; set to 0 to disable)",
    )
    parser.add_argument(
        "--report",
        help="Optional path to write a JSON report mapping primitive IDs to solved challenge IDs",
    )
    return parser.parse_args()


def write_library(primitives: list[Primitive], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(Library(primitives=primitives), fh)


def main() -> None:
    args = parse_args()

    load_transform_cache()

    library = load_library(args.library)
    conceptarc_map = load_conceptarc_challenges(args.conceptarc)

    combined_challenges = list(training_challenges.values()) + list(conceptarc_map.values())
    bundles = prepare_challenge_ios(combined_challenges)
    print(f"Evaluating {len(library.primitives)} primitives across {len(bundles)} challenges...")

    kept, details = collect_test_solving_primitives(
        library,
        bundles,
        timeout=args.timeout,
        progress_interval=max(0, args.progress_interval),
    )

    print(f"Primitives solving at least one test challenge: {len(kept)}")
    write_library(kept, args.out)
    print(f"Filtered library written to {args.out}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(details, fh, indent=2)
        print(f"Report written to {args.report}")

    save_transform_cache()


main()
