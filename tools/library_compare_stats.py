import argparse
import os
import pickle
import sys
import types
from collections import defaultdict
from typing import Iterable

# Stub optional dependencies before importing project modules
if "devtools" not in sys.modules:
    sys.modules["devtools"] = types.SimpleNamespace(debug=lambda *_, **__: None)

if "asyncpg" not in sys.modules:
    asyncpg_stub = types.SimpleNamespace(
        pool=types.SimpleNamespace(Pool=object),
        create_pool=lambda *_, **__: None,
    )
    sys.modules["asyncpg"] = asyncpg_stub

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import Library, Primitive  # noqa: E402


def load_library(path: str) -> Library:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} does not contain a Library-like object")
    return Library(primitives=list(primitives))


def normalize_code(code: str | None) -> str:
    return (code or "").strip()


def stats_for_libraries(libs: Iterable[Library]) -> dict:
    first, second = libs

    codes_a = {normalize_code(p.python_code_str) for p in first.primitives}
    codes_b = {normalize_code(p.python_code_str) for p in second.primitives}
    ids_a = {p.id or "" for p in first.primitives}
    ids_b = {p.id or "" for p in second.primitives}
    combos_a = {(p.id or "", normalize_code(p.python_code_str)) for p in first.primitives}
    combos_b = {(p.id or "", normalize_code(p.python_code_str)) for p in second.primitives}

    shared_codes = codes_a & codes_b
    shared_ids = ids_a & ids_b
    shared_combos = combos_a & combos_b

    code_to_ids_a: dict[str, set[str]] = defaultdict(set)
    code_to_ids_b: dict[str, set[str]] = defaultdict(set)
    id_to_codes_a: dict[str, set[str]] = defaultdict(set)
    id_to_codes_b: dict[str, set[str]] = defaultdict(set)

    for p in first.primitives:
        code = normalize_code(p.python_code_str)
        pid = p.id or ""
        code_to_ids_a[code].add(pid)
        id_to_codes_a[pid].add(code)
    for p in second.primitives:
        code = normalize_code(p.python_code_str)
        pid = p.id or ""
        code_to_ids_b[code].add(pid)
        id_to_codes_b[pid].add(code)

    shared_codes_diff_ids = [
        code
        for code in shared_codes
        if code_to_ids_a[code] != code_to_ids_b[code]
    ]
    shared_ids_diff_code = [
        pid
        for pid in shared_ids
        if id_to_codes_a[pid] != id_to_codes_b[pid]
    ]

    return {
        "count_a": len(first.primitives),
        "count_b": len(second.primitives),
        "shared_code": len(shared_codes),
        "shared_id": len(shared_ids),
        "shared_combo": len(shared_combos),
        "shared_code_diff_ids": len(shared_codes_diff_ids),
        "shared_id_diff_code": len(shared_ids_diff_code),
        "examples_code_diff_ids": shared_codes_diff_ids[:5],
        "examples_id_diff_code": shared_ids_diff_code[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare overlap between two library pickles")
    parser.add_argument("library_a", help="Path to first library pickle")
    parser.add_argument("library_b", help="Path to second library pickle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lib_a = load_library(args.library_a)
    lib_b = load_library(args.library_b)

    stats = stats_for_libraries((lib_a, lib_b))

    print(f"Library A primitives: {stats['count_a']}")
    print(f"Library B primitives: {stats['count_b']}")
    print(f"Shared code count: {stats['shared_code']}")
    print(f"Shared id count: {stats['shared_id']}")
    print(f"Shared (id, code) pairs: {stats['shared_combo']}")
    print(f"Shared code with different id sets: {stats['shared_code_diff_ids']}")
    print(f"Shared ids with different code sets: {stats['shared_id_diff_code']}")

    if stats['examples_code_diff_ids']:
        print("Examples (code snippet -> differing ids):")
        for code in stats['examples_code_diff_ids']:
            snippet = code.replace("\n", " ")[:80]
            print(f"  {snippet!r}")

    if stats['examples_id_diff_code']:
        print("Examples (id with differing code sets):")
        for pid in stats['examples_id_diff_code']:
            print(f"  {pid!r}")


main()
