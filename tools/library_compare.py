import argparse
import os
import pickle
import sys
import types
from collections import Counter, defaultdict
from typing import Iterable

# Provide a noop fallback if devtools is unavailable
try:
    import devtools  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    devtools = types.SimpleNamespace(debug=lambda *_, **__: None)
    sys.modules["devtools"] = devtools

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import Library, Primitive  # noqa: E402


def load_library(path: str) -> Library:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Library file not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} does not contain a Library-like object")
    return Library(primitives=list(primitives))


def normalize_code(code: str | None) -> str:
    return (code or "").strip()


def describe_overlap(lib_a: Library, lib_b: Library) -> dict[str, object]:
    codes_a = {normalize_code(p.python_code_str) for p in lib_a.primitives}
    codes_b = {normalize_code(p.python_code_str) for p in lib_b.primitives}
    ids_a = {p.id or "" for p in lib_a.primitives}
    ids_b = {p.id or "" for p in lib_b.primitives}
    combo_a = {(p.id or "", normalize_code(p.python_code_str)) for p in lib_a.primitives}
    combo_b = {(p.id or "", normalize_code(p.python_code_str)) for p in lib_b.primitives}

    same_code = codes_a & codes_b
    same_id = ids_a & ids_b
    same_combo = combo_a & combo_b

    # classify overlapping primitives by relationship
    code_to_ids_a: dict[str, set[str]] = defaultdict(set)
    code_to_ids_b: dict[str, set[str]] = defaultdict(set)
    id_to_codes_a: dict[str, set[str]] = defaultdict(set)
    id_to_codes_b: dict[str, set[str]] = defaultdict(set)

    for p in lib_a.primitives:
        code = normalize_code(p.python_code_str)
        pid = p.id or ""
        code_to_ids_a[code].add(pid)
        id_to_codes_a[pid].add(code)
    for p in lib_b.primitives:
        code = normalize_code(p.python_code_str)
        pid = p.id or ""
        code_to_ids_b[code].add(pid)
        id_to_codes_b[pid].add(code)

    code_overlap_with_diff_ids = {
        code
        for code in same_code
        if code_to_ids_a[code] != code_to_ids_b[code]
    }
    id_overlap_with_diff_code = {
        pid
        for pid in same_id
        if id_to_codes_a[pid] != id_to_codes_b[pid]
    }

    # count duplicates mapping (code -> count across libs)
    overlaps_by_code = []
    for code in same_code:
        count_a = len(code_to_ids_a[code])
        count_b = len(code_to_ids_b[code])
        overlaps_by_code.append((code, count_a, count_b))

    overlaps_by_id = []
    for pid in same_id:
        count_a = len(id_to_codes_a[pid])
        count_b = len(id_to_codes_b[pid])
        overlaps_by_id.append((pid, count_a, count_b))

    return {
        "count_a": len(lib_a.primitives),
        "count_b": len(lib_b.primitives),
        "unique_codes_a": len(codes_a),
        "unique_codes_b": len(codes_b),
        "shared_code_count": len(same_code),
        "shared_id_count": len(same_id),
        "shared_combo_count": len(same_combo),
        "code_overlap_with_diff_ids": len(code_overlap_with_diff_ids),
        "id_overlap_with_diff_code": len(id_overlap_with_diff_code),
        "overlaps_by_code": overlaps_by_code,
        "overlaps_by_id": overlaps_by_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two library pickles and report overlap statistics."
    )
    parser.add_argument("library_a", help="Path to the first library pickle")
    parser.add_argument("library_b", help="Path to the second library pickle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lib_a = load_library(args.library_a)
    lib_b = load_library(args.library_b)

    stats = describe_overlap(lib_a, lib_b)

    print(f"Library A primitives: {stats['count_a']}")
    print(f"Library B primitives: {stats['count_b']}")
    print(f"Unique codes A: {stats['unique_codes_a']}")
    print(f"Unique codes B: {stats['unique_codes_b']}")
    print(f"Shared codes: {stats['shared_code_count']}")
    print(f"Shared ids: {stats['shared_id_count']}")
    print(f"Shared (id, code) pairs: {stats['shared_combo_count']}")
    print(f"Shared codes with differing ids: {stats['code_overlap_with_diff_ids']}")
    print(f"Shared ids with differing code: {stats['id_overlap_with_diff_code']}")

    if stats["code_overlap_with_diff_ids"]:
        print("\nExamples of shared code w/ different ids (up to 5):")
        for code, count_a, count_b in stats["overlaps_by_code"][:5]:
            print(f"  code snippet: {code[:60]!r}... | counts A/B: {count_a}/{count_b}")

    if stats["id_overlap_with_diff_code"]:
        print("\nExamples of shared ids w/ different code (up to 5):")
        for pid, count_a, count_b in stats["overlaps_by_id"][:5]:
            print(f"  id: {pid!r} | codes A/B: {count_a}/{count_b}")


main()
