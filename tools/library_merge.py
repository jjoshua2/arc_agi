import argparse
import os
import pickle
import sys
import types
from typing import Iterable

# Provide fallback stubs for optional dependencies used by src modules
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Library file not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} does not contain a Library-like object")
    return Library(primitives=list(primitives))


def build_key(primitive: Primitive, mode: str) -> str | tuple[str, str]:
    code = (primitive.python_code_str or "").strip()
    pid = primitive.id or ""
    if mode == "code":
        return code
    if mode == "id":
        return pid
    if mode == "code_and_id":
        return (pid, code)
    raise ValueError(f"Unsupported match mode: {mode}")


def merge_primitives(
    libraries: Iterable[Library],
    *,
    match_mode: str,
) -> list[Primitive]:
    merged: list[Primitive] = []
    seen: set[str | tuple[str, str]] = set()
    for library in libraries:
        for primitive in library.primitives:
            key = build_key(primitive, match_mode)
            if key in seen:
                continue
            seen.add(key)
            merged.append(primitive)
    return merged


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two or more library pickles into one, deduplicating primitives."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to library pickle files (order determines precedence)",
    )
    parser.add_argument("--out", required=True, help="Output path for merged library pickle")
    parser.add_argument(
        "--match-on",
        dest="match_mode",
        choices=["code", "id", "code_and_id"],
        default="code",
        help="Criteria used to treat primitives as duplicates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    libraries: list[Library] = []
    for path in args.inputs:
        try:
            library = load_library(path)
        except Exception as exc:
            print(f"Failed to load library {path}: {exc}")
            return
        libraries.append(library)

    merged_primitives = merge_primitives(libraries, match_mode=args.match_mode)

    merged_library = Library(primitives=merged_primitives)
    ensure_parent_dir(args.out)
    with open(args.out, "wb") as fh:
        pickle.dump(merged_library, fh)

    print(f"Libraries merged: {len(libraries)}")
    print(
        "Total input primitives:",
        sum(len(lib.primitives) for lib in libraries),
    )
    print(f"Merged primitives (deduped): {len(merged_primitives)}")
    print(f"Output saved to {args.out}")


main()
