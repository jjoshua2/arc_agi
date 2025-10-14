import argparse
import os
import pickle
import sys
import types

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

from src.models import Library  # noqa: E402


def load_library(path: str) -> Library:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} lacks a 'primitives' list")
    return Library(primitives=list(primitives))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count primitives containing 'return grid.tolist()'")
    parser.add_argument("library", help="Path to library pickle")
    parser.add_argument(
        "--substring",
        default="return grid.tolist()",
        help="Substring to search for (default: 'return grid.tolist()')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_library(args.library)
    substring = args.substring

    count = 0
    matching_ids: list[str] = []
    for primitive in library.primitives:
        code = primitive.python_code_str or ""
        if substring in code:
            count += 1
            matching_ids.append(str(primitive.id))

    print(f"Total primitives: {len(library.primitives)}")
    print(f"Primitives containing '{substring}': {count}")
    if matching_ids:
        print("IDs:", ", ".join(matching_ids))


main()
