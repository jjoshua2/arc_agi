import argparse
import os
import pickle
import sys
import types

# Stub optional dependencies for smooth imports
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
        raise TypeError(f"Pickle at {path} lacks a 'primitives' list")
    return Library(primitives=list(primitives))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show primitive code by ID")
    parser.add_argument("library", help="Path to library pickle")
    parser.add_argument("primitive_id", help="Primitive ID to show")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_library(args.library)
    for primitive in library.primitives:
        if str(primitive.id) == args.primitive_id:
            print(f"Primitive ID: {primitive.id}")
            print("Code:\n" + (primitive.python_code_str or "<empty>"))
            return
    print(f"Primitive {args.primitive_id} not found.")


main()
