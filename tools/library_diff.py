import argparse
import os
import pickle
import sys
from typing import Iterable

# Ensure repo root is importable for unpickling project classes
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
    code = primitive.python_code_str or ""
    pid = primitive.id or ""
    if mode == "code":
        return code
    if mode == "id":
        return pid
    if mode == "code_and_id":
        return (pid, code)
    raise ValueError(f"Unsupported match mode: {mode}")


def diff_primitives(
    *,
    new_primitives: Iterable[Primitive],
    old_primitives: Iterable[Primitive],
    mode: str,
) -> list[Primitive]:
    old_keys = {build_key(p, mode) for p in old_primitives}
    result: list[Primitive] = []
    seen: set[str | tuple[str, str]] = set()
    for primitive in new_primitives:
        key = build_key(primitive, mode)
        if key in old_keys:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(primitive)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pickle containing only primitives present in the NEW library "
            "but missing from the OLD library."
        )
    )
    parser.add_argument("--new", dest="new_path", required=True, help="Path to the newer library pickle")
    parser.add_argument("--old", dest="old_path", required=True, help="Path to the baseline/older library pickle")
    parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Where to write the diff library pickle (will be overwritten if it exists)",
    )
    parser.add_argument(
        "--match-on",
        dest="match_mode",
        choices=["code", "id", "code_and_id"],
        default="code",
        help="Attribute(s) used to decide whether a primitive already existed in the old library",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def main() -> None:
    args = parse_args()

    try:
        new_lib = load_library(args.new_path)
    except Exception as exc:
        print(f"Failed to load new library {args.new_path}: {exc}")
        return
    try:
        old_lib = load_library(args.old_path)
    except Exception as exc:
        print(f"Failed to load old library {args.old_path}: {exc}")
        return

    diff = diff_primitives(
        new_primitives=new_lib.primitives,
        old_primitives=old_lib.primitives,
        mode=args.match_mode,
    )
    diff_lib = Library(primitives=diff)

    ensure_parent_dir(args.output_path)
    with open(args.output_path, "wb") as fh:
        pickle.dump(diff_lib, fh)

    print(f"New library primitives: {len(new_lib.primitives)}")
    print(f"Old library primitives: {len(old_lib.primitives)}")
    print(f"Diff primitives written: {len(diff)}")
    print(f"Output saved to {args.output_path}")


main()
