import os
import pickle
import sys
import re
from typing import Any

# Ensure repo root is importable for unpickling project classes
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_library(path: str) -> Any:
    if not os.path.exists(path):
        print(f"Library file not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    lib_path = os.environ.get("SUBMISSION_LIBRARY_PATH", "saved_library_1000.pkl")
    try:
        lib = load_library(lib_path)
    except Exception as e:
        print(f"Failed to load library {lib_path}: {e}")
        sys.exit(1)

    primitives = getattr(lib, "primitives", None)
    if not isinstance(primitives, list):
        print("Library does not have a 'primitives' list")
        sys.exit(1)

    total = len(primitives)
    has_ri = 0
    ids_with_ri = []
    samples = []

    for p in primitives:
        code = getattr(p, "python_code_str", None)
        pid = getattr(p, "id", "?")
        if isinstance(code, str) and ("r_i" in code):
            has_ri += 1
            if len(ids_with_ri) < 50:
                ids_with_ri.append(pid)
            if len(samples) < 5:
                # Grab a small window around the first occurrence for context
                idx = code.find("r_i")
                start = max(0, idx - 80)
                end = min(len(code), idx + 80)
                samples.append((pid, code[start:end].replace("\n", "\\n")))

    print(f"Total primitives: {total}")
    print(f"Primitives containing 'r_i': {has_ri}")
    if ids_with_ri:
        print("Example primitive IDs with 'r_i':", ids_with_ri[:10])
    if samples:
        print("\nCode samples containing 'r_i':")
        for pid, snippet in samples:
            print(f"[{pid}] ...{snippet}...")


if __name__ == "__main__":
    main()
