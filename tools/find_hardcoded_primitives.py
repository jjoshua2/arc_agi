import argparse
import os
import pickle
import re
import sys
import types
from dataclasses import dataclass
from typing import Iterable

# Stub optional dependencies so src modules import cleanly
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

MATRIX_PATTERN = re.compile(r"\[\s*(?:\[\s*[0-9,\s]+\]\s*,?){3,}\s*\]")


@dataclass
class PrimitiveReport:
    primitive: Primitive
    digits: int
    matrices: int
    uses_input: bool
    score: int


def load_library(path: str) -> Library:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    primitives = getattr(data, "primitives", None)
    if not isinstance(primitives, list):
        raise TypeError(f"Pickle at {path} lacks a 'primitives' list")
    return Library(primitives=list(primitives))


def analyze_primitive(primitive: Primitive) -> PrimitiveReport:
    code = primitive.python_code_str or ""
    digits = sum(ch.isdigit() for ch in code)
    matrices = len(MATRIX_PATTERN.findall(code))
    uses_input = any(token in code for token in ("input", "grid", "cells"))
    score = digits + matrices * 250
    return PrimitiveReport(
        primitive=primitive,
        digits=digits,
        matrices=matrices,
        uses_input=uses_input,
        score=score,
    )


def find_suspicious(primitives: Iterable[Primitive], *, limit: int) -> list[PrimitiveReport]:
    reports = [analyze_primitive(p) for p in primitives]
    # Filter for high digit count or multiple literal matrices
    suspects = [
        r
        for r in reports
        if r.digits >= 120 or r.matrices >= 1
    ]
    suspects.sort(key=lambda r: r.score, reverse=True)
    return suspects[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag primitives that likely hard-code training grids."
    )
    parser.add_argument("library", help="Path to library pickle")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of top suspicious primitives to display",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_library(args.library)
    suspects = find_suspicious(library.primitives, limit=args.limit)

    print(f"Analyzed primitives: {len(library.primitives)}")
    if not suspects:
        print("No suspicious primitives detected with current heuristics.")
        return

    for idx, report in enumerate(suspects, start=1):
        primitive = report.primitive
        code = (primitive.python_code_str or "").strip()
        snippet = code.replace("\n", " ")[:160]
        print("-" * 80)
        print(f"Rank #{idx}")
        print(f"ID: {primitive.id}")
        print(f"Digits: {report.digits}")
        print(f"Matrix literals detected: {report.matrices}")
        print(f"Uses input tokens: {report.uses_input}")
        print(f"Score: {report.score}")
        print(f"Snippet: {snippet}")


main()
