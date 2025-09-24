#!/usr/bin/env python
import os
import sys
import argparse
import pickle
import ast
import keyword
import random
from typing import Dict, List, Tuple, Optional

# Try dotenv like the main executables
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Ensure repository root is on sys.path so `import src...` works even if CWD differs
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local imports from the project
try:
    from src.models import Library, Primitive  # type: ignore
    from src.run_python import run_python_transform_sync  # type: ignore
except Exception as e:
    print("Error importing project modules. Try running from the project root.")
    print(e)
    sys.exit(1)


BUILTINS = set(dir(__builtins__))
PY_KEYWORDS = set(keyword.kwlist)


class NormalizeNames(ast.NodeTransformer):
    """AST transformer that:
    - Removes docstrings (module and function level)
    - Renames variable/argument identifiers to a canonical scheme (v1, v2, ...)
      while leaving Python keywords and builtins untouched.
    - Leaves attribute names alone (renames the base Name node only)
    """

    def __init__(self) -> None:
        super().__init__()
        self.name_map_stack: List[Dict[str, str]] = []
        self.counter_stack: List[int] = []

    def _push_scope(self) -> None:
        self.name_map_stack.append({})
        self.counter_stack.append(1)

    def _pop_scope(self) -> None:
        self.name_map_stack.pop()
        self.counter_stack.pop()

    def _current_map(self) -> Dict[str, str]:
        return self.name_map_stack[-1]

    def _next_var(self) -> str:
        i = self.counter_stack[-1]
        self.counter_stack[-1] = i + 1
        return f"v{i}"

    def _rename_if_needed(self, name: str) -> str:
        if name in PY_KEYWORDS or name in BUILTINS:
            return name
        cmap = self._current_map()
        if name not in cmap:
            cmap[name] = self._next_var()
        return cmap[name]

    def visit_Module(self, node: ast.Module) -> ast.AST:
        # Remove module docstring (first statement if it's a string expr)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        self._push_scope()
        node = self.generic_visit(node)
        self._pop_scope()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Remove function docstring
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        # New scope for function
        self._push_scope()
        # Normalize argument names
        for arg in node.args.args:
            arg.arg = self._rename_if_needed(arg.arg)
        if node.args.vararg:
            node.args.vararg.arg = self._rename_if_needed(node.args.vararg.arg)
        for arg in node.args.kwonlyargs:
            arg.arg = self._rename_if_needed(arg.arg)
        if node.args.kwarg:
            node.args.kwarg.arg = self._rename_if_needed(node.args.kwarg.arg)
        # Visit body where names get normalized
        node = self.generic_visit(node)
        self._pop_scope()
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Rename identifiers that aren't keywords/builtins
        return ast.copy_location(ast.Name(id=self._rename_if_needed(node.id), ctx=node.ctx), node)

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        # Drop standalone string expressions (docstrings elsewhere)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ast.Pass()
        return self.generic_visit(node)


def normalize_python_code(code: str) -> str:
    """Normalize code by removing docstrings/comments and renaming variables.
    If AST parsing fails, fall back to a whitespace-stripped version.
    """
    try:
        tree = ast.parse(code)
        tree = NormalizeNames().visit(tree)
        ast.fix_missing_locations(tree)
        # ast.unparse removes comments and normalizes formatting
        normalized = ast.unparse(tree)
        return normalized.strip()
    except Exception:
        # Fallback: crude normalization
        return "\n".join(line.strip() for line in code.splitlines() if line.strip())


def load_library(path: str) -> Library:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Library file not found: {path}")
    with open(path, "rb") as f:
        lib = pickle.load(f)
    if not isinstance(lib, Library):
        raise TypeError("Pickle did not contain a Library instance")
    return lib


def group_by_normalized_code(primitives: List[Primitive]) -> Dict[str, List[Primitive]]:
    groups: Dict[str, List[Primitive]] = {}
    for p in primitives:
        key = normalize_python_code(p.python_code_str)
        groups.setdefault(key, []).append(p)
    return groups


def generate_probe_grids(n: int, *, min_size: int = 2, max_size: int = 5, vocab: int = 10, seed: int = 0) -> List[List[List[int]]]:
    rnd = random.Random(seed)
    probes: List[List[List[int]]] = []
    for _ in range(n):
        rows = rnd.randint(min_size, max_size)
        cols = rnd.randint(min_size, max_size)
        grid = [[rnd.randint(0, vocab - 1) for _ in range(cols)] for _ in range(rows)]
        probes.append(grid)
    return probes


def behavioral_signature(code: str, probes: List[List[List[int]]], timeout: int = 5) -> Optional[Tuple[str, ...]]:
    """Return a signature tuple of outputs for each probe. If any run fails, return None.
    Uses the project’s run_python_transform_sync to execute code.
    """
    try:
        tr = run_python_transform_sync(code=code, grid_lists=probes, timeout=timeout, raise_exception=False)
        if not tr or not tr.transform_results:
            return None
        outs = tr.transform_results
        if len(outs) != len(probes):
            return None
        # Use a stable string form as a hashable unit
        sig = tuple(str(o) for o in outs)
        return sig
    except Exception:
        return None


def group_by_behavior_full_batch(primitives: List[Primitive], probes: List[List[List[int]]], timeout: int = 5) -> Dict[Tuple[str, ...], List[Primitive]]:
    """Current approach: run each primitive on all probes at once; group by full signature."""
    groups: Dict[Tuple[str, ...], List[Primitive]] = {}
    for p in primitives:
        sig = behavioral_signature(p.python_code_str, probes, timeout=timeout)
        if sig is None:
            continue
        groups.setdefault(sig, []).append(p)
    return {k: v for k, v in groups.items() if len(v) > 1}


def group_by_behavior_incremental(primitives: List[Primitive], probes: List[List[List[int]]], timeout: int = 5, progress: bool = True) -> Dict[Tuple[str, ...], List[Primitive]]:
    """Incremental grouping across probes with early exit for singletons.
    This reduces compute by not running further probes for primitives already in singleton groups.
    """
    # Start with all primitives in one bucket
    buckets: List[List[Primitive]] = [list(primitives)]
    # Keep per-primitive accumulated signature for the groups that remain multi
    partial_sigs: Dict[str, List[str]] = {}

    def refine_on_probe(probe: List[List[int]]) -> None:
        nonlocal buckets
        new_buckets: List[List[Primitive]] = []
        for bucket in buckets:
            if len(bucket) <= 1:
                new_buckets.append(bucket)
                continue
            # Evaluate each primitive in this bucket on the current probe
            out_to_list: Dict[str, List[Primitive]] = {}
            for p in bucket:
                sig = behavioral_signature(p.python_code_str, [probe], timeout=timeout)
                if sig is None:
                    # Treat failures as their own output key to avoid merging incorrectly
                    key = "__ERROR__"
                else:
                    key = sig[0]
                # Accumulate (only needed for the final key)
                prev = partial_sigs.get(p.id, [])
                prev.append(key)
                partial_sigs[p.id] = prev
                out_to_list.setdefault(key, []).append(p)
            # Split bucket by outputs
            new_buckets.extend(out_to_list.values())
        buckets = new_buckets

    for i, probe in enumerate(probes):
        # Stop if all buckets are singletons
        if all(len(b) <= 1 for b in buckets):
            break
        refine_on_probe(probe)
        if progress and (i + 1) % 25 == 0:
            multi = sum(1 for b in buckets if len(b) > 1)
            print(f"Processed {i+1}/{len(probes)} probes; {multi} multi-primitive groups remain")

    # Build final groups for reporting using the accumulated signatures
    grouped: Dict[Tuple[str, ...], List[Primitive]] = {}
    for bucket in buckets:
        if len(bucket) <= 1:
            continue
        # Use the accumulated signatures (fallback to empty tuple if missing)
        key = tuple(partial_sigs.get(bucket[0].id, []))
        grouped.setdefault(key, []).extend(bucket)
    return grouped


def count_arc_train_inputs() -> tuple[int, int]:
    """Return (#tasks, total #train inputs across all tasks)."""
    try:
        from src.data import training_challenges  # type: ignore
    except Exception as e:
        print("Failed to import training_challenges from src.data:", e)
        return 0, 0
    n_tasks = len(training_challenges)
    total_inputs = sum(len(ch.train) for ch in training_challenges.values())
    return n_tasks, total_inputs


def build_arc_train_probes(max_examples: Optional[int], max_per_task: Optional[int]) -> List[List[List[int]]]:
    """Collect input grids from ARC v1 training set.
    max_examples: total cap across tasks (None => no cap)
    max_per_task: how many train inputs to take per task (None => no cap)
    """
    try:
        from src.data import training_challenges  # type: ignore
    except Exception as e:
        print("Failed to import training_challenges from src.data:", e)
        return []
    probes: List[List[List[int]]] = []
    for ch in training_challenges.values():
        # ch.train is a list of examples with .input
        taken = 0
        for ex in ch.train:
            probes.append(ex.input)
            taken += 1
            if max_per_task is not None and taken >= max_per_task:
                break
            if max_examples is not None and len(probes) >= max_examples:
                return probes
        if max_examples is not None and len(probes) >= max_examples:
            break
    return probes


def summarize_groups(groups: Dict, max_display: int = 5) -> str:
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    lines = [f"duplicate groups: {len(groups)}", f"group sizes (top {max_display}): {sizes[:max_display]}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the ARC-AGI library for size and duplicates.")
    parser.add_argument("-p", "--path", type=str, default=os.getenv("SUBMISSION_LIBRARY_PATH", "saved_library_1000.pkl"), help="Path to pickled Library (default: SUBMISSION_LIBRARY_PATH or saved_library_1000.pkl)")
    parser.add_argument("--max-display", type=int, default=5, help="How many group sizes to display in summary")
    parser.add_argument("--behavior-source", choices=["random", "arc_v1"], default="arc_v1", help="Input source for behavioral dedupe: random grids or ARC v1 training inputs")
    parser.add_argument("--probes", type=int, default=8, help="Number of random probe grids for behavioral dedupe if needed")
    parser.add_argument("--min-size", type=int, default=2, help="Min probe grid size (random probes)")
    parser.add_argument("--max-size", type=int, default=5, help="Max probe grid size (random probes)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for probes")
    parser.add_argument("--arc-max-examples", type=int, default=-1, help="Max total ARC v1 train inputs to use (-1 => use all)")
    parser.add_argument("--arc-max-per-task", type=int, default=3, help="Max inputs per ARC v1 training task (0 => use all)")
    parser.add_argument("--incremental", action="store_true", help="Use incremental behavioral dedupe with early exit for singletons (faster)")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout per transform run (seconds)")
    args = parser.parse_args()

    lib = load_library(args.path)
    primitives = lib.primitives
    print(f"Library path: {args.path}")
    print(f"Total primitives: {len(primitives)}")

    # 1) Normalized-code dedupe
    norm_groups = group_by_normalized_code(primitives)
    norm_dupes = {k: v for k, v in norm_groups.items() if len(v) > 1}
    if norm_dupes:
        print("\nNormalized-code duplicates found:")
        print(summarize_groups(norm_dupes, args.max_display))
        # Show a small sample of one group
        k0, v0 = next(iter(norm_dupes.items()))
        print("\nExample duplicate group (showing up to first 5 IDs):")
        print([p.id for p in v0[:5]])
        return

    print("\nNo normalized-code duplicates found. Proceeding to behavioral dedupe...")
    if args.behavior_source == "arc_v1":
        n_tasks, total_inputs = count_arc_train_inputs()
        print(f"ARC v1 training set: {n_tasks} tasks, {total_inputs} total train inputs available")
        max_examples = None if args.arc_max_examples == -1 else max(0, args.arc_max_examples)
        max_per_task = None if args.arc_max_per_task == 0 else max(0, args.arc_max_per_task)
        probes = build_arc_train_probes(max_examples, max_per_task)
        if not probes:
            print("Falling back to random probes because ARC v1 probes could not be built.")
            probes = generate_probe_grids(args.probes, min_size=args.min_size, max_size=args.max_size, seed=args.seed)
    else:
        probes = generate_probe_grids(args.probes, min_size=args.min_size, max_size=args.max_size, seed=args.seed)
    print(f"Using {len(probes)} probe inputs for behavioral dedupe")

    if args.incremental:
        beh_groups = group_by_behavior_incremental(primitives, probes, timeout=args.timeout, progress=True)
    else:
        beh_groups = group_by_behavior_full_batch(primitives, probes, timeout=args.timeout)
    if beh_groups:
        print("\nBehavioral duplicates found:")
        print(summarize_groups(beh_groups, args.max_display))
        k0, v0 = next(iter(beh_groups.items()))
        print("\nExample behavioral duplicate group (showing up to first 5 IDs):")
        print([p.id for p in v0[:5]])
    else:
        print("\nNo behavioral duplicates found on the probe set.")
        print("Library appears reasonably unique. No action needed.")


if __name__ == "__main__":
    main()
