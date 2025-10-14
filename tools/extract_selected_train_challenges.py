from __future__ import annotations

import json
from pathlib import Path

SELECTED_IDS = [
    "264363fd",
    "50846271",
    "57aa92db",
    "5c2c9af4",
    "a64e4611",
    "a8d7556c",
    "e5062a87",
    "e73095fd",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    source_path = repo_root / "test_data" / "train_challenges_subset.json"
    target_path = repo_root / "test_data" / "unsolved.json"

    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    subset: dict[str, object] = {}
    missing: list[str] = []

    for challenge_id in SELECTED_IDS:
        entry = data.get(challenge_id)
        if entry is None:
            missing.append(challenge_id)
        else:
            subset[challenge_id] = entry

    with target_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
        f.write("\n")

    if missing:
        print("Missing IDs:", ", ".join(missing))
    else:
        print("All IDs exported successfully.")
    print(f"Wrote {len(subset)} challenges to {target_path.relative_to(repo_root)}")
main()
