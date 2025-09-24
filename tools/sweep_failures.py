#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

FAILURES_DIR = Path("failures")
MANUAL_DIR = Path("manual_responses")
BUNDLES_DIR = FAILURES_DIR / "bundles"

@dataclass
class FailureItem:
    challenge_id: str
    prompt_hash: str
    prompt_txt: Path
    prompt_json: Optional[Path]
    mtime: float


def _iter_failures() -> List[FailureItem]:
    items: List[FailureItem] = []
    if not FAILURES_DIR.exists():
        return items
    for cid_dir in FAILURES_DIR.iterdir():
        if not cid_dir.is_dir():
            continue
        for txt in cid_dir.glob("*.txt"):
            prompt_hash = txt.stem
            json_path = txt.with_suffix(".json")
            try:
                stat = txt.stat()
                items.append(
                    FailureItem(
                        challenge_id=cid_dir.name,
                        prompt_hash=prompt_hash,
                        prompt_txt=txt,
                        prompt_json=json_path if json_path.exists() else None,
                        mtime=stat.st_mtime,
                    )
                )
            except Exception:
                continue
    # oldest first
    items.sort(key=lambda x: x.mtime)
    return items


def _lock_path(prompt_hash: str) -> Path:
    return MANUAL_DIR / f"{prompt_hash}.lock"


def _manual_path(prompt_hash: str) -> Path:
    return MANUAL_DIR / f"{prompt_hash}.txt"


def _now_ts() -> float:
    return time.time()


def _is_locked(prompt_hash: str) -> tuple[bool, Optional[datetime]]:
    lp = _lock_path(prompt_hash)
    if not lp.exists():
        return False, None
    try:
        data = json.loads(lp.read_text(encoding="utf-8"))
        exp = float(data.get("expires_at", 0))
        if _now_ts() < exp:
            return True, datetime.fromtimestamp(exp)
        else:
            # expired; cleanup
            try:
                lp.unlink(missing_ok=True)
            except Exception:
                pass
            return False, None
    except Exception:
        # unreadable -> treat as unlocked
        return False, None


def _lock(prompt_hash: str, duration_minutes: int, challenge_id: str) -> bool:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    locked, exp = _is_locked(prompt_hash)
    if locked:
        return False
    expires_at = _now_ts() + duration_minutes * 60
    payload = {
        "prompt_hash": prompt_hash,
        "challenge_id": challenge_id,
        "locked_at": _now_ts(),
        "expires_at": expires_at,
    }
    _lock_path(prompt_hash).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _unlock(prompt_hash: str) -> None:
    try:
        _lock_path(prompt_hash).unlink(missing_ok=True)
    except Exception:
        pass


def cmd_list(args: argparse.Namespace) -> int:
    items = _iter_failures()
    pending = 0
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    for it in items:
        manual = _manual_path(it.prompt_hash)
        has_manual = manual.exists()
        locked, exp = _is_locked(it.prompt_hash)
        status = []
        status.append("manual=YES" if has_manual else "manual=NO")
        if locked:
            status.append(f"locked_until={exp}")
        print(f"{it.challenge_id} {it.prompt_hash} :: {' '.join(status)} :: {it.prompt_txt}")
        if not has_manual:
            pending += 1
    print(f"Total failures: {len(items)}; Pending manual: {pending}")
    return 0


def cmd_placeholders(args: argparse.Namespace) -> int:
    count = args.count
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    for it in _iter_failures():
        manual = _manual_path(it.prompt_hash)
        if manual.exists():
            continue
        header = (
            f"prompt_hash: {it.prompt_hash}\n"
            f"challenge_id: {it.challenge_id}\n\n"
            "Paste the assistant response below. The program will pick it up next run.\n"
            f"Original prompt file: {it.prompt_txt}\n\n"
        )
        manual.write_text(header, encoding="utf-8")
        created += 1
        print(f"Created placeholder: {manual}")
        if count and created >= count:
            break
    print(f"Created {created} placeholder file(s)")
    return 0


def _select_pending(n: Optional[int], include_locked: bool) -> List[FailureItem]:
    out: List[FailureItem] = []
    for it in _iter_failures():
        manual = _manual_path(it.prompt_hash)
        if manual.exists():
            continue
        locked, _ = _is_locked(it.prompt_hash)
        if locked and not include_locked:
            continue
        out.append(it)
        if n and len(out) >= n:
            break
    return out


def cmd_lock(args: argparse.Namespace) -> int:
    n = args.count
    dur = args.minutes
    items = _select_pending(n, include_locked=False)
    if not items:
        print("No pending items to lock")
        return 0
    locked_any = False
    for it in items:
        ok = _lock(it.prompt_hash, dur, it.challenge_id)
        if ok:
            locked_any = True
            print(f"Locked {it.prompt_hash} for {dur} minutes")
    if not locked_any:
        print("Nothing locked")
    return 0


def cmd_unlock(args: argparse.Namespace) -> int:
    if args.all:
        for it in _iter_failures():
            _unlock(it.prompt_hash)
        print("Unlocked all")
        return 0
    # unlock expired only
    for it in _iter_failures():
        locked, exp = _is_locked(it.prompt_hash)
        if locked and exp and datetime.now() > exp:
            _unlock(it.prompt_hash)
            print(f"Unlocked expired {it.prompt_hash}")
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    n = args.count
    include_locked = args.include_locked
    items = _select_pending(n, include_locked=include_locked)
    if not items:
        print("No pending items to bundle")
        return 0
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = BUNDLES_DIR / f"bundle_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Bundle {ts}\n\n")
        for it in items:
            text = it.prompt_txt.read_text(encoding="utf-8")
            f.write(f"## challenge_id={it.challenge_id} prompt_hash={it.prompt_hash}\n\n")
            f.write("```text path=null start=null\n")
            f.write(text)
            f.write("\n```\n\n")
    print(f"Wrote bundle: {out_path}")
    if args.lock_minutes:
        for it in items:
            _lock(it.prompt_hash, args.lock_minutes, it.challenge_id)
        print(f"Locked {len(items)} items for {args.lock_minutes} minutes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep and manage ARC AGI failure prompts for manual processing")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List failures and manual response status")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("placeholders", help="Create empty manual response placeholders for pending failures")
    sp.add_argument("--count", type=int, default=0, help="Limit number of placeholders to create (0 = all pending)")
    sp.set_defaults(func=cmd_placeholders)

    sp = sub.add_parser("lock", help="Lock a subset of pending failures for manual processing")
    sp.add_argument("--count", type=int, default=10, help="Number of items to lock (oldest first)")
    sp.add_argument("--minutes", type=int, default=60, help="Lock duration in minutes")
    sp.set_defaults(func=cmd_lock)

    sp = sub.add_parser("unlock", help="Unlock failures (expired by default)")
    sp.add_argument("--all", action="store_true", help="Unlock all locks")
    sp.set_defaults(func=cmd_unlock)

    sp = sub.add_parser("bundle", help="Export a markdown bundle of prompts for quick copy/paste into multiple tabs")
    sp.add_argument("--count", type=int, default=10, help="Number of items to include (oldest first)")
    sp.add_argument("--include-locked", action="store_true", help="Include locked items in bundle selection")
    sp.add_argument("--lock-minutes", type=int, default=0, help="Also lock bundled items for this many minutes (0 = no lock)")
    sp.set_defaults(func=cmd_bundle)

    return p


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))