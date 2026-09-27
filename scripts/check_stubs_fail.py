#!/usr/bin/env python
"""Maintainer check: every trial must FAIL against the untouched stubs.

A trial that passes before the learner has written anything is not a trial.
Runs each trial file in a subprocess with recording disabled and reports any
file whose stub run passes (or collects nothing).

Usage:
    python scripts/check_stubs_fail.py            # every floor
    python scripts/check_stubs_fail.py 0 3        # only floors 0 and 3
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dungeon.registry import load_floors, resolve_floor  # noqa: E402


def main(argv: list[str]) -> int:
    floors = [resolve_floor(a, ROOT) for a in argv] if argv else load_floors(ROOT)
    env = dict(os.environ, DUNGEON_NO_RECORD="1", NO_COLOR="1")
    env.pop("DUNGEON_SOLUTIONS", None)
    problems: list[str] = []
    for floor in floors:
        for room in floor.rooms:
            trial = floor.path / room.trial
            if not trial.is_file():
                problems.append(f"{floor.id}/{room.id}: trial file missing ({trial})")
                continue
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(trial.relative_to(ROOT)), "-q", "-x", "-p", "no:cacheprovider"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            verdict = "ok  (fails on stubs, as it should)"
            if proc.returncode == 0:
                verdict = "!!  PASSES ON STUBS"
                problems.append(f"{floor.id}/{room.id}: trial passes against the stub")
            elif proc.returncode == 5:
                verdict = "!!  NO TESTS COLLECTED"
                problems.append(f"{floor.id}/{room.id}: no tests collected")
            elif proc.returncode not in (1,):
                verdict = f"!!  pytest exit code {proc.returncode}"
                problems.append(f"{floor.id}/{room.id}: unexpected exit code {proc.returncode}\n{proc.stdout[-2000:]}")
            print(f"  {floor.id}/{room.trial_stem:<14} {verdict}")
    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
        return 1
    print("Every trial fails on stubs. Good: nothing is free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
