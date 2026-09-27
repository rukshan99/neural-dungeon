#!/usr/bin/env python
"""Maintainer check: every floor manifest points at files that exist and agree.

For every floor:
- floor.toml parses and has a README, __init__.py in floor/rooms/solutions/trials
- every room's `file` exists in rooms/ AND solutions/ (same name)
- every room's `trial` exists and its load_room() call names that room module
- every room has hints/<room_id>.md with at least 2 hints
- every loot file exists
- exactly one boss, at most one secret
- the ASCII map is under 80 columns
- floor numbers are contiguous from 0

Usage: python scripts/lint_floors.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dungeon.registry import load_floors  # noqa: E402

LOAD_ROOM = re.compile(r"load_room\(\s*__file__\s*,\s*['\"]([\w]+)['\"]\s*\)")


def main() -> int:
    problems: list[str] = []
    floors = load_floors(ROOT)
    numbers = [f.number for f in floors]
    if numbers != list(range(len(floors))):
        problems.append(f"floor numbers are not contiguous from 0: {numbers}")

    for floor in floors:
        fid = floor.id
        for rel in ("README.md", "__init__.py", "rooms/__init__.py", "solutions/__init__.py", "trials/__init__.py"):
            if not (floor.path / rel).is_file():
                problems.append(f"{fid}: missing {rel}")
        if floor.boss is None:
            problems.append(f"{fid}: no boss")
        for line in floor.map_art.splitlines():
            if len(line) > 80:
                problems.append(f"{fid}: map line wider than 80 columns: {line[:40]}...")
                break
        for room in floor.rooms:
            room_file = floor.path / room.file
            if not room_file.is_file():
                problems.append(f"{fid}/{room.id}: room file missing: {room.file}")
            solution = floor.path / "solutions" / Path(room.file).name
            if not solution.is_file():
                problems.append(f"{fid}/{room.id}: solution missing: solutions/{Path(room.file).name}")
            trial = floor.path / room.trial
            if not trial.is_file():
                problems.append(f"{fid}/{room.id}: trial missing: {room.trial}")
            else:
                loaded = LOAD_ROOM.findall(trial.read_text())
                stem = Path(room.file).stem
                if stem not in loaded:
                    problems.append(f"{fid}/{room.id}: {room.trial} never load_room()s '{stem}' (found {loaded})")
            hints = floor.path / "hints" / f"{room.id}.md"
            if not hints.is_file():
                problems.append(f"{fid}/{room.id}: no hints file hints/{room.id}.md")
            else:
                n = len([h for h in hints.read_text().split("\n---\n") if h.strip()])
                if n < 2:
                    problems.append(f"{fid}/{room.id}: only {n} hint(s)")
        for loot in floor.loot:
            if not (floor.path / loot.file).is_file():
                problems.append(f"{fid}: loot file missing: {loot.file}")
        print(f"  {fid:<9} {floor.name:<34} rooms={len(floor.regular_rooms)} boss={'yes' if floor.boss else 'NO'} "
              f"secret={'yes' if floor.secret else 'no'} loot={len(floor.loot)}")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"{len(floors)} floors, every manifest consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
