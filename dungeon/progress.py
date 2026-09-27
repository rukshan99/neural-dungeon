"""Remember what you have cleared.

Progress is a single JSON file at ``.dungeon/progress.json`` in your clone. It is
git-ignored: it is *your* save file. Delete it (or run ``dungeon reset``) to start
a fresh run.

Shape of the file::

    {
      "version": 1,
      "trials": {
        "floor_01/test_room_1": {
          "cleared": true,
          "passed": 6, "failed": 0, "skipped": 0, "errors": 0,
          "collected": 6, "deselected": 0,
          "attempts": 3,
          "first_cleared": "2026-09-27T18:00:00+00:00",
          "last_run": "2026-09-27T18:00:00+00:00"
        }
      },
      "hints": {"floor_01/room_1": 2}
    }

A trial is *cleared* only when every test in its file was collected (nothing
deselected with ``-k``) and every one of them passed. Running a single test is a
fine way to iterate, but it will not clear the room.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dungeon.registry import PROGRESS_DIRNAME, Floor, Room, find_root

PROGRESS_FILE = "progress.json"
VERSION = 1


@dataclass
class TrialResult:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    collected: int = 0
    deselected: int = 0
    unwritten: int = 0  # failures that were plain NotImplementedError

    @property
    def ran(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def cleared(self) -> bool:
        return (
            self.collected > 0
            and self.deselected == 0
            and self.failed == 0
            and self.errors == 0
            and self.passed + self.skipped == self.collected
            and self.passed > 0
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def progress_path(root: Path | None = None) -> Path:
    return (root or find_root()) / PROGRESS_DIRNAME / PROGRESS_FILE


def load(root: Path | None = None) -> dict:
    path = progress_path(root)
    if not path.is_file():
        return {"version": VERSION, "trials": {}, "hints": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        # A corrupt save file should never stop you from playing.
        backup = path.with_suffix(".corrupt.json")
        path.replace(backup)
        return {"version": VERSION, "trials": {}, "hints": {}}
    data.setdefault("trials", {})
    data.setdefault("hints", {})
    return data


def save(data: dict, root: Path | None = None) -> None:
    path = progress_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def recording_disabled() -> bool:
    """Runs against the reference solutions must never clear a floor for you."""
    return os.environ.get("DUNGEON_SOLUTIONS", "") == "1" or os.environ.get("DUNGEON_NO_RECORD") == "1"


def record_trial(key: str, result: TrialResult, root: Path | None = None) -> dict:
    """Merge one trial-file result into the save file. Returns the stored entry."""
    data = load(root)
    entry = data["trials"].get(key, {})
    now = _now()
    was_cleared = bool(entry.get("cleared"))
    entry.update(asdict(result))
    entry["cleared"] = result.cleared or was_cleared
    entry["cleared_this_run"] = result.cleared
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["last_run"] = now
    if result.cleared and not entry.get("first_cleared"):
        entry["first_cleared"] = now
    data["trials"][key] = entry
    save(data, root)
    return entry


def is_cleared(data: dict, key: str) -> bool:
    return bool(data["trials"].get(key, {}).get("cleared"))


def trial_entry(data: dict, key: str) -> dict | None:
    return data["trials"].get(key)


def hint_level(data: dict, key: str) -> int:
    return int(data["hints"].get(key, 0))


def set_hint_level(key: str, level: int, root: Path | None = None) -> None:
    data = load(root)
    data["hints"][key] = level
    save(data, root)


def reset(root: Path | None = None, floor: Floor | None = None) -> int:
    """Forget progress. With a floor, only that floor's entries. Returns entries removed."""
    data = load(root)
    if floor is None:
        removed = len(data["trials"]) + len(data["hints"])
        data = {"version": VERSION, "trials": {}, "hints": {}}
    else:
        prefix = floor.id + "/"
        before = len(data["trials"]) + len(data["hints"])
        data["trials"] = {k: v for k, v in data["trials"].items() if not k.startswith(prefix)}
        data["hints"] = {k: v for k, v in data["hints"].items() if not k.startswith(prefix)}
        removed = before - len(data["trials"]) - len(data["hints"])
    save(data, root)
    return removed


# ------------------------------------------------------------- floor summaries

@dataclass
class FloorStatus:
    floor: Floor
    cleared_rooms: list[Room]
    open_rooms: list[Room]
    boss_defeated: bool
    secret_found: bool
    started: bool

    @property
    def cleared(self) -> bool:
        return not self.open_rooms and self.boss_defeated

    @property
    def required_total(self) -> int:
        return len(self.floor.required_rooms)

    @property
    def required_done(self) -> int:
        return len(self.cleared_rooms) + (1 if self.boss_defeated else 0)


def floor_status(floor: Floor, data: dict | None = None, root: Path | None = None) -> FloorStatus:
    data = data if data is not None else load(root)
    cleared, open_ = [], []
    for room in floor.regular_rooms:
        (cleared if is_cleared(data, floor.trial_key(room)) else open_).append(room)
    boss = floor.boss
    boss_defeated = bool(boss and is_cleared(data, floor.trial_key(boss)))
    secret = floor.secret
    secret_found = bool(secret and is_cleared(data, floor.trial_key(secret)))
    started = any(k.startswith(floor.id + "/") for k in data["trials"])
    return FloorStatus(
        floor=floor,
        cleared_rooms=cleared,
        open_rooms=open_,
        boss_defeated=boss_defeated,
        secret_found=secret_found,
        started=started,
    )


# ------------------------------------------------------------ whole-dungeon

RANKS = [
    (0.00, "Wanderer at the Threshold"),
    (0.08, "Apprentice of the Descent"),
    (0.20, "Keeper of Gradients"),
    (0.32, "Layer-Smith"),
    (0.45, "Token Scribe"),
    (0.55, "Warden of a Thousand Heads"),
    (0.65, "Tower Climber"),
    (0.75, "Librarian of Echoes"),
    (0.85, "Prompt Weaver"),
    (0.95, "Engineer of the Deep"),
    (1.00, "Dungeon Master"),
]


def rank(fraction: float) -> str:
    title = RANKS[0][1]
    for threshold, name in RANKS:
        if fraction >= threshold:
            title = name
    return title


@dataclass
class Summary:
    floors_total: int
    floors_cleared: int
    rooms_total: int
    rooms_done: int
    bosses_total: int
    bosses_done: int
    secrets_total: int
    secrets_done: int
    attempts: int
    hints_used: int

    @property
    def fraction(self) -> float:
        required = self.rooms_total + self.bosses_total
        return (self.rooms_done + self.bosses_done) / required if required else 0.0

    @property
    def rank(self) -> str:
        return rank(self.fraction)


def summarize(floors: list[Floor], data: dict) -> Summary:
    statuses = [floor_status(f, data) for f in floors]
    return Summary(
        floors_total=len(floors),
        floors_cleared=sum(s.cleared for s in statuses),
        rooms_total=sum(len(f.regular_rooms) for f in floors),
        rooms_done=sum(len(s.cleared_rooms) for s in statuses),
        bosses_total=sum(1 for f in floors if f.boss),
        bosses_done=sum(s.boss_defeated for s in statuses),
        secrets_total=sum(1 for f in floors if f.secret),
        secrets_done=sum(s.secret_found for s in statuses),
        attempts=sum(int(e.get("attempts", 0)) for e in data["trials"].values()),
        hints_used=sum(int(v) for v in data["hints"].values()),
    )
