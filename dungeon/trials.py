"""Trials: the pytest plugin that turns test runs into dungeon progress.

Two jobs live here:

1. ``load_room(__file__, "room_1_altar_of_loss")`` - used by every trial file to
   import the learner's code from ``rooms/``. When ``DUNGEON_SOLUTIONS=1`` it
   imports the reference implementation from ``solutions/`` instead, which is how
   the repository's own CI checks that the trials are correct. Solution runs
   never write to your save file.

2. pytest hooks that tally results per trial file, write them to
   ``.dungeon/progress.json`` and narrate the outcome after the usual summary.

The plugin is registered by the root ``conftest.py``.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from dungeon import progress as prog
from dungeon import ui
from dungeon.registry import FLOORS_DIRNAME, Floor, Room, find_root, floor_for_path

# ---------------------------------------------------------------------------
# Loading rooms
# ---------------------------------------------------------------------------


def solutions_mode() -> bool:
    return os.environ.get("DUNGEON_SOLUTIONS", "") == "1"


def floor_dir_of(test_file: str | Path) -> Path:
    """``.../floors/<floor>/trials/test_x.py`` -> ``.../floors/<floor>``."""
    path = Path(test_file).resolve()
    for parent in path.parents:
        if parent.parent.name == FLOORS_DIRNAME:
            return parent
    raise RuntimeError(f"{test_file} does not live inside a floor directory")


def load_room(test_file: str | Path, module_name: str):
    """Import a room module for the trial at ``test_file``.

    ``module_name`` is the file name without ``.py``, e.g. ``"room_1_altar_of_loss"``.
    Set ``DUNGEON_SOLUTIONS=1`` to load ``solutions/<module_name>.py`` instead.
    """
    floor_dir = floor_dir_of(test_file)
    root = floor_dir.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sub = "solutions" if solutions_mode() else "rooms"
    if not (floor_dir / sub / f"{module_name}.py").is_file():
        raise FileNotFoundError(
            f"No {sub}/{module_name}.py in {floor_dir.name}. "
            + ("The reference solution is missing." if solutions_mode() else "Did the room file get renamed?")
        )
    qualname = f"{FLOORS_DIRNAME}.{floor_dir.name}.{sub}.{module_name}"
    module = importlib.import_module(qualname)
    return module


def room_path(test_file: str | Path, module_name: str) -> Path:
    """Where the learner's file for a room lives (always ``rooms/``)."""
    return floor_dir_of(test_file) / "rooms" / f"{module_name}.py"


def unwritten(what: str = "this room") -> NotImplementedError:
    """The exception a stub raises. Trials show these as 'unwritten' rather than 'failed'."""
    return NotImplementedError(f"{what} is still unwritten - open the room file and fill in the TODOs")


# ---------------------------------------------------------------------------
# pytest plugin
# ---------------------------------------------------------------------------


class _State:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.results: dict[str, prog.TrialResult] = {}
        self.order: list[str] = []
        self.floors: dict[str, Floor] = {}
        self.rooms: dict[str, Room] = {}
        self.skip_reasons: dict[str, str] = {}
        self._key_cache: dict[str, str | None] = {}

    def key_for(self, nodeid: str, rootpath: Path) -> str | None:
        file_part = nodeid.split("::", 1)[0]
        if file_part in self._key_cache:
            return self._key_cache[file_part]
        path = (rootpath / file_part).resolve()
        floor = floor_for_path(path, self.root) if self.root else None
        key = None
        if floor is not None:
            stem = path.stem
            for room in floor.rooms:
                if room.trial_stem == stem:
                    key = floor.trial_key(room)
                    self.floors[key] = floor
                    self.rooms[key] = room
                    break
        self._key_cache[file_part] = key
        return key

    def result(self, key: str) -> prog.TrialResult:
        if key not in self.results:
            self.results[key] = prog.TrialResult()
            self.order.append(key)
        return self.results[key]


def pytest_configure(config: pytest.Config) -> None:
    state = _State()
    try:
        state.root = find_root(Path(str(config.rootpath)))
    except Exception:
        state.root = None
    config._dungeon_state = state  # type: ignore[attr-defined]


def _state(config: pytest.Config) -> _State | None:
    return getattr(config, "_dungeon_state", None)


def pytest_collection_finish(session: pytest.Session) -> None:
    state = _state(session.config)
    if state is None or state.root is None:
        return
    for item in session.items:
        key = state.key_for(item.nodeid, session.config.rootpath)
        if key:
            state.result(key).collected += 1


def pytest_deselected(items) -> None:
    if not items:
        return
    config = items[0].config
    state = _state(config)
    if state is None or state.root is None:
        return
    for item in items:
        key = state.key_for(item.nodeid, config.rootpath)
        if key:
            state.result(key).deselected += 1


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    exc = call.excinfo.value if call.excinfo is not None else None
    if isinstance(exc, NotImplementedError) and report.when in ("setup", "call"):
        report.dungeon_unwritten = True  # type: ignore[attr-defined]
        message = str(exc) or "this room is still unwritten"
        report.longrepr = f"UNWRITTEN  {message}"
        report.sections = []


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # This hook has no access to config; we stash state on the module.
    state = _ACTIVE.get("state")
    if state is None or state.root is None:
        return
    key = state.key_for(report.nodeid, _ACTIVE["rootpath"])
    if not key:
        return
    res = state.result(key)
    if report.when == "call":
        if report.passed:
            res.passed += 1
        elif report.failed:
            res.failed += 1
            if getattr(report, "dungeon_unwritten", False):
                res.unwritten += 1
        elif report.skipped:
            res.skipped += 1
            _remember_skip(state, key, report)
    elif report.when == "setup":
        if report.failed:
            res.errors += 1
            if getattr(report, "dungeon_unwritten", False):
                res.unwritten += 1
        elif report.skipped:
            res.skipped += 1
            _remember_skip(state, key, report)
    elif report.when == "teardown" and report.failed:
        res.errors += 1


def _remember_skip(state: _State, key: str, report: pytest.TestReport) -> None:
    reason = ""
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    elif longrepr is not None:
        reason = str(longrepr)
    if reason.startswith("Skipped: "):
        reason = reason[len("Skipped: "):]
    if reason:
        state.skip_reasons.setdefault(key, reason)


_ACTIVE: dict = {}


def pytest_sessionstart(session: pytest.Session) -> None:
    _ACTIVE["state"] = _state(session.config)
    _ACTIVE["rootpath"] = Path(str(session.config.rootpath))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = _state(session.config)
    if state is None or state.root is None:
        return
    state.cleared_before = {}  # type: ignore[attr-defined]
    if prog.recording_disabled():
        return
    data = prog.load(state.root)
    floors_before = {f.id: prog.floor_status(f, data).cleared for f in state.floors.values()}
    for key in state.order:
        res = state.results[key]
        if res.ran == 0 and res.skipped == 0:
            continue
        state.cleared_before[key] = prog.is_cleared(data, key)  # type: ignore[attr-defined]
        prog.record_trial(key, res, state.root)
    state.floors_before = floors_before  # type: ignore[attr-defined]


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config):
    # Run as the outermost wrapper so the dungeon's verdict is the last thing printed.
    yield
    state = _state(config)
    if state is None or state.root is None or not state.order:
        return
    tr = terminalreporter
    write = tr.write_line
    write("")
    write(ui.title("DUNGEON TRIALS"))
    solutions = solutions_mode()
    if solutions:
        write(ui.yellow("  (reference solutions mode: nothing is recorded to your save file)"))

    data = prog.load(state.root) if not solutions else None
    newly_cleared_floors: list[Floor] = []
    seen_floors: set[str] = set()

    for key in state.order:
        res = state.results[key]
        floor = state.floors[key]
        room = state.rooms[key]
        label = _room_label(floor, room)
        seen_floors.add(floor.id)

        if res.collected == 0:
            continue
        total = res.collected
        if res.deselected:
            write(
                ui.yellow(f"  ~ {label}: ran a subset ({res.passed}/{res.ran} passed, "
                          f"{res.deselected} deselected). Subsets never clear a room.")
            )
            continue
        if res.skipped == total and res.passed == 0:
            reason = state.skip_reasons.get(key, "skipped")
            write(ui.yellow(f"  ⊘ {label}: skipped - {reason}"))
            continue
        if res.cleared:
            glyph = ui.glyph("boss_defeated") if room.is_boss else (
                ui.glyph("secret_found") if room.is_secret else ui.glyph("cleared"))
            if room.is_boss:
                write(ui.green(ui.bold(f"  {glyph} BOSS DEFEATED - {room.name} ({res.passed}/{total})")))
                write(ui.green("    " + floor.line("boss_defeated", "Its weakness was exactly what you thought it was.")))
            elif room.is_secret:
                write(ui.green(ui.bold(f"  {glyph} SECRET ROOM CLEARED - {room.name} ({res.passed}/{total})")))
            else:
                write(ui.green(f"  {glyph} {label} - cleared ({res.passed}/{total})"))
                write(ui.green("    " + floor.line("room_cleared", "The way forward opens.")))
        elif res.unwritten and res.unwritten == res.failed + res.errors:
            path = floor.path.relative_to(state.root) / room.file
            write(ui.cyan(f"  ✎ {label} - unwritten ({res.passed}/{total}). Open {path}"))
        else:
            write(ui.red(f"  ✗ {label} - {res.passed}/{total} trials passed"))
            write(ui.red("    " + floor.line("room_failed", "The trial is not fooled. Read the failures above.")))

    if data is not None:
        floors_before = getattr(state, "floors_before", {})
        for fid in seen_floors:
            floor = next(f for f in state.floors.values() if f.id == fid)
            status = prog.floor_status(floor, data)
            if status.cleared and not floors_before.get(fid, False):
                newly_cleared_floors.append(floor)

    for floor in newly_cleared_floors:
        write("")
        lines = [ui.bold(f"FLOOR {floor.number} CLEARED - {floor.name}")]
        cleared_line = floor.line("floor_cleared", "")
        if cleared_line:
            lines.append(cleared_line)
        if floor.loot:
            lines.append("")
            lines.append("Loot unlocked:")
            for loot in floor.loot:
                lines.append(f"  {ui.glyph('secret_found')} {loot.name}  ->  {floor.path.relative_to(state.root) / loot.file}")
        lines.append("")
        lines.append(ui.dim("Run `dungeon map` to see the way down."))
        for line in ui.box(lines).splitlines():
            write(ui.green(line))

    hint_line = ui.dim("  Stuck? `dungeon hint <floor> <room>` reveals one hint at a time.")
    if any(not r.cleared for r in state.results.values()) and not solutions:
        write(hint_line)
    write("")


def _room_label(floor: Floor, room: Room) -> str:
    if room.is_boss:
        return f"Boss: {room.name}"
    if room.is_secret:
        return f"Secret: {room.name}"
    idx = [r.id for r in floor.regular_rooms]
    n = idx.index(room.id) + 1 if room.id in idx else "?"
    return f"Room {floor.number}.{n} {room.name}"
