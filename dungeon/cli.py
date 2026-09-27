"""The ``dungeon`` command: your torch, map and save file.

    dungeon                 the map (same as `dungeon map`)
    dungeon enter 1         walk onto a floor: lore, rooms, status
    dungeon trial 1         run every room trial on floor 1
    dungeon trial 1 room_2  run one room's trial
    dungeon fight 1         fight the boss
    dungeon trial 1 --secret  attempt the secret room
    dungeon hint 1 room_2   reveal the next hint for a room
    dungeon loot            what you have unlocked
    dungeon status          your rank and numbers
    dungeon doctor          check your environment
    dungeon serve           play in the browser
    dungeon reset           forget everything (asks first)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

from dungeon import __version__, ui
from dungeon import progress as prog
from dungeon.registry import DungeonError, Floor, Room, find_root, load_floors, resolve_floor

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def missing_requirements(floor: Floor) -> list[str]:
    return [m for m in floor.requires if not _module_available(m)]


def install_hint(missing: list[str]) -> str:
    if "torch" in missing:
        return (
            "Install PyTorch (CPU is plenty):\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  or, if you want the default wheel for your platform:\n"
            "    pip install -e '.[torch]'"
        )
    return "Install: pip install " + " ".join(missing)


def _room_status_glyph(status: prog.FloorStatus, room: Room, data: dict) -> str:
    key = status.floor.trial_key(room)
    if room.is_boss:
        return ui.green(ui.glyph("boss_defeated")) if status.boss_defeated else ui.red(ui.glyph("boss_alive"))
    if room.is_secret:
        return ui.green(ui.glyph("secret_found")) if status.secret_found else ui.dim(ui.glyph("secret_hidden"))
    return ui.green(ui.glyph("room_done")) if prog.is_cleared(data, key) else ui.dim(ui.glyph("room_open"))


def _progress_bar(status: prog.FloorStatus, data: dict) -> str:
    cells = []
    for room in status.floor.regular_rooms:
        done = prog.is_cleared(data, status.floor.trial_key(room))
        cells.append(ui.green(ui.glyph("room_done")) if done else ui.dim(ui.glyph("room_open")))
    if status.floor.boss:
        cells.append(" " + (ui.green(ui.glyph("boss_defeated")) if status.boss_defeated else ui.red(ui.glyph("boss_alive"))))
    if status.floor.secret:
        cells.append(ui.green(ui.glyph("secret_found")) if status.secret_found else ui.dim(ui.glyph("secret_hidden")))
    return "".join(cells)


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------


def cmd_map(args: argparse.Namespace, root: Path) -> int:
    floors = load_floors(root)
    data = prog.load(root)
    print(ui.banner())
    print()
    print(f"  {ui.glyph('surface')} the surface  {ui.dim('- you came in this way. Every floor is open; go in order the first time.')}")
    print("  │")
    rows: list[str] = []
    you_are_here_done = False
    for floor in floors:
        status = prog.floor_status(floor, data)
        name = f"F{floor.number:02d}  {floor.name}"
        bar = _progress_bar(status, data)
        if status.cleared:
            state = ui.green(ui.bold(f"CLEARED {ui.glyph('cleared')}"))
        elif status.started:
            state = ui.yellow(f"{status.required_done}/{status.required_total}")
        else:
            state = ui.dim("unexplored")
        marker = ""
        if not status.cleared and not you_are_here_done:
            marker = ui.cyan(ui.bold("  <- you are here"))
            you_are_here_done = True
        missing = missing_requirements(floor)
        if missing:
            state += ui.dim(f"  (needs {', '.join(missing)})")
        rows.append(f"{ui.pad(name, 38)} {ui.pad(bar, 10)} {state}{marker}")
    plural = "floor" if len(floors) == 1 else "floors"
    print(ui.box(rows, header=f"NEURAL DUNGEON - {len(floors)} {plural}"))
    print()
    print(ui.dim(
        f"  legend  {ui.glyph('room_done')} room cleared  {ui.glyph('room_open')} room open  "
        f"{ui.glyph('boss_alive')} boss  {ui.glyph('secret_hidden')} secret room  {ui.glyph('cleared')} floor cleared"
    ))
    print(ui.dim("  next    dungeon enter <floor>    ·    dungeon status    ·    dungeon doctor"))
    return 0


def cmd_enter(args: argparse.Namespace, root: Path) -> int:
    floor = resolve_floor(args.floor, root)
    data = prog.load(root)
    status = prog.floor_status(floor, data)

    print(ui.title(f"FLOOR {floor.number} - {floor.name.upper()}"))
    if floor.tagline:
        print(ui.wrap(ui.paint(floor.tagline, "italic")))
    if floor.topics:
        print()
        print("  " + ui.dim("you will learn: ") + ", ".join(floor.topics))
    missing = missing_requirements(floor)
    if missing:
        print()
        print(ui.yellow(f"  This floor needs: {', '.join(missing)}"))
        print(ui.yellow("  " + install_hint(missing).replace("\n", "\n  ")))
    if floor.map_art:
        print()
        print(ui.cyan(floor.map_art))
    print()
    print(ui.bold("  Rooms"))
    for i, room in enumerate(floor.regular_rooms, start=1):
        g = _room_status_glyph(status, room, data)
        tag = ui.magenta(" ☿ cursed code") if room.is_cursed else ""
        print(f"   {g} {floor.number}.{i} {ui.bold(room.name):<40} {ui.dim(room.file)}{tag}")
        if room.blurb:
            print(ui.dim(ui.wrap(room.blurb, indent="       ")))
    if floor.boss:
        g = _room_status_glyph(status, floor.boss, data)
        print(f"   {g} {ui.bold('BOSS  ' + floor.boss.name):<44} {ui.dim(floor.boss.file)}")
        if floor.boss.blurb:
            print(ui.dim(ui.wrap(floor.boss.blurb, indent="       ")))
    if floor.secret:
        g = _room_status_glyph(status, floor.secret, data)
        print(f"   {g} {ui.bold('SECRET  ' + floor.secret.name):<44} {ui.dim(floor.secret.file)}")
        if floor.secret.blurb:
            print(ui.dim(ui.wrap(floor.secret.blurb, indent="       ")))
    if floor.loot:
        print()
        print(ui.bold("  Loot") + ui.dim("  (unlocked when the floor is cleared)"))
        for loot in floor.loot:
            lock = ui.green(ui.glyph("secret_found")) if status.cleared else ui.dim("🔒")
            print(f"   {lock} {loot.name:<40} {ui.dim(_rel(root, floor.path / loot.file))}")
    print()
    readme = _rel(root, floor.readme)
    print(ui.bold("  Begin: ") + f"read {ui.cyan(readme)}" + ui.dim(f"   (or: dungeon enter {floor.number} --readme)"))
    print(ui.bold("  Then:  ") + f"dungeon trial {floor.number}      "
          + ui.dim("· dungeon trial %d <room> · dungeon fight %d · dungeon hint %d <room>" % (floor.number, floor.number, floor.number)))
    if args.readme:
        print()
        print(ui.hr())
        print(floor.readme.read_text())
    return 0


def _run_pytest(root: Path, paths: list[Path], extra: list[str], verbose: bool) -> int:
    cmd = [sys.executable, "-m", "pytest", *[_rel(root, p) for p in paths]]
    if not verbose and not any(a.startswith("-v") for a in extra):
        cmd.append("-q")
    cmd.extend(extra)
    print(ui.dim("  $ " + " ".join(cmd[1:]).replace("-m pytest", "pytest")))
    print()
    sys.stdout.flush()  # keep our header ahead of pytest's output when piped
    proc = subprocess.run(cmd, cwd=root)
    return proc.returncode


def cmd_trial(args: argparse.Namespace, root: Path, boss: bool = False) -> int:
    floor = resolve_floor(args.floor, root)
    missing = missing_requirements(floor)
    if missing:
        print(ui.red(f"  Floor {floor.number} needs {', '.join(missing)} before its trials can run."))
        print(ui.yellow("  " + install_hint(missing).replace("\n", "\n  ")))
        return 2

    rooms: list[Room]
    if boss:
        if floor.boss is None:
            raise DungeonError(f"Floor {floor.number} has no boss. Lucky you.")
        rooms = [floor.boss]
        print(ui.title(f"BOSS FIGHT - {floor.boss.name.upper()}"))
        print(ui.wrap(ui.paint(floor.line("boss_intro", "It has been waiting for you."), "italic")))
        print()
    elif getattr(args, "room", None):
        room = floor.room_by_ref(args.room)
        if room is None:
            names = ", ".join(r.id for r in floor.rooms)
            raise DungeonError(f"No room {args.room!r} on floor {floor.number}. Rooms: {names}")
        rooms = [room]
        print(ui.title(f"TRIAL - {room.name.upper()}"))
    elif getattr(args, "secret", False):
        if floor.secret is None:
            raise DungeonError(f"Floor {floor.number} has no secret room. Or does it?")
        rooms = [floor.secret]
        print(ui.title(f"SECRET ROOM - {floor.secret.name.upper()}"))
    elif getattr(args, "all", False):
        rooms = list(floor.rooms)
        print(ui.title(f"EVERY TRIAL ON FLOOR {floor.number}"))
    else:
        rooms = floor.regular_rooms
        print(ui.title(f"TRIALS OF FLOOR {floor.number} - {floor.name.upper()}"))
        print(ui.dim("  (rooms only; `dungeon fight %d` for the boss, `--secret` for the secret room)" % floor.number))
    paths = [floor.path / r.trial for r in rooms]
    for p in paths:
        if not p.is_file():
            raise DungeonError(f"Trial file missing: {p}. This is a bug in the dungeon, not in you.")
    return _run_pytest(root, paths, args.pytest_args, args.verbose)


def cmd_fight(args: argparse.Namespace, root: Path) -> int:
    return cmd_trial(args, root, boss=True)


def cmd_hint(args: argparse.Namespace, root: Path) -> int:
    floor = resolve_floor(args.floor, root)
    room = floor.room_by_ref(args.room)
    if room is None:
        names = ", ".join(r.id for r in floor.rooms)
        raise DungeonError(f"No room {args.room!r} on floor {floor.number}. Rooms: {names}")
    hint_file = floor.path / "hints" / f"{room.id}.md"
    if not hint_file.is_file():
        print(ui.yellow(f"  No hints were written for {room.name}. The README is your hint."))
        return 0
    hints = [h.strip() for h in hint_file.read_text().split("\n---\n") if h.strip()]
    key = f"{floor.id}/{room.id}"
    data = prog.load(root)
    level = prog.hint_level(data, key)
    if args.all:
        level = len(hints)
    elif level < len(hints):
        level += 1
    prog.set_hint_level(key, level, root)
    print(ui.title(f"HINTS - {room.name.upper()}"))
    for i, hint in enumerate(hints[:level], start=1):
        print(ui.bold(f"  Hint {i}/{len(hints)}"))
        print(ui.wrap(hint, indent="    "))
        print()
    if level < len(hints):
        print(ui.dim(f"  {len(hints) - level} more hint(s) available. Run the same command again to reveal the next."))
    else:
        print(ui.dim("  That was the last hint. The solution lives in solutions/ if you must, but try once more first."))
    return 0


def cmd_loot(args: argparse.Namespace, root: Path) -> int:
    floors = [resolve_floor(args.floor, root)] if args.floor else load_floors(root)
    data = prog.load(root)
    print(ui.title("LOOT"))
    any_loot = False
    for floor in floors:
        if not floor.loot:
            continue
        any_loot = True
        status = prog.floor_status(floor, data)
        print(ui.bold(f"  Floor {floor.number} - {floor.name}"))
        for loot in floor.loot:
            path = _rel(root, floor.path / loot.file)
            if status.cleared:
                print(f"   {ui.green(ui.glyph('secret_found'))} {loot.name:<40} {ui.cyan(path)}")
            else:
                boss = floor.boss.name if floor.boss else "the floor"
                print(f"   {ui.dim(ui.glyph('locked'))} {ui.dim(loot.name):<40} {ui.dim('locked - defeat ' + boss)}")
            if loot.blurb and (status.cleared or args.peek):
                print(ui.dim(ui.wrap(loot.blurb, indent="       ")))
        print()
    if not any_loot:
        print("  Nothing here yet.")
    print(ui.dim("  Loot files are ordinary files in the repo. Locks are on the honour system."))
    return 0


def cmd_status(args: argparse.Namespace, root: Path) -> int:
    s = prog.summarize(load_floors(root), prog.load(root))

    print(ui.title("STATUS"))
    print(f"  Rank            {ui.bold(ui.magenta(s.rank))}")
    print(f"  Floors cleared  {s.floors_cleared}/{s.floors_total}")
    print(f"  Rooms cleared   {s.rooms_done}/{s.rooms_total}")
    print(f"  Bosses defeated {s.bosses_done}/{s.bosses_total}")
    print(f"  Secrets found   {s.secrets_done}/{s.secrets_total}")
    print(f"  Trial attempts  {s.attempts}")
    print(f"  Hints used      {s.hints_used}")
    if s.fraction >= 1.0:
        print()
        print(ui.green(ui.bold("  You have cleared the dungeon. Go outside; the surface has missed you.")))
    return 0


def cmd_doctor(args: argparse.Namespace, root: Path) -> int:
    print(ui.title("DOCTOR"))
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = ui.green("ok ") if good else ui.red("!! ")
        print(f"  {mark} {label:<22} {detail}")

    py = sys.version_info
    check("python", py >= (3, 11), f"{platform.python_version()} ({sys.executable})")
    for mod in ("numpy", "pytest"):
        try:
            m = importlib.import_module(mod)
            check(mod, True, getattr(m, "__version__", "?"))
        except Exception as exc:  # pragma: no cover
            check(mod, False, f"missing ({exc})")
    try:
        torch = importlib.import_module("torch")
        cuda = torch.cuda.is_available()
        mps = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
        device = "cuda" if cuda else ("mps" if mps else "cpu")
        check("torch", True, f"{torch.__version__}  device: {device}" + (" (CPU is fine for every floor)" if device == "cpu" else ""))
    except Exception:
        print(f"  {ui.yellow('-- ')} {'torch':<22} not installed - needed from floor 4 on. " + install_hint(['torch']).splitlines()[1].strip())
    check("dungeon root", True, str(root))
    floors = load_floors(root)
    check("floors", len(floors) > 0, f"{len(floors)} found")
    save = prog.progress_path(root)
    check("save file", True, str(save) + ("" if save.is_file() else "  (none yet - it appears after your first trial)"))
    print()
    if ok:
        print(ui.green("  You are ready. `dungeon enter 0` opens the Threshold."))
    else:
        print(ui.red("  Fix the items marked !! and run `dungeon doctor` again."))
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace, root: Path) -> int:
    try:
        from dungeon.web.server import serve
    except ImportError:
        print(ui.red("  The browser needs two more packages:"))
        print(ui.yellow('    pip install -e ".[web]"'))
        return 2
    return serve(root, port=args.port, open_browser=not args.no_browser)


def cmd_reset(args: argparse.Namespace, root: Path) -> int:
    floor = resolve_floor(args.floor, root) if args.floor else None
    what = f"floor {floor.number} ({floor.name})" if floor else "the ENTIRE dungeon"
    if not args.yes:
        answer = input(f"  Forget all progress for {what}? This cannot be undone. [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("  Nothing forgotten.")
            return 0
    removed = prog.reset(root, floor)
    print(ui.yellow(f"  Forgotten: {removed} record(s). The dungeon does not remember you. Yet."))
    return 0


# ----------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dungeon",
        description="Neural Dungeon - a code-first dungeon crawl through AI engineering.",
        epilog=__doc__.split("\n", 1)[1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"neural-dungeon {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("map", help="show the dungeon and your progress")

    p = sub.add_parser("enter", help="walk onto a floor")
    p.add_argument("floor", help="floor number, id or name")
    p.add_argument("--readme", action="store_true", help="also print the floor README")

    def add_pytest_passthrough(p: argparse.ArgumentParser) -> None:
        p.add_argument("-v", "--verbose", action="store_true", help="verbose pytest output")
        p.epilog = "Anything after `--` (or any option this command does not know) is passed to pytest, e.g. `-- -x -k shape`."

    p = sub.add_parser("trial", help="run trials for a floor or a room")
    p.add_argument("floor")
    p.add_argument("room", nargs="?", help="a room id (room_2), number (2), 'boss' or 'secret'")
    p.add_argument("--secret", action="store_true", help="attempt the secret room")
    p.add_argument("--all", action="store_true", help="every trial on the floor including boss and secret")
    add_pytest_passthrough(p)

    p = sub.add_parser("fight", help="fight the boss of a floor")
    p.add_argument("floor")
    add_pytest_passthrough(p)

    p = sub.add_parser("hint", help="reveal the next hint for a room")
    p.add_argument("floor")
    p.add_argument("room")
    p.add_argument("--all", action="store_true", help="reveal every hint")

    p = sub.add_parser("loot", help="list loot, unlocked and locked")
    p.add_argument("floor", nargs="?")
    p.add_argument("--peek", action="store_true", help="show descriptions of locked loot too")

    sub.add_parser("status", help="your rank and numbers")
    sub.add_parser("doctor", help="check that your environment can play")

    p = sub.add_parser("serve", help="play in the browser (local server on 127.0.0.1)")
    p.add_argument("--port", type=int, default=8642)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    p = sub.add_parser("reset", help="forget progress")
    p.add_argument("floor", nargs="?", help="only this floor")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask")
    return parser


COMMANDS = {
    "map": cmd_map,
    "enter": cmd_enter,
    "trial": cmd_trial,
    "fight": cmd_fight,
    "hint": cmd_hint,
    "loot": cmd_loot,
    "status": cmd_status,
    "doctor": cmd_doctor,
    "serve": cmd_serve,
    "reset": cmd_reset,
}


PYTEST_COMMANDS = {"trial", "fight"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after a bare "--" goes to pytest verbatim.
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        passthrough, argv = argv[cut + 1:], argv[:cut]
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    command = args.command or "map"
    if command in PYTEST_COMMANDS:
        # Options this command does not know (-x, -k, --pdb, ...) are pytest's.
        args.pytest_args = unknown + passthrough
    elif unknown or passthrough:
        parser.error(f"unrecognized arguments: {' '.join(unknown + passthrough)}")
    try:
        root = find_root()
        return COMMANDS[command](args, root)
    except DungeonError as exc:
        print(ui.red(f"  {exc}"), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        print(ui.dim("  You back away slowly."))
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
