"""Discover the floors of the dungeon.

Every floor is a directory under ``floors/`` containing a ``floor.toml`` that
describes its rooms, boss, secret room and loot. This module turns those files
into plain dataclasses that the CLI and the pytest plugin both rely on.

Nothing here imports numpy or torch; the registry must work even in a broken
environment so that ``dungeon doctor`` can tell you what is wrong.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

FLOORS_DIRNAME = "floors"
FLOOR_FILE = "floor.toml"
PROGRESS_DIRNAME = ".dungeon"

ROOM_KINDS = ("room", "boss", "secret")


class DungeonError(RuntimeError):
    """Raised when the dungeon itself (not your code) is misconfigured."""


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    kind: str
    file: str
    trial: str
    blurb: str = ""

    @property
    def trial_stem(self) -> str:
        return Path(self.trial).stem

    @property
    def is_boss(self) -> bool:
        return self.kind == "boss"

    @property
    def is_secret(self) -> bool:
        return self.kind == "secret"


@dataclass(frozen=True)
class Loot:
    name: str
    file: str
    blurb: str = ""


@dataclass(frozen=True)
class Floor:
    id: str
    number: int
    slug: str
    name: str
    tagline: str
    path: Path
    topics: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    map_art: str = ""
    rooms: tuple[Room, ...] = ()
    loot: tuple[Loot, ...] = ()
    lines: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def regular_rooms(self) -> list[Room]:
        return [r for r in self.rooms if r.kind == "room"]

    @property
    def boss(self) -> Room | None:
        return next((r for r in self.rooms if r.kind == "boss"), None)

    @property
    def secret(self) -> Room | None:
        return next((r for r in self.rooms if r.kind == "secret"), None)

    @property
    def required_rooms(self) -> list[Room]:
        """Everything that must be cleared for the floor to count as cleared."""
        return [r for r in self.rooms if r.kind != "secret"]

    @property
    def readme(self) -> Path:
        return self.path / "README.md"

    @property
    def trials_dir(self) -> Path:
        return self.path / "trials"

    def trial_key(self, room: Room) -> str:
        """The key under which progress for this room's trial is stored."""
        return f"{self.id}/{room.trial_stem}"

    def room_by_ref(self, ref: str) -> Room | None:
        """Find a room by id, trial stem, kind ("boss"/"secret") or a substring."""
        ref = ref.strip().lower()
        for room in self.rooms:
            if ref in (room.id.lower(), room.trial_stem.lower()):
                return room
        if ref in ("boss", "secret"):
            return self.boss if ref == "boss" else self.secret
        for room in self.rooms:
            if ref in room.name.lower() or ref in room.file.lower():
                return room
        # "1", "2" ... as shorthand for room_1, room_2
        if ref.isdigit():
            return self.room_by_ref(f"room_{int(ref)}")
        return None

    def line(self, key: str, default: str) -> str:
        return self.lines.get(key, default)


# ---------------------------------------------------------------------- root

def find_root(start: Path | None = None) -> Path:
    """Locate the repository root (the directory that contains ``floors/``).

    Walks upwards from ``start`` (default: the current directory). Falls back to
    the parent of this package, which is right for an editable install.
    """
    candidates = []
    here = (start or Path.cwd()).resolve()
    candidates.extend([here, *here.parents])
    candidates.append(Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / FLOORS_DIRNAME).is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise DungeonError(
        "I can't find the dungeon. Run this from inside your clone of neural-dungeon "
        "(the directory that contains `floors/`)."
    )


# --------------------------------------------------------------------- loading

def _parse_room(raw: dict, floor_dir: Path) -> Room:
    kind = raw.get("kind", "room")
    if kind not in ROOM_KINDS:
        raise DungeonError(f"{floor_dir.name}: room kind must be one of {ROOM_KINDS}, got {kind!r}")
    for key in ("id", "name", "file", "trial"):
        if key not in raw:
            raise DungeonError(f"{floor_dir.name}: room is missing required key {key!r}: {raw}")
    return Room(
        id=raw["id"],
        name=raw["name"],
        kind=kind,
        file=raw["file"],
        trial=raw["trial"],
        blurb=raw.get("blurb", "").strip(),
    )


def load_floor(floor_dir: Path) -> Floor:
    toml_path = floor_dir / FLOOR_FILE
    with toml_path.open("rb") as fh:
        raw = tomllib.load(fh)
    try:
        rooms = tuple(_parse_room(r, floor_dir) for r in raw.get("rooms", []))
        loot = tuple(
            Loot(name=item["name"], file=item["file"], blurb=item.get("blurb", "").strip())
            for item in raw.get("loot", [])
        )
        floor = Floor(
            id=raw["id"],
            number=int(raw["number"]),
            slug=raw["slug"],
            name=raw["name"],
            tagline=raw.get("tagline", "").strip(),
            path=floor_dir,
            topics=tuple(raw.get("topics", [])),
            requires=tuple(raw.get("requires", [])),
            map_art=raw.get("map", "").rstrip("\n"),
            rooms=rooms,
            loot=loot,
            lines={k: str(v).strip() for k, v in raw.get("lines", {}).items()},
        )
    except KeyError as exc:  # pragma: no cover - authoring error
        raise DungeonError(f"{toml_path}: missing key {exc}") from exc

    bosses = [r for r in rooms if r.is_boss]
    if len(bosses) > 1:
        raise DungeonError(f"{floor.id}: a floor has at most one boss, found {len(bosses)}")
    ids = [r.id for r in rooms]
    if len(ids) != len(set(ids)):
        raise DungeonError(f"{floor.id}: duplicate room ids {ids}")
    return floor


@lru_cache(maxsize=None)
def _load_all(root: Path) -> tuple[Floor, ...]:
    floors_dir = root / FLOORS_DIRNAME
    found = []
    for toml_path in sorted(floors_dir.glob(f"*/{FLOOR_FILE}")):
        found.append(load_floor(toml_path.parent))
    found.sort(key=lambda f: f.number)
    return tuple(found)


def load_floors(root: Path | None = None) -> list[Floor]:
    """All floors, sorted by number."""
    return list(_load_all((root or find_root()).resolve()))


def resolve_floor(ref: str | int, root: Path | None = None) -> Floor:
    """Accept ``3``, ``"3"``, ``"03"``, ``"floor_03"`` or any unique part of the name/slug."""
    floors = load_floors(root)
    text = str(ref).strip().lower()
    if text.isdigit():
        for floor in floors:
            if floor.number == int(text):
                return floor
    for floor in floors:
        if text in (floor.id.lower(), floor.slug.lower(), floor.path.name.lower()):
            return floor
    matches = [f for f in floors if text in f.name.lower() or text in f.slug.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(f"{m.number}: {m.name}" for m in matches)
        raise DungeonError(f"{ref!r} matches several floors ({names}). Be more specific.")
    raise DungeonError(f"No floor matches {ref!r}. Try `dungeon map` to see them all.")


def floor_for_path(path: Path, root: Path | None = None) -> Floor | None:
    """Which floor does a file (e.g. a trial) belong to? None if not in a floor."""
    root = (root or find_root()).resolve()
    path = path.resolve()
    for floor in load_floors(root):
        try:
            path.relative_to(floor.path.resolve())
        except ValueError:
            continue
        return floor
    return None
