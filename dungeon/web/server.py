"""The HTTP surface: JSON for the React app, Server-Sent Events for live output.

Every endpoint reads through ``dungeon.registry`` and ``dungeon.progress``. The
browser is a view; it never computes its own idea of "cleared".
"""

from __future__ import annotations

import asyncio
import importlib
import json
import platform
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dungeon import __version__
from dungeon import progress as prog
from dungeon.cli import install_hint, missing_requirements
from dungeon.registry import DungeonError, Floor, Room, load_floors, resolve_floor
from dungeon.web.runs import Run, RunManager

STATIC_DIR = Path(__file__).resolve().parent / "static"
READABLE_SUFFIXES = {".md", ".py", ".toml", ".txt", ".json", ".yml", ".yaml"}
READABLE_ROOTS = ("floors", "README.md", "EPILOGUE.md", "CONTRIBUTING.md", "LICENSE")
MAX_FILE_BYTES = 2_000_000


# --------------------------------------------------------------------- helpers

def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _hints_for(floor: Floor, room: Room) -> list[str]:
    hint_file = floor.path / "hints" / f"{room.id}.md"
    if not hint_file.is_file():
        return []
    return [h.strip() for h in hint_file.read_text().split("\n---\n") if h.strip()]


def _room_state(cleared: bool, entry: dict | None) -> str:
    if cleared:
        return "cleared"
    if entry is None or not entry.get("attempts"):
        return "unexplored"
    failures = int(entry.get("failed", 0)) + int(entry.get("errors", 0))
    if entry.get("unwritten") and entry["unwritten"] == failures:
        return "unwritten"
    if failures == 0 and int(entry.get("deselected", 0)) > 0:
        return "partial"
    return "failed"


def _room_payload(root: Path, floor: Floor, room: Room, data: dict, label: str) -> dict:
    key = floor.trial_key(room)
    entry = prog.trial_entry(data, key)
    cleared = prog.is_cleared(data, key)
    return {
        "id": room.id,
        "key": key,
        "label": label,
        "name": room.name,
        "kind": room.kind,
        "mode": room.mode,
        "blurb": room.blurb,
        "file": _rel(root, floor.path / room.file),
        "trial": _rel(root, floor.path / room.trial),
        "solution": _rel(root, floor.path / "solutions" / Path(room.file).name),
        "cleared": cleared,
        "state": _room_state(cleared, entry),
        "result": entry,
        "hints": {
            "used": prog.hint_level(data, f"{floor.id}/{room.id}"),
            "total": len(_hints_for(floor, room)),
        },
    }


def _floor_payload(root: Path, floor: Floor, data: dict) -> dict:
    status = prog.floor_status(floor, data)
    rooms = []
    for i, room in enumerate(floor.regular_rooms, start=1):
        rooms.append(_room_payload(root, floor, room, data, f"{floor.number}.{i}"))
    if floor.boss:
        rooms.append(_room_payload(root, floor, floor.boss, data, "BOSS"))
    if floor.secret:
        rooms.append(_room_payload(root, floor, floor.secret, data, "SECRET"))
    missing = missing_requirements(floor)
    return {
        "id": floor.id,
        "number": floor.number,
        "slug": floor.slug,
        "name": floor.name,
        "tagline": floor.tagline,
        "topics": list(floor.topics),
        "requires": list(floor.requires),
        "missing": missing,
        "install_hint": install_hint(missing) if missing else None,
        "map": floor.map_art,
        "lines": floor.lines,
        "readme": _rel(root, floor.readme),
        "rooms": rooms,
        "loot": [
            {
                "name": loot.name,
                "file": _rel(root, floor.path / loot.file),
                "blurb": loot.blurb,
                "unlocked": status.cleared,
            }
            for loot in floor.loot
        ],
        "status": {
            "cleared": status.cleared,
            "started": status.started,
            "required_done": status.required_done,
            "required_total": status.required_total,
            "boss_defeated": status.boss_defeated,
            "secret_found": status.secret_found,
        },
    }


def _doctor(root: Path, floors: list[Floor]) -> list[dict]:
    checks: list[dict] = []

    def check(label: str, ok: bool, detail: str = "", level: str | None = None) -> None:
        checks.append({"label": label, "ok": ok, "detail": detail, "level": level or ("ok" if ok else "error")})

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
        check("torch", True, f"{torch.__version__}  device: {device}")
    except Exception:
        check("torch", True, "not installed - needed from floor 4 on", level="warn")
    check("dungeon root", True, str(root))
    check("floors", len(floors) > 0, f"{len(floors)} found")
    save = prog.progress_path(root)
    check("save file", True, str(save) + ("" if save.is_file() else "  (none yet)"))
    return checks


class _Watcher:
    """Polls the save file and the room files; bumps a version when anything changes."""

    def __init__(self, root: Path, floors: list[Floor]) -> None:
        self.root = root
        self.paths = [prog.progress_path(root)] + [
            floor.path / room.file for floor in floors for room in floor.rooms
        ]
        self.version = 0
        self._last = self._stamp()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _stamp(self) -> tuple:
        out = []
        for p in self.paths:
            try:
                st = p.stat()
                out.append((st.st_mtime_ns, st.st_size))
            except FileNotFoundError:
                out.append(None)
        return tuple(out)

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            stamp = self._stamp()
            if stamp != self._last:
                self._last = stamp
                self.version += 1

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------------------ requests

class RunRequest(BaseModel):
    floor: str
    scope: str = Field("rooms", pattern="^(rooms|room|boss|secret|all)$")
    room: str | None = None
    verbose: bool = False
    fail_fast: bool = False
    keyword: str | None = Field(None, max_length=200, pattern=r"^[\w\s\-().]*$")


class ResetRequest(BaseModel):
    floor: str | None = None


# ----------------------------------------------------------------------- app

def create_app(root: Path) -> FastAPI:
    root = root.resolve()
    floors = load_floors(root)
    runs = RunManager(root, floors)
    watcher = _Watcher(root, floors)
    app = FastAPI(title="Neural Dungeon", version=__version__, docs_url=None, redoc_url=None)

    def floor_or_404(ref: str) -> Floor:
        try:
            return resolve_floor(ref, root)
        except DungeonError as exc:
            raise HTTPException(404, str(exc)) from exc

    def room_or_404(floor: Floor, ref: str) -> Room:
        room = floor.room_by_ref(ref)
        if room is None:
            raise HTTPException(404, f"No room {ref!r} on floor {floor.number}.")
        return room

    # ------------------------------------------------------------ state
    @app.get("/api/state")
    def state() -> dict:
        data = prog.load(root)
        summary = prog.summarize(floors, data)
        return {
            "version": __version__,
            "root": str(root),
            "floors": [_floor_payload(root, f, data) for f in floors],
            "summary": {**asdict(summary), "fraction": summary.fraction, "rank": summary.rank},
            "ranks": [{"threshold": t, "title": name} for t, name in prog.RANKS],
            "active_run": runs.active.summary() if runs.active else None,
        }

    @app.get("/api/doctor")
    def doctor() -> dict:
        return {"checks": _doctor(root, floors)}

    @app.get("/api/file")
    def read_file(path: str = Query(..., max_length=500)) -> dict:
        if path.startswith("/") or ".." in Path(path).parts or not path.startswith(READABLE_ROOTS):
            raise HTTPException(403, "That path is outside the dungeon.")
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404, f"No such file: {path}")
        if target.suffix not in READABLE_SUFFIXES and target.name != "LICENSE":
            raise HTTPException(403, "Only text files can be read here.")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise HTTPException(413, "That file is too large to show.")
        return {"path": path, "content": target.read_text(errors="replace"), "suffix": target.suffix}

    # ------------------------------------------------------------ hints
    def hints_payload(floor: Floor, room: Room, data: dict) -> dict:
        hints = _hints_for(floor, room)
        level = min(prog.hint_level(data, f"{floor.id}/{room.id}"), len(hints))
        return {"total": len(hints), "level": level, "hints": hints[:level]}

    @app.get("/api/floors/{floor_ref}/rooms/{room_ref}/hints")
    def get_hints(floor_ref: str, room_ref: str) -> dict:
        floor = floor_or_404(floor_ref)
        room = room_or_404(floor, room_ref)
        return hints_payload(floor, room, prog.load(root))

    @app.post("/api/floors/{floor_ref}/rooms/{room_ref}/hints/reveal")
    def reveal_hint(floor_ref: str, room_ref: str, all: bool = False) -> dict:
        floor = floor_or_404(floor_ref)
        room = room_or_404(floor, room_ref)
        hints = _hints_for(floor, room)
        key = f"{floor.id}/{room.id}"
        level = prog.hint_level(prog.load(root), key)
        level = len(hints) if all else min(level + 1, len(hints))
        prog.set_hint_level(key, level, root)
        return hints_payload(floor, room, prog.load(root))

    # ------------------------------------------------------------ runs
    @app.post("/api/runs", status_code=201)
    def start_run(req: RunRequest) -> dict:
        floor = floor_or_404(req.floor)
        missing = missing_requirements(floor)
        if missing:
            raise HTTPException(409, f"Floor {floor.number} needs {', '.join(missing)}. " + install_hint(missing))
        if req.scope == "room":
            if not req.room:
                raise HTTPException(422, "scope=room needs a room.")
            room = room_or_404(floor, req.room)
            rooms = [room]
            title = ("BOSS FIGHT" if room.is_boss else "SECRET ROOM" if room.is_secret else "TRIAL") + f" - {room.name}"
        elif req.scope == "boss":
            if floor.boss is None:
                raise HTTPException(404, f"Floor {floor.number} has no boss. Lucky you.")
            rooms = [floor.boss]
            title = f"BOSS FIGHT - {floor.boss.name}"
        elif req.scope == "secret":
            if floor.secret is None:
                raise HTTPException(404, f"Floor {floor.number} has no secret room. Or does it?")
            rooms = [floor.secret]
            title = f"SECRET ROOM - {floor.secret.name}"
        elif req.scope == "all":
            rooms = list(floor.rooms)
            title = f"EVERY TRIAL ON FLOOR {floor.number}"
        else:
            rooms = floor.regular_rooms
            title = f"TRIALS OF FLOOR {floor.number} - {floor.name}"
        try:
            run = runs.start(floor, rooms, title, verbose=req.verbose, fail_fast=req.fail_fast, keyword=req.keyword)
        except RuntimeError as exc:
            active = runs.active
            raise HTTPException(409, str(exc), headers={"X-Active-Run": active.id if active else ""}) from exc
        return run.summary()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "No such run.")
        return {**run.summary(), "output": run.text}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "No such run.")
        runs.cancel(run)
        return run.summary()

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str, request: Request) -> StreamingResponse:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "No such run.")
        return StreamingResponse(_stream(run, request), media_type="text/event-stream", headers=_SSE_HEADERS)

    # ------------------------------------------------------------ events
    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        return StreamingResponse(_events(watcher, request), media_type="text/event-stream", headers=_SSE_HEADERS)

    # ------------------------------------------------------------ reset
    @app.post("/api/reset")
    def reset(req: ResetRequest) -> dict:
        floor = floor_or_404(req.floor) if req.floor else None
        return {"removed": prog.reset(root, floor)}

    # ------------------------------------------------------------ static
    if (STATIC_DIR / "index.html").is_file():
        if (STATIC_DIR / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (STATIC_DIR / path).resolve()
            if path and candidate.is_relative_to(STATIC_DIR) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")
    else:
        @app.get("/{path:path}", include_in_schema=False)
        def unbuilt(path: str) -> HTMLResponse:
            return HTMLResponse(
                "<pre>The web frontend is not built.\n\n"
                "  cd web && npm install && npm run build\n\n"
                "then start `dungeon serve` again. The API is live under /api.</pre>",
                status_code=503,
            )

    app.state.watcher = watcher
    app.state.runs = runs
    return app


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream(run: Run, request: Request) -> AsyncIterator[str]:
    """Replay a run's output so far, then follow it until `done`. Closing the tab never cancels the run."""
    yield _sse("start", run.summary())
    sent = 0
    while not await request.is_disconnected():
        chunks = run.chunks
        if sent < len(chunks):
            yield _sse("output", {"text": "".join(chunks[sent:])})
            sent = len(chunks)
        if run.done and sent >= len(run.chunks):
            yield _sse("done", run.summary())
            return
        await asyncio.sleep(0.05)


async def _events(watcher: _Watcher, request: Request) -> AsyncIterator[str]:
    seen = watcher.version
    last_beat = time.time()
    yield _sse("hello", {"version": seen})
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)
        if watcher.version != seen:
            seen = watcher.version
            yield _sse("changed", {"version": seen})
            last_beat = time.time()
        elif time.time() - last_beat > 15:
            yield ": keep-alive\n\n"
            last_beat = time.time()


def serve(root: Path, port: int = 8642, open_browser: bool = True) -> int:
    import uvicorn

    app = create_app(root)
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"  Neural Dungeon is open at {url}   (Ctrl-C to close the gate)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0
