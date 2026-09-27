"""Tests for `dungeon serve`, the browser interface.

They build a two-room dungeon in a temporary directory and drive the API with
FastAPI's TestClient, including real pytest subprocesses, so they never touch
your save file. They are skipped when the web extras are not installed.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="pip install -e '.[web]'")
pytest.importorskip("httpx", reason="pip install httpx")
from fastapi.testclient import TestClient  # noqa: E402

from dungeon.web.server import _events, create_app  # noqa: E402

STUB = '''"""ROOM 0.1"""


def answer() -> int:
    """Return 42."""
    raise NotImplementedError("answer() is unwritten")
'''

SOLVED = '''"""ROOM 0.1 (solved)"""


def answer() -> int:
    return 42
'''

BOSS_STUB = '''"""BOSS"""


def roar() -> str:
    raise NotImplementedError("roar() is unwritten")
'''

BOSS_SOLVED = '''"""BOSS (solved)"""


def roar() -> str:
    return "ROAR"
'''


def make_dungeon(root: Path) -> Path:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fake-dungeon"\nversion = "0"\n\n[tool.pytest.ini_options]\naddopts = "-rN"\n'
    )
    (root / "conftest.py").write_text('pytest_plugins = ["dungeon.trials"]\n')
    floors = root / "floors"
    floor = floors / "floor_00_test"
    for d in (floors, floor, floor / "rooms", floor / "solutions", floor / "trials", floor / "hints", floor / "loot"):
        d.mkdir(parents=True, exist_ok=True)
    for d in (floors, floor, floor / "rooms", floor / "solutions", floor / "trials"):
        (d / "__init__.py").write_text("")
    (floor / "floor.toml").write_text(
        textwrap.dedent(
            """
            id = "floor_00"
            number = 0
            slug = "test"
            name = "The Test Floor"
            tagline = "A floor for testing."
            topics = ["testing"]
            map = "[ 0.1 ]--[ boss ]"

            [lines]
            room_cleared = "Runes flicker."
            boss_defeated = "It falls."
            floor_cleared = "Onward."

            [[rooms]]
            id = "room_1"
            name = "The Answer"
            kind = "room"
            file = "rooms/room_1_answer.py"
            trial = "trials/test_room_1.py"
            blurb = "Answer."

            [[rooms]]
            id = "boss"
            name = "The Roarer"
            kind = "boss"
            file = "rooms/boss_roarer.py"
            trial = "trials/test_boss.py"

            [[loot]]
            name = "A Sheet"
            file = "loot/sheet.md"
            """
        )
    )
    (floor / "README.md").write_text("# Floor 0 — The Test Floor\n\nSee [the epilogue](../../EPILOGUE.md).\n")
    (root / "EPILOGUE.md").write_text("# Epilogue\n")
    (floor / "rooms" / "room_1_answer.py").write_text(STUB)
    (floor / "solutions" / "room_1_answer.py").write_text(SOLVED)
    (floor / "rooms" / "boss_roarer.py").write_text(BOSS_STUB)
    (floor / "solutions" / "boss_roarer.py").write_text(BOSS_SOLVED)
    (floor / "trials" / "test_room_1.py").write_text(
        textwrap.dedent(
            """
            from dungeon.trials import load_room

            room = load_room(__file__, "room_1_answer")


            def test_the_answer():
                assert room.answer() == 42


            def test_the_answer_is_an_int():
                assert isinstance(room.answer(), int)
            """
        )
    )
    (floor / "trials" / "test_boss.py").write_text(
        textwrap.dedent(
            """
            import pytest

            from dungeon.trials import load_room

            pytestmark = pytest.mark.boss
            room = load_room(__file__, "boss_roarer")


            def test_phase_1_it_roars():
                assert room.roar() == "ROAR"
            """
        )
    )
    (floor / "hints" / "room_1.md").write_text("Think of a number.\n---\nThe number is 42.\n")
    (floor / "loot" / "sheet.md").write_text("# Sheet\n")
    return root


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("DUNGEON_SOLUTIONS", raising=False)
    monkeypatch.delenv("DUNGEON_NO_RECORD", raising=False)
    root = make_dungeon(tmp_path)
    app = create_app(root)
    with TestClient(app) as c:
        c.root = root  # type: ignore[attr-defined]
        yield c
    app.state.watcher.stop()


def wait_for_run(client: TestClient, run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["done"]:
            return run
        time.sleep(0.1)
    raise AssertionError("run did not finish")


# ---------------------------------------------------------------------- state


def test_state_describes_floors_rooms_and_summary(client):
    state = client.get("/api/state").json()
    assert [f["number"] for f in state["floors"]] == [0]
    floor = state["floors"][0]
    assert [r["label"] for r in floor["rooms"]] == ["0.1", "BOSS"]
    assert floor["rooms"][0]["state"] == "unexplored"
    assert floor["rooms"][0]["hints"] == {"used": 0, "total": 2}
    assert floor["loot"][0]["unlocked"] is False
    assert state["summary"]["rank"] == "Wanderer at the Threshold"
    assert state["ranks"][-1]["title"] == "Dungeon Master"
    assert state["active_run"] is None


def test_doctor_reports_checks(client):
    checks = client.get("/api/doctor").json()["checks"]
    assert {c["label"] for c in checks} >= {"python", "numpy", "pytest", "floors", "save file"}


# ---------------------------------------------------------------------- files


def test_file_endpoint_reads_inside_the_dungeon_only(client):
    ok = client.get("/api/file", params={"path": "floors/floor_00_test/README.md"})
    assert ok.status_code == 200 and ok.json()["content"].startswith("# Floor 0")
    assert client.get("/api/file", params={"path": "EPILOGUE.md"}).status_code == 200
    for bad in ["/etc/passwd", "../x", "floors/../pyproject.toml", "conftest.py", ".dungeon/progress.json"]:
        assert client.get("/api/file", params={"path": bad}).status_code in (403, 404), bad
    assert client.get("/api/file", params={"path": "floors/floor_00_test/nope.md"}).status_code == 404


# ---------------------------------------------------------------------- hints


def test_hints_reveal_one_at_a_time_and_are_remembered(client):
    assert client.get("/api/floors/0/rooms/room_1/hints").json() == {"total": 2, "level": 0, "hints": []}
    first = client.post("/api/floors/0/rooms/room_1/hints/reveal").json()
    assert first["level"] == 1 and first["hints"] == ["Think of a number."]
    client.post("/api/floors/0/rooms/room_1/hints/reveal")
    capped = client.post("/api/floors/0/rooms/room_1/hints/reveal").json()
    assert capped["level"] == 2 and len(capped["hints"]) == 2
    assert client.get("/api/state").json()["floors"][0]["rooms"][0]["hints"]["used"] == 2
    assert client.get("/api/state").json()["summary"]["hints_used"] == 2
    assert client.get("/api/floors/0/rooms/room_9/hints").status_code == 404
    assert client.get("/api/floors/7/rooms/room_1/hints").status_code == 404


# ----------------------------------------------------------------------- runs


def test_run_request_validation(client):
    assert client.post("/api/runs", json={"floor": "0", "scope": "nope"}).status_code == 422
    assert client.post("/api/runs", json={"floor": "0", "scope": "room"}).status_code == 422
    assert client.post("/api/runs", json={"floor": "0", "scope": "rooms", "keyword": "x; rm -rf /"}).status_code == 422
    assert client.post("/api/runs", json={"floor": "0", "scope": "room", "room": "room_9"}).status_code == 404
    assert client.post("/api/runs", json={"floor": "0", "scope": "secret"}).status_code == 404
    assert client.get("/api/runs/nope").status_code == 404


def test_the_full_journey_stub_then_solved_then_boss(client):
    root: Path = client.root
    room_file = root / "floors/floor_00_test/rooms/room_1_answer.py"

    # 1. against the stub: recorded as unwritten, nothing cleared
    started = client.post("/api/runs", json={"floor": "0", "scope": "room", "room": "room_1"})
    assert started.status_code == 201
    run = wait_for_run(client, started.json()["id"])
    assert run["returncode"] != 0 and run["newly_cleared"] == [] and run["floors_cleared"] == []
    assert "UNWRITTEN" in run["output"] and "DUNGEON TRIALS" in run["output"]
    assert "\x1b[" in run["output"], "output keeps its colours for the browser"
    room = client.get("/api/state").json()["floors"][0]["rooms"][0]
    assert room["state"] == "unwritten" and room["result"]["attempts"] == 1

    # 2. a -k subset passes but never clears
    room_file.write_text(SOLVED)
    started = client.post("/api/runs", json={"floor": "0", "scope": "room", "room": "room_1", "keyword": "is_an_int"})
    run = wait_for_run(client, started.json()["id"])
    assert run["returncode"] == 0 and run["newly_cleared"] == []
    room = client.get("/api/state").json()["floors"][0]["rooms"][0]
    assert room["state"] == "partial"

    # 3. the whole file passes: cleared, streamed events end with `done`
    started = client.post("/api/runs", json={"floor": "0", "scope": "room", "room": "room_1"})
    events = []
    with client.stream("GET", f"/api/runs/{started.json()['id']}/stream") as resp:
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((event, json.loads(line[5:])))
                if event == "done":
                    break
    assert events[0][0] == "start" and events[-1][0] == "done"
    done = events[-1][1]
    assert done["newly_cleared"] == ["floor_00/test_room_1"] and done["floors_cleared"] == []
    output = "".join(d["text"] for e, d in events if e == "output")
    assert "cleared (2/2)" in output
    state = client.get("/api/state").json()
    assert state["floors"][0]["rooms"][0]["state"] == "cleared"
    assert state["floors"][0]["status"]["cleared"] is False

    # 4. the boss falls: the floor is cleared and the loot unlocks
    (root / "floors/floor_00_test/rooms/boss_roarer.py").write_text(BOSS_SOLVED)
    started = client.post("/api/runs", json={"floor": "0", "scope": "boss"})
    run = wait_for_run(client, started.json()["id"])
    assert run["title"].startswith("BOSS FIGHT") and run["newly_cleared"] == ["floor_00/test_boss"]
    assert run["floors_cleared"] == ["floor_00"]
    state = client.get("/api/state").json()
    assert state["floors"][0]["status"]["cleared"] is True
    assert state["floors"][0]["loot"][0]["unlocked"] is True
    assert state["summary"]["floors_cleared"] == 1 and state["summary"]["fraction"] == 1.0

    # 5. forgetting the floor puts everything back
    assert client.post("/api/reset", json={"floor": "0"}).json()["removed"] >= 2
    assert client.get("/api/state").json()["floors"][0]["rooms"][0]["state"] == "unexplored"


def test_events_stream_greets_reports_changes_and_stops_when_the_tab_closes(client):
    root: Path = client.root
    watcher = client.app.state.watcher

    class Tab:
        def __init__(self, open_for: int) -> None:
            self.polls = 0
            self.open_for = open_for

        async def is_disconnected(self) -> bool:
            self.polls += 1
            return self.polls > self.open_for

    async def collect(tab: Tab, poke: bool) -> list[str]:
        out = []
        async for chunk in _events(watcher, tab):  # type: ignore[arg-type]
            out.append(chunk)
            if poke:
                (root / "floors/floor_00_test/rooms/room_1_answer.py").write_text(SOLVED)
                poke = False
                await asyncio.sleep(1.5)  # the watcher polls once a second
        return out

    chunks = asyncio.run(collect(Tab(open_for=0), poke=False))
    assert chunks == [f'event: hello\ndata: {{"version": {watcher.version}}}\n\n'] or chunks[0].startswith("event: hello")
    chunks = asyncio.run(collect(Tab(open_for=3), poke=True))
    assert chunks[0].startswith("event: hello") and any(c.startswith("event: changed") for c in chunks)


# --------------------------------------------------------------------- static


def test_the_built_page_is_served_for_every_route(client):
    for path in ["/", "/floors/3", "/floors/0/rooms/room_1", "/loot"]:
        res = client.get(path)
        assert res.status_code == 200, path
        assert 'id="root"' in res.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/nope.js").status_code == 404
