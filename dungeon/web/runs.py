"""Run pytest for the browser and stream its output.

One run at a time. Output is read in raw chunks (not lines) so pytest's progress
dots trickle in as they do in a terminal. Verdicts are not parsed from the text:
the run snapshots the save file before and after and reports what changed.
"""

from __future__ import annotations

import codecs
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from dungeon import progress as prog
from dungeon.registry import Floor, Room

MAX_KEPT_RUNS = 20


def _snapshot(root: Path, floors: list[Floor]) -> tuple[set[str], set[str]]:
    data = prog.load(root)
    cleared_trials = {k for k, v in data["trials"].items() if v.get("cleared")}
    cleared_floors = {f.id for f in floors if prog.floor_status(f, data).cleared}
    return cleared_trials, cleared_floors


@dataclass
class Run:
    id: str
    title: str
    floor_id: str
    command: list[str]
    keys: list[str]
    started: float = field(default_factory=time.time)
    chunks: list[str] = field(default_factory=list)
    done: bool = False
    returncode: int | None = None
    cancelled: bool = False
    newly_cleared: list[str] = field(default_factory=list)
    floors_cleared: list[str] = field(default_factory=list)
    _proc: subprocess.Popen | None = None

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "floor": self.floor_id,
            "command": " ".join(self.command),
            "keys": self.keys,
            "started": self.started,
            "done": self.done,
            "returncode": self.returncode,
            "cancelled": self.cancelled,
            "newly_cleared": self.newly_cleared,
            "floors_cleared": self.floors_cleared,
        }


class RunManager:
    def __init__(self, root: Path, floors: list[Floor]) -> None:
        self.root = root
        self.floors = floors
        self.runs: dict[str, Run] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------ queries
    @property
    def active(self) -> Run | None:
        for rid in reversed(self.order):
            run = self.runs[rid]
            if not run.done:
                return run
        return None

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    # ------------------------------------------------------------ control
    def start(
        self,
        floor: Floor,
        rooms: list[Room],
        title: str,
        *,
        verbose: bool = False,
        fail_fast: bool = False,
        keyword: str | None = None,
    ) -> Run:
        with self.lock:
            if self.active is not None:
                raise RuntimeError("a trial is already running")
            paths = [str((floor.path / r.trial).relative_to(self.root)) for r in rooms]
            cmd = [sys.executable, "-m", "pytest", *paths, "--color=yes", "-p", "no:cacheprovider"]
            cmd.append("-v" if verbose else "-q")
            if fail_fast:
                cmd.append("-x")
            if keyword:
                cmd.extend(["-k", keyword])
            run = Run(
                id=uuid.uuid4().hex[:12],
                title=title,
                floor_id=floor.id,
                command=["pytest", *cmd[3:]],
                keys=[floor.trial_key(r) for r in rooms],
            )
            self.runs[run.id] = run
            self.order.append(run.id)
            while len(self.order) > MAX_KEPT_RUNS:
                old = self.order.pop(0)
                self.runs.pop(old, None)
        threading.Thread(target=self._execute, args=(run, cmd), daemon=True).start()
        return run

    def cancel(self, run: Run) -> None:
        proc = run._proc
        if proc is not None and proc.poll() is None:
            run.cancelled = True
            proc.terminate()

    # ------------------------------------------------------------ worker
    def _execute(self, run: Run, cmd: list[str]) -> None:
        env = dict(os.environ)
        env.update(FORCE_COLOR="1", PYTHONUNBUFFERED="1", COLUMNS="100")
        env.pop("NO_COLOR", None)
        before_trials, before_floors = _snapshot(self.root, self.floors)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            run.chunks.append(f"could not start pytest: {exc}\n")
            run.returncode = 1
            run.done = True
            return
        run._proc = proc
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                run.chunks.append(text)
        tail = decoder.decode(b"", final=True)
        if tail:
            run.chunks.append(tail)
        run.returncode = proc.wait()
        after_trials, after_floors = _snapshot(self.root, self.floors)
        run.newly_cleared = sorted(k for k in after_trials - before_trials if k in run.keys)
        run.floors_cleared = sorted(after_floors - before_floors)
        run.done = True
