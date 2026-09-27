import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api";
import { Badge, Code, CopyButton, EditorLink, Empty, GLYPH, languageFor, roomGlyph, StateBadge, useRepoFile } from "../components";
import { Markdown } from "../components/Markdown";
import { useFloor, useStore } from "../store";
import type { Floor, Hints, Room } from "../types";

type FileTab = "room" | "trial" | "solution";

export function RoomPage() {
  const { num, roomId } = useParams();
  const floor = useFloor(num);
  const { state, startRun, running } = useStore();
  const [tab, setTab] = useState<FileTab>("room");
  const [spoiler, setSpoiler] = useState(false);
  const [keyword, setKeyword] = useState("");
  useEffect(() => {
    setTab("room");
    setSpoiler(false);
    setKeyword("");
  }, [roomId]);

  if (!state) return null;
  const room = floor?.rooms.find((r) => r.id === roomId);
  if (!floor || !room) return <Empty>No such room. The map knows the way.</Empty>;

  const g = roomGlyph(room);
  const blocked = floor.missing.length > 0;
  const path = tab === "room" ? room.file : tab === "trial" ? room.trial : room.solution;
  const showSolution = tab !== "solution" || spoiler;
  const cli = room.kind === "boss" ? `dungeon fight ${floor.number}` : room.kind === "secret" ? `dungeon trial ${floor.number} --secret` : `dungeon trial ${floor.number} ${room.id}`;
  const siblings = floor.rooms;
  const pos = siblings.findIndex((r) => r.id === room.id);
  const prevRoom = pos > 0 ? siblings[pos - 1] : null;
  const nextRoom = pos < siblings.length - 1 ? siblings[pos + 1] : null;

  return (
    <div className="page">
      <nav className="crumbs">
        <Link to="/">Map</Link> <span>/</span> <Link to={`/floors/${floor.number}?tab=rooms`}>Floor {floor.number}</Link> <span>/</span>{" "}
        <span>{room.label}</span>
      </nav>

      <header className={`room-head room-${room.kind}`}>
        <span className={`room-head-glyph ${g.cls}`}>{g.glyph}</span>
        <div className="room-head-main">
          <div className="kicker">
            {room.kind === "boss" ? "BOSS" : room.kind === "secret" ? "SECRET ROOM" : `ROOM ${room.label}`}
            {room.mode === "cursed" && (
              <>
                {" "}
                <Badge tone="magenta">{GLYPH.cursed} cursed code</Badge>
              </>
            )}
          </div>
          <h1>{room.name}</h1>
          <p className="tagline">{room.blurb}</p>
          {room.kind === "boss" && floor.lines.boss_intro && <p className="taunt">“{floor.lines.boss_intro}”</p>}
        </div>
        <div className="room-head-side">
          <StateBadge room={room} />
          <div className="floor-nav">
            {prevRoom && <Link to={`/floors/${floor.number}/rooms/${prevRoom.id}`}>← {prevRoom.label}</Link>}
            {nextRoom && <Link to={`/floors/${floor.number}/rooms/${nextRoom.id}`}>{nextRoom.label} →</Link>}
          </div>
        </div>
      </header>

      {blocked && (
        <div className="callout callout-orange">
          <strong>This floor needs {floor.missing.join(", ")}.</strong>
          <pre>{floor.install_hint}</pre>
        </div>
      )}

      <div className="actions">
        <button
          className={`btn ${room.kind === "boss" ? "btn-danger" : "btn-primary"}`}
          disabled={blocked || running}
          onClick={() => startRun({ floor: String(floor.number), scope: "room", room: room.id, keyword: keyword.trim() || undefined })}
        >
          {room.kind === "boss" ? `${GLYPH.boss} Fight` : room.mode === "cursed" ? "Test the curse" : "Run the trial"}
        </button>
        <input
          className="input"
          placeholder="-k filter (subsets never clear a room)"
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          spellCheck={false}
        />
        <EditorLink root={state.root} path={room.file} />
        <CopyButton text={`${state.root}/${room.file}`} />
        <code className="cli-hint">{cli}</code>
      </div>

      <div className="room-grid">
        <div className="room-col">
          <Result floor={floor} room={room} />
          <HintsPanel floor={floor} room={room} />
        </div>
        <div className="room-col room-col-wide">
          <div className="tabs tabs-sm" role="tablist">
            <button className={tab === "room" ? "active" : ""} onClick={() => setTab("room")}>
              Your code
            </button>
            <button className={tab === "trial" ? "active" : ""} onClick={() => setTab("trial")}>
              The trial
            </button>
            <button className={tab === "solution" ? "active" : ""} onClick={() => setTab("solution")}>
              Reference solution
            </button>
          </div>
          {showSolution ? (
            <FileView path={path} />
          ) : (
            <div className="spoiler">
              <p>
                The reference solution is here so the repository can test itself and so you can compare <em>after</em> an honest
                attempt. Hints first? There are {room.hints.total - room.hints.used} left.
              </p>
              <button className="btn" onClick={() => setSpoiler(true)}>
                I have tried. Show me.
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FileView({ path }: { path: string }) {
  const { file, error, loading } = useRepoFile(path);
  return (
    <div className="file-view">
      <div className="file-bar">
        <code>{path}</code>
        {loading && <span className="g-dim small">refreshing…</span>}
      </div>
      {error && <div className="callout callout-red">{error}</div>}
      {file && <Code code={file.content} language={languageFor(file.path)} lineNumbers />}
    </div>
  );
}

function Result({ floor, room }: { floor: Floor; room: Room }) {
  const r = room.result;
  if (!r || !r.attempts) {
    return (
      <section className="panel">
        <h3>Trial</h3>
        <p className="g-dim">
          Not attempted yet. Open <code>{room.file}</code>, replace every <code>raise NotImplementedError</code>
          {room.mode === "cursed" ? " and lift the curses" : ""}, then run the trial.
        </p>
      </section>
    );
  }
  const total = r.collected || r.passed + r.failed + r.errors;
  const hp = Math.max(0, total - r.passed);
  const line =
    room.cleared
      ? room.kind === "boss"
        ? floor.lines.boss_defeated
        : floor.lines.room_cleared
      : room.state === "unwritten"
        ? "Unwritten. The trial found stubs, not code."
        : room.state === "partial"
          ? "Only a subset ran (-k). Subsets never clear a room."
          : room.mode === "cursed"
          ? floor.lines.room_cursed
          : floor.lines.room_failed;
  return (
    <section className="panel">
      <h3>
        Trial{" "}
        <span className="g-dim small">
          {r.attempts} attempt{r.attempts === 1 ? "" : "s"}
        </span>
      </h3>
      {room.kind === "boss" ? (
        <div className="hp">
          <div className="hp-label">
            {room.cleared ? "Defeated" : `${hp} of ${total} HP remaining`}
          </div>
          <div className="hp-bar">
            {Array.from({ length: total }).map((_, i) => (
              <span key={i} className={`hp-cell ${i < r.passed ? "hp-down" : "hp-up"}`} />
            ))}
          </div>
        </div>
      ) : (
        <div className="score">
          <span className="score-big">
            {r.passed}/{total}
          </span>{" "}
          <span className="g-dim">{room.mode === "cursed" ? "curses lifted" : "trials passed"}</span>
        </div>
      )}
      <dl className="stats">
        <dt>passed</dt>
        <dd className="g-green">{r.passed}</dd>
        <dt>failed</dt>
        <dd className={r.failed ? "g-red" : ""}>{r.failed}</dd>
        <dt>errors</dt>
        <dd className={r.errors ? "g-red" : ""}>{r.errors}</dd>
        <dt>unwritten</dt>
        <dd className={r.unwritten ? "g-cyan" : ""}>{r.unwritten}</dd>
        {r.skipped > 0 && (
          <>
            <dt>skipped</dt>
            <dd>{r.skipped}</dd>
          </>
        )}
      </dl>
      {line && <p className={`verdict-line ${room.cleared ? "g-green" : "g-dim"}`}>{line}</p>}
      {r.last_run && <p className="g-dim small">last run {new Date(r.last_run).toLocaleString()}</p>}
    </section>
  );
}

function HintsPanel({ floor, room }: { floor: Floor; room: Room }) {
  const [hints, setHints] = useState<Hints | null>(null);
  const [busy, setBusy] = useState(false);
  const { refresh } = useStore();
  useEffect(() => {
    setHints(null);
    api.hints(floor.number, room.id).then(setHints).catch(() => setHints({ total: 0, level: 0, hints: [] }));
  }, [floor.number, room.id]);

  const reveal = async (all = false) => {
    setBusy(true);
    try {
      setHints(await api.revealHint(floor.number, room.id, all));
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  if (!hints) return null;
  if (hints.total === 0)
    return (
      <section className="panel">
        <h3>Hints</h3>
        <p className="g-dim">No hints were written for this room. The README is your hint.</p>
      </section>
    );
  return (
    <section className="panel">
      <h3>
        Hints <span className="g-dim small">{hints.level}/{hints.total} revealed</span>
      </h3>
      <ol className="hints">
        {hints.hints.map((h, i) => (
          <li key={i}>
            <Markdown source={h} base={`floors/${floor.id}/hints/x.md`} />
          </li>
        ))}
      </ol>
      {hints.level < hints.total ? (
        <div className="hint-actions">
          <button className="btn btn-sm" disabled={busy} onClick={() => reveal(false)}>
            Reveal hint {hints.level + 1}
          </button>
          <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => reveal(true)}>
            Reveal all
          </button>
          <span className="g-dim small">Hints are counted in your status. Nothing else.</span>
        </div>
      ) : (
        <p className="g-dim small">That was the last hint. The solution tab is there if you must, but try once more first.</p>
      )}
    </section>
  );
}
