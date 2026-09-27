import { Link, useParams, useSearchParams } from "react-router-dom";

import { Ascii, Badge, Empty, GLYPH, roomGlyph, StateBadge, useRepoFile } from "../components/bits";
import { Markdown } from "../components/Markdown";
import { useFloor, useStore } from "../store";
import type { Floor, Room } from "../types";

type Tab = "rooms" | "lesson" | "loot";

export function FloorPage() {
  const { num } = useParams();
  const floor = useFloor(num);
  const { state, startRun, running } = useStore();
  const [params, setParams] = useSearchParams();
  if (!state) return null;
  if (!floor) return <Empty>No floor {num}. The map knows the way.</Empty>;

  const tab = ((params.get("tab") as Tab) || (floor.status.started ? "rooms" : "lesson")) as Tab;
  const setTab = (t: Tab) => setParams({ tab: t }, { replace: true });
  const blocked = floor.missing.length > 0;
  const regular = floor.rooms.filter((r) => r.kind === "room");
  const boss = floor.rooms.find((r) => r.kind === "boss");
  const secret = floor.rooms.find((r) => r.kind === "secret");
  const openRooms = regular.filter((r) => !r.cleared).length;
  const next = state.floors.find((f) => f.number === floor.number + 1);
  const prev = state.floors.find((f) => f.number === floor.number - 1);

  return (
    <div className="page">
      <nav className="crumbs">
        <Link to="/">Map</Link> <span>/</span> <span>Floor {floor.number}</span>
      </nav>
      <header className="floor-head">
        <div className="floor-head-main">
          <div className="kicker">FLOOR {floor.number}</div>
          <h1>{floor.name}</h1>
          <p className="tagline">{floor.tagline}</p>
          <div className="topics">
            {floor.topics.map((t) => (
              <span key={t} className="chip">
                {t}
              </span>
            ))}
          </div>
        </div>
        <div className="floor-head-side">
          {floor.status.cleared ? (
            <Badge tone="green">CLEARED {GLYPH.cleared}</Badge>
          ) : (
            <Badge tone={floor.status.started ? "yellow" : "dim"}>
              {floor.status.required_done}/{floor.status.required_total} required
            </Badge>
          )}
          <div className="floor-nav">
            {prev && <Link to={`/floors/${prev.number}`}>↑ F{String(prev.number).padStart(2, "0")}</Link>}
            {next && <Link to={`/floors/${next.number}`}>F{String(next.number).padStart(2, "0")} ↓</Link>}
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
        <button className="btn btn-primary" disabled={blocked || running} onClick={() => startRun({ floor: String(floor.number), scope: "rooms" })}>
          Run every room trial
        </button>
        {boss && (
          <button
            className="btn btn-danger"
            disabled={blocked || running}
            onClick={() => startRun({ floor: String(floor.number), scope: "boss" })}
            title={openRooms ? `${openRooms} room(s) still open. The boss assumes you have their habits.` : undefined}
          >
            {GLYPH.boss} Fight the boss
          </button>
        )}
        {secret && (
          <button className="btn" disabled={blocked || running} onClick={() => startRun({ floor: String(floor.number), scope: "secret" })}>
            {GLYPH.secretHidden} Attempt the secret room
          </button>
        )}
        <code className="cli-hint">dungeon enter {floor.number}</code>
      </div>

      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "lesson"} className={tab === "lesson" ? "active" : ""} onClick={() => setTab("lesson")}>
          Lesson
        </button>
        <button role="tab" aria-selected={tab === "rooms"} className={tab === "rooms" ? "active" : ""} onClick={() => setTab("rooms")}>
          Rooms &amp; boss
        </button>
        <button role="tab" aria-selected={tab === "loot"} className={tab === "loot" ? "active" : ""} onClick={() => setTab("loot")}>
          Loot {floor.loot.length ? `(${floor.loot.length})` : ""}
        </button>
      </div>

      {tab === "lesson" && <Lesson floor={floor} onRooms={() => setTab("rooms")} />}
      {tab === "rooms" && <Rooms floor={floor} regular={regular} boss={boss} secret={secret} />}
      {tab === "loot" && <LootTab floor={floor} />}
    </div>
  );
}

function Lesson({ floor, onRooms }: { floor: Floor; onRooms: () => void }) {
  const { file, error } = useRepoFile(floor.readme);
  return (
    <section className="lesson">
      {error && <div className="callout callout-red">{error}</div>}
      {file ? <Markdown source={file.content} base={file.path} /> : <Empty>Unrolling the scroll…</Empty>}
      <div className="lesson-foot">
        <button className="btn btn-primary" onClick={onRooms}>
          To the rooms →
        </button>
      </div>
    </section>
  );
}

function Rooms({ floor, regular, boss, secret }: { floor: Floor; regular: Room[]; boss?: Room; secret?: Room }) {
  return (
    <section className="rooms-layout">
      {floor.map && <Ascii art={floor.map} className="floor-map" />}
      <div className="room-list">
        {regular.map((r) => (
          <RoomCard key={r.id} floor={floor} room={r} />
        ))}
        {boss && <RoomCard floor={floor} room={boss} />}
        {secret && <RoomCard floor={floor} room={secret} />}
      </div>
    </section>
  );
}

function RoomCard({ floor, room }: { floor: Floor; room: Room }) {
  const g = roomGlyph(room);
  const r = room.result;
  return (
    <Link to={`/floors/${floor.number}/rooms/${room.id}`} className={`room-card room-${room.kind} state-${room.state}`}>
      <span className={`room-glyph ${g.cls}`}>{g.glyph}</span>
      <span className="room-main">
        <span className="room-title">
          <span className="room-label">{room.label}</span> {room.name}
          {room.mode === "cursed" && <Badge tone="magenta">{GLYPH.cursed} cursed code</Badge>}
        </span>
        <span className="room-blurb">{room.blurb}</span>
        <code className="room-file">{room.file}</code>
      </span>
      <span className="room-side">
        <StateBadge room={room} />
        {r && r.collected > 0 && (
          <span className="room-score">
            {r.passed}/{r.collected}
          </span>
        )}
        {room.hints.used > 0 && (
          <span className="room-hints">
            {room.hints.used}/{room.hints.total} hints
          </span>
        )}
      </span>
    </Link>
  );
}

function LootTab({ floor }: { floor: Floor }) {
  const boss = floor.rooms.find((r) => r.kind === "boss");
  if (!floor.loot.length) return <Empty>Nothing here yet.</Empty>;
  return (
    <section className="loot-list">
      {floor.loot.map((l) => (
        <div key={l.file} className={`loot-card ${l.unlocked ? "unlocked" : "locked"}`}>
          <span className="loot-glyph">{l.unlocked ? <span className="g-green">{GLYPH.secretFound}</span> : GLYPH.locked}</span>
          <div className="loot-main">
            <div className="loot-name">{l.name}</div>
            <p className="loot-blurb">{l.blurb}</p>
            <code className="room-file">{l.file}</code>
          </div>
          <div className="loot-side">
            {l.unlocked ? (
              <Link to={`/files/${l.file}`} className="btn btn-sm">
                Open
              </Link>
            ) : (
              <span className="g-dim">locked — defeat {boss?.name ?? "the boss"}</span>
            )}
          </div>
        </div>
      ))}
      <p className="g-dim small">Loot files are ordinary files in the repo. Locks are on the honour system.</p>
    </section>
  );
}
