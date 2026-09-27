import { Link } from "react-router-dom";

import { Empty, GLYPH } from "../components/bits";
import { useStore } from "../store";

export function LootPage() {
  const { state } = useStore();
  if (!state) return null;
  const floors = state.floors.filter((f) => f.loot.length);
  const unlocked = floors.reduce((n, f) => n + f.loot.filter((l) => l.unlocked).length, 0);
  const total = floors.reduce((n, f) => n + f.loot.length, 0);

  return (
    <div className="page">
      <div className="kicker">LOOT</div>
      <h1>
        {unlocked}/{total} unlocked
      </h1>
      <p className="lede">Cheat sheets, reference implementations and tools. Each floor's loot unlocks when its boss falls.</p>
      {!floors.length && <Empty>Nothing here yet.</Empty>}
      {floors.map((f) => {
        const boss = f.rooms.find((r) => r.kind === "boss");
        return (
          <section key={f.id} className="loot-floor">
            <h2>
              <Link to={`/floors/${f.number}`}>
                Floor {f.number} — {f.name}
              </Link>
            </h2>
            <div className="loot-list">
              {f.loot.map((l) => (
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
                      <span className="g-dim small">defeat {boss?.name ?? "the boss"}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>
        );
      })}
      <p className="g-dim small">Loot files are ordinary files in the repo. Locks are on the honour system.</p>
    </div>
  );
}
