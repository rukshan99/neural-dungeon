import { Link } from "react-router-dom";

import { useStore } from "../store";
import { GLYPH } from "./bits";

export function Celebration() {
  const { celebration, dismissCelebration } = useStore();
  if (!celebration) return null;
  const { floors, bosses, secrets, rooms } = celebration;
  const big = floors.length > 0 || bosses.length > 0;

  return (
    <div className="overlay" onClick={dismissCelebration} role="dialog" aria-modal="true">
      <div className={`party ${big ? "party-big" : ""}`} onClick={(e) => e.stopPropagation()}>
        {floors.map((f) => (
          <div key={f.id} className="party-block party-floor">
            <div className="party-kicker">{GLYPH.cleared} FLOOR {f.number} CLEARED</div>
            <h2>{f.name}</h2>
            {f.lines.floor_cleared && <p className="party-line">{f.lines.floor_cleared}</p>}
            {f.loot.length > 0 && (
              <div className="party-loot">
                <div className="party-loot-title">Loot unlocked</div>
                {f.loot.map((l) => (
                  <Link key={l.file} to={`/files/${l.file}`} onClick={dismissCelebration} className="party-loot-item">
                    <span className="g-green">{GLYPH.secretFound}</span> {l.name}
                  </Link>
                ))}
              </div>
            )}
            {f.number + 1 < 14 ? (
              <Link to={`/floors/${f.number + 1}`} onClick={dismissCelebration} className="btn btn-primary">
                Take the stairs down
              </Link>
            ) : (
              <Link to="/files/EPILOGUE.md" onClick={dismissCelebration} className="btn btn-primary">
                Read the epilogue
              </Link>
            )}
          </div>
        ))}
        {bosses.map(({ floor, room }) => (
          <div key={room.key} className="party-block party-boss">
            <div className="party-kicker g-red">{GLYPH.boss} BOSS DEFEATED</div>
            <h2>{room.name}</h2>
            <p className="party-line">{floor.lines.boss_defeated ?? "Its weakness was exactly what you thought it was."}</p>
          </div>
        ))}
        {secrets.map(({ room }) => (
          <div key={room.key} className="party-block">
            <div className="party-kicker g-green">{GLYPH.secretFound} SECRET ROOM CLEARED</div>
            <h2>{room.name}</h2>
          </div>
        ))}
        {rooms.length > 0 && (
          <div className="party-block">
            <div className="party-kicker g-green">
              {GLYPH.cleared} {rooms.length === 1 ? "ROOM CLEARED" : `${rooms.length} ROOMS CLEARED`}
            </div>
            <ul className="party-rooms">
              {rooms.map(({ room }) => (
                <li key={room.key}>
                  <strong>
                    {room.label} {room.name}
                  </strong>
                </li>
              ))}
            </ul>
            {rooms[0].floor.lines.room_cleared && <p className="party-line">{rooms[0].floor.lines.room_cleared}</p>}
          </div>
        )}
        <button className="btn btn-ghost" onClick={dismissCelebration}>
          Onward
        </button>
      </div>
    </div>
  );
}
