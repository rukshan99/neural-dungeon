import { NavLink } from "react-router-dom";

import { useStore } from "../store";
import { Cells, GLYPH } from "./bits";

export function Sidebar() {
  const { state } = useStore();
  const floors = state?.floors ?? [];
  const summary = state?.summary;
  const here = floors.find((f) => !f.status.cleared);

  return (
    <aside className="sidebar">
      <NavLink to="/" className="brand">
        <span className="brand-glyph">{GLYPH.boss}</span>
        <span>
          <span className="brand-name">Neural Dungeon</span>
          <span className="brand-sub">a code-first descent</span>
        </span>
      </NavLink>

      {summary && (
        <div className="rank-card">
          <div className="rank-title">{summary.rank}</div>
          <div className="meter">
            <div className="meter-fill" style={{ width: `${Math.round(summary.fraction * 100)}%` }} />
          </div>
          <div className="rank-numbers">
            {summary.rooms_done + summary.bosses_done}/{summary.rooms_total + summary.bosses_total} required rooms ·{" "}
            {summary.floors_cleared}/{summary.floors_total} floors
          </div>
        </div>
      )}

      <nav className="nav">
        <NavLink to="/" end>
          Map
        </NavLink>
        <NavLink to="/loot">Loot</NavLink>
        <NavLink to="/status">Status</NavLink>
      </nav>

      <div className="side-floors">
        <div className="side-heading">
          <span className="g-yellow">{GLYPH.surface}</span> the surface
        </div>
        {floors.map((f) => (
          <NavLink
            key={f.id}
            to={`/floors/${f.number}`}
            className={({ isActive }) =>
              `side-floor ${isActive ? "active" : ""} ${f.status.cleared ? "cleared" : ""} ${here?.id === f.id ? "here" : ""}`
            }
          >
            <span className="side-num">F{String(f.number).padStart(2, "0")}</span>
            <span className="side-name">{f.name}</span>
            <Cells floor={f} />
          </NavLink>
        ))}
      </div>
      <div className="side-foot">v{state?.version ?? "…"} · 127.0.0.1 only</div>
    </aside>
  );
}
