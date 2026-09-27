import { useEffect, useState } from "react";

import { api } from "../api";
import { Empty } from "../components/bits";
import { useStore } from "../store";
import type { DoctorCheck } from "../types";

export function StatusPage() {
  const { state, refresh, notify } = useStore();
  const [checks, setChecks] = useState<DoctorCheck[] | null>(null);
  const [resetFloor, setResetFloor] = useState<string>("");
  useEffect(() => {
    api.doctor().then((d) => setChecks(d.checks)).catch(() => setChecks([]));
  }, []);
  if (!state) return null;
  const s = state.summary;
  const pct = Math.round(s.fraction * 100);

  const doReset = async (floor?: string) => {
    const what = floor ? `floor ${floor}` : "the ENTIRE dungeon";
    if (!window.confirm(`Forget all progress for ${what}? This cannot be undone.`)) return;
    const res = await api.reset(floor);
    notify(`Forgotten: ${res.removed} record(s). The dungeon does not remember you. Yet.`);
    await refresh();
  };

  return (
    <div className="page">
      <div className="kicker">STATUS</div>
      <h1>{s.rank}</h1>
      <div className="meter meter-lg">
        <div className="meter-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="g-dim">
        {pct}% of the required rooms and bosses. {s.fraction >= 1 ? "You have cleared the dungeon. Go outside; the surface has missed you." : ""}
      </p>

      <div className="stat-grid">
        <Stat label="Floors cleared" value={`${s.floors_cleared}/${s.floors_total}`} />
        <Stat label="Rooms cleared" value={`${s.rooms_done}/${s.rooms_total}`} />
        <Stat label="Bosses defeated" value={`${s.bosses_done}/${s.bosses_total}`} />
        <Stat label="Secrets found" value={`${s.secrets_done}/${s.secrets_total}`} />
        <Stat label="Trial attempts" value={String(s.attempts)} />
        <Stat label="Hints used" value={String(s.hints_used)} />
      </div>

      <section className="panel">
        <h3>Ranks</h3>
        <ol className="ranks">
          {state.ranks.map((r) => (
            <li key={r.title} className={s.fraction >= r.threshold ? "reached" : ""}>
              <span className="rank-pct">{Math.round(r.threshold * 100)}%</span> {r.title}
            </li>
          ))}
        </ol>
      </section>

      <section className="panel">
        <h3>Doctor</h3>
        {!checks && <Empty>Examining…</Empty>}
        {checks && (
          <dl className="doctor">
            {checks.map((c) => (
              <div key={c.label} className={`doctor-row level-${c.level}`}>
                <dt>
                  <span className="doctor-mark">{c.level === "ok" ? "ok" : c.level === "warn" ? "--" : "!!"}</span> {c.label}
                </dt>
                <dd>{c.detail}</dd>
              </div>
            ))}
          </dl>
        )}
        <p className="g-dim small">
          Save file: <code>{state.root}/.dungeon/progress.json</code> (git-ignored; it is yours).
        </p>
      </section>

      <section className="panel panel-danger">
        <h3>Forget</h3>
        <div className="actions">
          <select className="input" value={resetFloor} onChange={(e) => setResetFloor(e.target.value)}>
            <option value="">choose a floor…</option>
            {state.floors.map((f) => (
              <option key={f.id} value={String(f.number)}>
                Floor {f.number} — {f.name}
              </option>
            ))}
          </select>
          <button className="btn" disabled={!resetFloor} onClick={() => doReset(resetFloor)}>
            Forget this floor
          </button>
          <button className="btn btn-danger" onClick={() => doReset()}>
            Forget everything
          </button>
        </div>
        <p className="g-dim small">Same as <code>dungeon reset</code>. Your room files are never touched, only the save file.</p>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
