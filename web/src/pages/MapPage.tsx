import { Link } from "react-router-dom";

import { Ascii, Badge, Cells, GLYPH } from "../components/bits";
import { useStore } from "../store";

const BANNER = String.raw`
 _   _                      _   ____
| \ | | ___ _   _ _ __ __ _| | |  _ \ _   _ _ __   __ _  ___  ___  _ __
|  \| |/ _ \ | | | '__/ _\` | | | | | | | | | '_ \ / _\` |/ _ \/ _ \| '_ \
| |\  |  __/ |_| | | | (_| | | | |_| | |_| | | | | (_| |  __/ (_) | | | |
|_| \_|\___|\__,_|_|  \__,_|_| |____/ \__,_|_| |_|\__, |\___|\___/|_| |_|
                                                   |___/
`.replace(/\\`/g, "`").replace(/^\n/, "");

export function MapPage() {
  const { state } = useStore();
  if (!state) return null;
  const floors = state.floors;
  const here = floors.find((f) => !f.status.cleared);
  const allCleared = !here;

  return (
    <div className="page">
      <Ascii art={BANNER} className="banner" />
      <p className="lede">
        Fourteen floors from shapes to production. Every room is code you write; every verdict comes from a trial. Go down in
        order the first time.
      </p>

      <ol className="descent">
        <li className="descent-surface">
          <span className="g-yellow">{GLYPH.surface}</span> the surface — you came in this way
        </li>
        {floors.map((f) => {
          const isHere = here?.id === f.id;
          const tone = f.status.cleared ? "cleared" : f.status.started ? "started" : "unexplored";
          return (
            <li key={f.id} className={`descent-floor ${tone} ${isHere ? "here" : ""}`}>
              <Link to={`/floors/${f.number}`} className="floor-card">
                <div className="floor-card-head">
                  <span className="floor-num">F{String(f.number).padStart(2, "0")}</span>
                  <span className="floor-name">{f.name}</span>
                  <span className="floor-state">
                    {f.status.cleared ? (
                      <Badge tone="green">CLEARED {GLYPH.cleared}</Badge>
                    ) : f.status.started ? (
                      <Badge tone="yellow">
                        {f.status.required_done}/{f.status.required_total}
                      </Badge>
                    ) : (
                      <Badge>unexplored</Badge>
                    )}
                    {f.missing.length > 0 && <Badge tone="orange">needs {f.missing.join(", ")}</Badge>}
                    {isHere && <span className="here-marker">← you are here</span>}
                  </span>
                </div>
                <p className="floor-tagline">{f.tagline}</p>
                <div className="floor-card-foot">
                  <Cells floor={f} />
                  <span className="topics">
                    {f.topics.slice(0, 5).map((t) => (
                      <span key={t} className="chip">
                        {t}
                      </span>
                    ))}
                    {f.topics.length > 5 && <span className="chip chip-more">+{f.topics.length - 5}</span>}
                  </span>
                </div>
              </Link>
            </li>
          );
        })}
        <li className="descent-bottom">
          {allCleared ? (
            <Link to="/files/EPILOGUE.md" className="btn btn-primary">
              Every floor is cleared. Read the epilogue.
            </Link>
          ) : (
            <span className="g-dim">■ the bottom. Nothing below. Yet.</span>
          )}
        </li>
      </ol>

      <p className="legend">
        <span className="g-green">{GLYPH.roomDone}</span> room cleared · <span className="g-dim">{GLYPH.roomOpen}</span> room open ·{" "}
        <span className="g-red">{GLYPH.boss}</span> boss · <span className="g-dim">{GLYPH.secretHidden}</span> secret room ·{" "}
        <span className="g-green">{GLYPH.cleared}</span> floor cleared
      </p>
    </div>
  );
}
