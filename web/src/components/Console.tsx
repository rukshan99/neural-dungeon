import { useEffect, useRef } from "react";

import { Ansi } from "../ansi";
import { useStore } from "../store";
import { Spinner } from "./bits";

export function Console() {
  const { run, running, consoleOpen, setConsoleOpen, cancelRun, clearRun, runOptions, setRunOptions, notice } = useStore();
  const bodyRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = bodyRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [run?.output, consoleOpen]);

  const onScroll = () => {
    const el = bodyRef.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  const verdict = run?.done
    ? run.cancelled
      ? { tone: "yellow", text: "Cancelled. The trial waits." }
      : run.returncode === 0
        ? / -k /.test(run.command)
          ? { tone: "yellow", text: "Every selected trial passed. Subsets never clear a room; run without the filter." }
          : { tone: "green", text: "Every trial in this run passed." }
        : run.returncode === 5
          ? { tone: "yellow", text: "No tests ran. Check the keyword filter." }
          : { tone: "red", text: "The trial is not fooled. Read the failures above." }
    : null;

  return (
    <section className={`console ${consoleOpen ? "open" : ""}`} aria-label="Trial console">
      <header className="console-bar">
        <button className="console-toggle" onClick={() => setConsoleOpen(!consoleOpen)} aria-expanded={consoleOpen}>
          <span className="console-caret">{consoleOpen ? "▾" : "▴"}</span>
          <span className="console-title">
            {running && <Spinner />}
            {run ? run.title : "Trial console"}
          </span>
          {run && <code className="console-cmd">$ {run.command}</code>}
        </button>
        <div className="console-actions">
          <label className="check">
            <input type="checkbox" checked={runOptions.verbose} onChange={(e) => setRunOptions({ ...runOptions, verbose: e.target.checked })} />
            verbose
          </label>
          <label className="check">
            <input type="checkbox" checked={runOptions.fail_fast} onChange={(e) => setRunOptions({ ...runOptions, fail_fast: e.target.checked })} />
            stop at first failure
          </label>
          {running && (
            <button className="btn btn-danger btn-sm" onClick={cancelRun}>
              Cancel
            </button>
          )}
          {run?.done && (
            <button className="btn btn-ghost btn-sm" onClick={clearRun}>
              Clear
            </button>
          )}
        </div>
      </header>
      {notice && <div className="console-notice">{notice}</div>}
      {consoleOpen && (
        <div className="console-body" ref={bodyRef} onScroll={onScroll}>
          {run ? (
            <pre className="terminal">
              <Ansi text={run.output} />
              {running && <span className="cursor">▌</span>}
            </pre>
          ) : (
            <div className="console-empty">
              Run a trial from a floor or a room and its output streams here. The verdict at the bottom of every run is written by the
              dungeon's pytest plugin, exactly as in the terminal.
            </div>
          )}
          {verdict && <div className={`verdict verdict-${verdict.tone}`}>{verdict.text}</div>}
        </div>
      )}
    </section>
  );
}
