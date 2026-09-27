import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { api, ApiError } from "./api";
import type { DungeonState, Floor, Room, RunRequest, RunSummary, RunView } from "./types";

export interface Celebration {
  floors: Floor[];
  bosses: { floor: Floor; room: Room }[];
  secrets: { floor: Floor; room: Room }[];
  rooms: { floor: Floor; room: Room }[];
}

export interface RunOptions {
  verbose: boolean;
  fail_fast: boolean;
}

interface Store {
  state: DungeonState | null;
  error: string | null;
  refresh: () => Promise<void>;
  fileVersion: number;

  run: RunView | null;
  running: boolean;
  startRun: (req: RunRequest) => Promise<void>;
  cancelRun: () => Promise<void>;
  clearRun: () => void;
  runOptions: RunOptions;
  setRunOptions: (o: RunOptions) => void;

  consoleOpen: boolean;
  setConsoleOpen: (open: boolean) => void;

  celebration: Celebration | null;
  dismissCelebration: () => void;

  notice: string | null;
  notify: (message: string | null) => void;
}

const StoreContext = createContext<Store | null>(null);

export function useStore(): Store {
  const store = useContext(StoreContext);
  if (!store) throw new Error("useStore outside <StoreProvider>");
  return store;
}

function findRoomByKey(floors: Floor[], key: string): { floor: Floor; room: Room } | null {
  for (const floor of floors) {
    for (const room of floor.rooms) {
      if (room.key === key) return { floor, room };
    }
  }
  return null;
}

function buildCelebration(state: DungeonState, done: RunSummary): Celebration | null {
  const floors = state.floors.filter((f) => done.floors_cleared.includes(f.id));
  const bosses: Celebration["bosses"] = [];
  const secrets: Celebration["secrets"] = [];
  const rooms: Celebration["rooms"] = [];
  for (const key of done.newly_cleared) {
    const hit = findRoomByKey(state.floors, key);
    if (!hit) continue;
    if (hit.room.kind === "boss") bosses.push(hit);
    else if (hit.room.kind === "secret") secrets.push(hit);
    else rooms.push(hit);
  }
  if (!floors.length && !bosses.length && !secrets.length && !rooms.length) return null;
  return { floors, bosses, secrets, rooms };
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<DungeonState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileVersion, setFileVersion] = useState(0);
  const [run, setRun] = useState<RunView | null>(null);
  const [runOptions, setRunOptions] = useState<RunOptions>({ verbose: false, fail_fast: false });
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [celebration, setCelebration] = useState<Celebration | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const streamRef = useRef<EventSource | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await api.state();
      setState(next);
      setError(null);
      return next;
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      return null;
    }
  }, []);

  const attach = useCallback(
    (summary: RunSummary) => {
      streamRef.current?.close();
      setRun({ ...summary, output: "" });
      setConsoleOpen(true);
      const source = new EventSource(`/api/runs/${summary.id}/stream`);
      streamRef.current = source;
      source.addEventListener("output", (ev) => {
        const { text } = JSON.parse((ev as MessageEvent).data) as { text: string };
        setRun((prev) => (prev && prev.id === summary.id ? { ...prev, output: prev.output + text } : prev));
      });
      source.addEventListener("done", async (ev) => {
        const done = JSON.parse((ev as MessageEvent).data) as RunSummary;
        source.close();
        if (streamRef.current === source) streamRef.current = null;
        setRun((prev) => (prev && prev.id === done.id ? { ...prev, ...done } : prev));
        const fresh = await api.state().catch(() => null);
        if (fresh) {
          setState(fresh);
          const party = buildCelebration(fresh, done);
          if (party) setCelebration(party);
        }
      });
      source.onerror = () => {
        // The server went away mid-run; poll the run once so the UI can settle.
        source.close();
        api
          .run(summary.id)
          .then((full) => setRun(full))
          .catch(() => undefined);
      };
    },
    [],
  );

  const startRun = useCallback(
    async (req: RunRequest) => {
      try {
        const summary = await api.startRun({ ...runOptions, ...req });
        attach(summary);
      } catch (exc) {
        if (exc instanceof ApiError && exc.status === 409) {
          setNotice(exc.message);
          setConsoleOpen(true);
        } else {
          setNotice(exc instanceof Error ? exc.message : String(exc));
        }
      }
    },
    [attach, runOptions],
  );

  const cancelRun = useCallback(async () => {
    if (run && !run.done) await api.cancelRun(run.id).catch(() => undefined);
  }, [run]);

  const clearRun = useCallback(() => {
    if (run?.done) setRun(null);
  }, [run]);

  useEffect(() => {
    let debounce: number | undefined;
    refresh().then((first) => {
      if (first?.active_run && !first.active_run.done) attach(first.active_run);
    });
    const events = new EventSource("/api/events");
    events.addEventListener("changed", () => {
      window.clearTimeout(debounce);
      debounce = window.setTimeout(() => {
        refresh();
        setFileVersion((v) => v + 1);
      }, 250);
    });
    events.onerror = () => setError("Lost the connection to `dungeon serve`. Is it still running?");
    events.onopen = () => setError(null);
    return () => {
      events.close();
      streamRef.current?.close();
      window.clearTimeout(debounce);
    };
  }, [refresh, attach]);

  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => setNotice(null), 6000);
    return () => window.clearTimeout(t);
  }, [notice]);

  const value = useMemo<Store>(
    () => ({
      state,
      error,
      refresh: async () => {
        await refresh();
      },
      fileVersion,
      run,
      running: !!run && !run.done,
      startRun,
      cancelRun,
      clearRun,
      runOptions,
      setRunOptions,
      consoleOpen,
      setConsoleOpen,
      celebration,
      dismissCelebration: () => setCelebration(null),
      notice,
      notify: setNotice,
    }),
    [state, error, refresh, fileVersion, run, startRun, cancelRun, clearRun, runOptions, consoleOpen, celebration, notice],
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useFloor(ref: string | undefined): Floor | undefined {
  const { state } = useStore();
  if (!state || ref === undefined) return undefined;
  const n = Number(ref);
  return state.floors.find((f) => f.number === n || f.id === ref || f.slug === ref);
}
