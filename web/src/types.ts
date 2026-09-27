export type RoomKind = "room" | "boss" | "secret";
export type RoomMode = "build" | "cursed";
export type RoomState = "cleared" | "unexplored" | "unwritten" | "partial" | "failed";

export interface TrialEntry {
  passed: number;
  failed: number;
  skipped: number;
  errors: number;
  collected: number;
  deselected: number;
  unwritten: number;
  cleared: boolean;
  cleared_this_run?: boolean;
  attempts: number;
  first_cleared?: string;
  last_run?: string;
}

export interface Room {
  id: string;
  key: string;
  label: string;
  name: string;
  kind: RoomKind;
  mode: RoomMode;
  blurb: string;
  file: string;
  trial: string;
  solution: string;
  cleared: boolean;
  state: RoomState;
  result: TrialEntry | null;
  hints: { used: number; total: number };
}

export interface Loot {
  name: string;
  file: string;
  blurb: string;
  unlocked: boolean;
}

export interface FloorStatus {
  cleared: boolean;
  started: boolean;
  required_done: number;
  required_total: number;
  boss_defeated: boolean;
  secret_found: boolean;
}

export interface Floor {
  id: string;
  number: number;
  slug: string;
  name: string;
  tagline: string;
  topics: string[];
  requires: string[];
  missing: string[];
  install_hint: string | null;
  map: string;
  lines: Record<string, string>;
  readme: string;
  rooms: Room[];
  loot: Loot[];
  status: FloorStatus;
}

export interface Summary {
  floors_total: number;
  floors_cleared: number;
  rooms_total: number;
  rooms_done: number;
  bosses_total: number;
  bosses_done: number;
  secrets_total: number;
  secrets_done: number;
  attempts: number;
  hints_used: number;
  fraction: number;
  rank: string;
}

export interface RunSummary {
  id: string;
  title: string;
  floor: string;
  command: string;
  keys: string[];
  started: number;
  done: boolean;
  returncode: number | null;
  cancelled: boolean;
  newly_cleared: string[];
  floors_cleared: string[];
}

export interface RunView extends RunSummary {
  output: string;
}

export interface DungeonState {
  version: string;
  root: string;
  floors: Floor[];
  summary: Summary;
  ranks: { threshold: number; title: string }[];
  active_run: RunSummary | null;
}

export interface Hints {
  total: number;
  level: number;
  hints: string[];
}

export interface DoctorCheck {
  label: string;
  ok: boolean;
  detail: string;
  level: "ok" | "warn" | "error";
}

export type RunScope = "rooms" | "room" | "boss" | "secret" | "all";

export interface RunRequest {
  floor: string;
  scope: RunScope;
  room?: string;
  verbose?: boolean;
  fail_fast?: boolean;
  keyword?: string;
}

export interface FileContent {
  path: string;
  content: string;
  suffix: string;
}
