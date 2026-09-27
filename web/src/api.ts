import type {
  DoctorCheck,
  DungeonState,
  FileContent,
  Hints,
  RunRequest,
  RunSummary,
  RunView,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  state: () => request<DungeonState>("/api/state"),
  doctor: () => request<{ checks: DoctorCheck[] }>("/api/doctor"),
  file: (path: string) => request<FileContent>(`/api/file?path=${encodeURIComponent(path)}`),
  hints: (floor: string | number, room: string) =>
    request<Hints>(`/api/floors/${floor}/rooms/${room}/hints`),
  revealHint: (floor: string | number, room: string, all = false) =>
    request<Hints>(`/api/floors/${floor}/rooms/${room}/hints/reveal?all=${all}`, { method: "POST" }),
  startRun: (req: RunRequest) =>
    request<RunSummary>("/api/runs", { method: "POST", body: JSON.stringify(req) }),
  run: (id: string) => request<RunView>(`/api/runs/${id}`),
  cancelRun: (id: string) => request<RunSummary>(`/api/runs/${id}/cancel`, { method: "POST" }),
  reset: (floor?: string | number) =>
    request<{ removed: number }>("/api/reset", {
      method: "POST",
      body: JSON.stringify({ floor: floor === undefined ? null : String(floor) }),
    }),
};
