import { useEffect, useState, type ReactNode } from "react";

import { api } from "../api";
import { useStore } from "../store";
import type { FileContent, Floor, Room, RoomState } from "../types";

export const GLYPH = {
  cleared: "★",
  boss: "☠",
  secretFound: "◆",
  secretHidden: "◇",
  roomDone: "■",
  roomOpen: "□",
  surface: "☀",
  cursed: "☿",
  unwritten: "✎",
  failed: "✗",
  locked: "🔒",
};

export function roomGlyph(room: Room): { glyph: string; cls: string } {
  if (room.kind === "boss") return { glyph: GLYPH.boss, cls: room.cleared ? "g-green" : "g-red" };
  if (room.kind === "secret")
    return { glyph: room.cleared ? GLYPH.secretFound : GLYPH.secretHidden, cls: room.cleared ? "g-green" : "g-dim" };
  return { glyph: room.cleared ? GLYPH.roomDone : GLYPH.roomOpen, cls: room.cleared ? "g-green" : "g-dim" };
}

export function stateLabel(room: Room): string {
  const s: RoomState = room.state;
  if (s === "cleared") return room.kind === "boss" ? "defeated" : "cleared";
  if (s === "unwritten") return "unwritten";
  if (s === "partial") return "subset only";
  if (s === "failed") return room.mode === "cursed" ? "still cursed" : room.kind === "boss" ? "undefeated" : "failing";
  return "unexplored";
}

export function Cells({ floor }: { floor: Floor }) {
  return (
    <span className="cells" aria-label={`${floor.status.required_done} of ${floor.status.required_total} required rooms cleared`}>
      {floor.rooms
        .filter((r) => r.kind === "room")
        .map((r) => (
          <span key={r.id} className={r.cleared ? "g-green" : "g-dim"}>
            {r.cleared ? GLYPH.roomDone : GLYPH.roomOpen}
          </span>
        ))}
      {floor.rooms
        .filter((r) => r.kind === "boss")
        .map((r) => (
          <span key={r.id} className={`cell-boss ${r.cleared ? "g-green" : "g-red"}`}>
            {GLYPH.boss}
          </span>
        ))}
      {floor.rooms
        .filter((r) => r.kind === "secret")
        .map((r) => (
          <span key={r.id} className={r.cleared ? "g-green" : "g-dim"}>
            {r.cleared ? GLYPH.secretFound : GLYPH.secretHidden}
          </span>
        ))}
    </span>
  );
}

export function Badge({ tone = "dim", children }: { tone?: string; children: ReactNode }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function StateBadge({ room }: { room: Room }) {
  const tone =
    room.state === "cleared"
      ? "green"
      : room.state === "unwritten"
        ? "cyan"
        : room.state === "partial"
          ? "yellow"
          : room.state === "failed"
            ? room.mode === "cursed"
              ? "magenta"
              : "red"
            : "dim";
  return <Badge tone={tone}>{stateLabel(room)}</Badge>;
}

export function Ascii({ art, className = "" }: { art: string; className?: string }) {
  return <pre className={`ascii ${className}`.trim()}>{art}</pre>;
}

/** Fetches a repo file and refetches whenever the watcher says something changed. */
export function useRepoFile(path: string | null | undefined): { file: FileContent | null; error: string | null; loading: boolean } {
  const { fileVersion } = useStore();
  const [file, setFile] = useState<FileContent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    setLoading(true);
    api
      .file(path)
      .then((f) => {
        if (!cancelled) {
          setFile(f);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [path, fileVersion]);
  return { file, error, loading };
}

export function CopyButton({ text, label = "Copy path" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="btn btn-ghost"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        } catch {
          window.prompt("Copy this path:", text);
        }
      }}
    >
      {copied ? "Copied" : label}
    </button>
  );
}

export function EditorLink({ root, path }: { root: string; path: string }) {
  return (
    <a className="btn btn-ghost" href={`vscode://file/${root}/${path}`} title="Opens in VS Code if it is installed">
      Open in VS Code
    </a>
  );
}

export function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
