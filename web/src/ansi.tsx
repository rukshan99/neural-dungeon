import { memo } from "react";

interface Span {
  text: string;
  classes: string;
}

const NAMES = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"];
const CSI = /\x1b\[([0-9;?]*)([A-Za-z])/g;

interface Style {
  bold: boolean;
  dim: boolean;
  italic: boolean;
  underline: boolean;
  fg: string | null;
  bg: string | null;
}

const fresh = (): Style => ({ bold: false, dim: false, italic: false, underline: false, fg: null, bg: null });

function apply(style: Style, params: string): Style {
  const codes = params === "" ? [0] : params.split(";").map((p) => parseInt(p || "0", 10));
  const next = { ...style };
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i];
    if (c === 0) Object.assign(next, fresh());
    else if (c === 1) next.bold = true;
    else if (c === 2) next.dim = true;
    else if (c === 3) next.italic = true;
    else if (c === 4) next.underline = true;
    else if (c === 22) (next.bold = false), (next.dim = false);
    else if (c === 23) next.italic = false;
    else if (c === 24) next.underline = false;
    else if (c >= 30 && c <= 37) next.fg = NAMES[c - 30];
    else if (c === 39) next.fg = null;
    else if (c >= 90 && c <= 97) next.fg = `bright-${NAMES[c - 90]}`;
    else if (c >= 40 && c <= 47) next.bg = NAMES[c - 40];
    else if (c === 49) next.bg = null;
    else if (c === 38 || c === 48) {
      // Extended colours: skip their arguments, keep the default colour.
      i += codes[i + 1] === 5 ? 2 : codes[i + 1] === 2 ? 4 : 0;
    }
  }
  return next;
}

function classesOf(s: Style): string {
  const out: string[] = [];
  if (s.bold) out.push("a-bold");
  if (s.dim) out.push("a-dim");
  if (s.italic) out.push("a-italic");
  if (s.underline) out.push("a-underline");
  if (s.fg) out.push(`a-fg-${s.fg}`);
  if (s.bg) out.push(`a-bg-${s.bg}`);
  return out.join(" ");
}

export function parseAnsi(text: string): Span[] {
  const spans: Span[] = [];
  let style = fresh();
  let last = 0;
  const clean = text.replace(/\r\n/g, "\n").replace(/\r/g, "");
  for (const m of clean.matchAll(CSI)) {
    const idx = m.index ?? 0;
    if (idx > last) spans.push({ text: clean.slice(last, idx), classes: classesOf(style) });
    if (m[2] === "m") style = apply(style, m[1]);
    last = idx + m[0].length;
  }
  if (last < clean.length) spans.push({ text: clean.slice(last), classes: classesOf(style) });
  return spans;
}

export const Ansi = memo(function Ansi({ text }: { text: string }) {
  const spans = parseAnsi(text);
  return (
    <>
      {spans.map((s, i) =>
        s.classes ? (
          <span key={i} className={s.classes}>
            {s.text}
          </span>
        ) : (
          <span key={i}>{s.text}</span>
        ),
      )}
    </>
  );
});

export function stripAnsi(text: string): string {
  return text.replace(CSI, "");
}
