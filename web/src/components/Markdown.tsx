import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { useNavigate } from "react-router-dom";
import remarkGfm from "remark-gfm";

import { useStore } from "../store";
import { Code, languageFor } from "./Code";

/** Resolve a relative link found in a markdown file against that file's directory. */
export function resolvePath(base: string, href: string): string {
  const dir = base.includes("/") ? base.slice(0, base.lastIndexOf("/")) : "";
  const parts = (dir ? dir.split("/") : []).concat(href.split("/"));
  const out: string[] = [];
  for (const p of parts) {
    if (p === "" || p === ".") continue;
    if (p === "..") out.pop();
    else out.push(p);
  }
  return out.join("/");
}

export function Markdown({ source, base }: { source: string; base: string }) {
  const navigate = useNavigate();
  const { state } = useStore();

  const onLink = (href: string) => (ev: React.MouseEvent) => {
    if (/^(https?:|mailto:|#)/.test(href)) return;
    ev.preventDefault();
    const target = resolvePath(base, href.split("#")[0]);
    const floor = state?.floors.find((f) => f.readme === target);
    if (floor) navigate(`/floors/${floor.number}`);
    else if (target === "EPILOGUE.md" || target.endsWith(".md") || target.endsWith(".py") || target.endsWith(".toml"))
      navigate(`/files/${target}`);
    else navigate(`/files/${target}`);
  };

  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children }) => {
            const text = String(children ?? "");
            const match = /language-([\w-]+)/.exec(className ?? "");
            const isBlock = !!match || text.includes("\n");
            if (!isBlock) return <code className="inline">{text}</code>;
            const lang = match ? match[1] : "text";
            if (lang === "text" || lang === "" ) {
              return <pre className="code plain">{text.replace(/\n$/, "")}</pre>;
            }
            const language = lang === "py" ? "python" : lang === "sh" || lang === "console" ? "bash" : languageFor(`x.${lang}`) === "markup" ? (lang as never) : languageFor(`x.${lang}`);
            return <Code code={text} language={language} />;
          },
          a: ({ href, children }) => (
            <a href={href} onClick={href ? onLink(href) : undefined} target={/^https?:/.test(href ?? "") ? "_blank" : undefined} rel="noreferrer">
              {children as ReactNode}
            </a>
          ),
          table: ({ children }) => (
            <div className="table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
