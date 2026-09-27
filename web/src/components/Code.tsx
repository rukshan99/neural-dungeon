import { Highlight, type Language, type PrismTheme } from "prism-react-renderer";

const theme: PrismTheme = {
  plain: { color: "#e6dccb", backgroundColor: "transparent" },
  styles: [
    { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#7b7590", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#a9a1b8" } },
    { types: ["string", "char", "attr-value", "inserted"], style: { color: "#a6d189" } },
    { types: ["triple-quoted-string"], style: { color: "#a6d189", fontStyle: "italic" } },
    { types: ["number", "boolean", "constant", "symbol"], style: { color: "#f0a340" } },
    { types: ["keyword", "operator", "atrule"], style: { color: "#d58cff" } },
    { types: ["function", "class-name", "builtin", "decorator", "tag"], style: { color: "#7dd3fc" } },
    { types: ["variable", "attr-name", "property"], style: { color: "#e6dccb" } },
    { types: ["deleted"], style: { color: "#ff6b6b" } },
    { types: ["title", "important", "bold"], style: { fontWeight: "bold" } },
    { types: ["italic"], style: { fontStyle: "italic" } },
    { types: ["url", "link"], style: { color: "#7dd3fc", textDecorationLine: "underline" } },
  ],
};

export function languageFor(path: string): Language {
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".md")) return "markdown";
  if (path.endsWith(".json")) return "json";
  if (path.endsWith(".toml") || path.endsWith(".yml") || path.endsWith(".yaml")) return "yaml";
  if (path.endsWith(".sh")) return "bash";
  return "markup";
}

export function Code({
  code,
  language,
  lineNumbers = false,
  className = "",
}: {
  code: string;
  language: Language;
  lineNumbers?: boolean;
  className?: string;
}) {
  const source = code.replace(/\n$/, "");
  return (
    <Highlight code={source} language={language} theme={theme}>
      {({ className: cls, style, tokens, getLineProps, getTokenProps }) => (
        <pre className={`code ${cls} ${className}`.trim()} style={style}>
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })} className="code-line">
              {lineNumbers && <span className="code-ln">{i + 1}</span>}
              <span className="code-text">
                {line.map((token, k) => (
                  <span key={k} {...getTokenProps({ token })} />
                ))}
              </span>
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
}
