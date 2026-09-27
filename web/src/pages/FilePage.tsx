import { Link, useParams } from "react-router-dom";

import { Code, CopyButton, EditorLink, Empty, languageFor, useRepoFile } from "../components";
import { Markdown } from "../components/Markdown";
import { useStore } from "../store";

export function FilePage() {
  const params = useParams();
  const path = params["*"] ?? "";
  const { state } = useStore();
  const { file, error } = useRepoFile(path || null);
  const floor = state?.floors.find((f) => path.startsWith(f.readme.slice(0, f.readme.lastIndexOf("/") + 1)));

  return (
    <div className="page">
      <nav className="crumbs">
        <Link to="/">Map</Link> <span>/</span>
        {floor && (
          <>
            <Link to={`/floors/${floor.number}`}>Floor {floor.number}</Link> <span>/</span>
          </>
        )}
        <code>{path}</code>
      </nav>
      {state && path && (
        <div className="actions">
          <EditorLink root={state.root} path={path} />
          <CopyButton text={`${state.root}/${path}`} />
        </div>
      )}
      {error && <Empty>{error}</Empty>}
      {file &&
        (file.suffix === ".md" ? (
          <Markdown source={file.content} base={file.path} />
        ) : (
          <Code code={file.content} language={languageFor(file.path)} lineNumbers />
        ))}
    </div>
  );
}
