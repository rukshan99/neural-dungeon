"""ROOM 12.5 - THE GATE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A small HTTP server on the standard library. Three routes, JSON in and out,
newline-delimited JSON for the stream, 400 with a JSON body for bad requests.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch


@torch.no_grad()
def greedy_token_stream(model, idx: torch.Tensor, max_new_tokens: int) -> Iterator[int]:
    """Yield one greedy token id at a time (no cache). Stops at block_size."""
    n = min(max_new_tokens, model.cfg.block_size - idx.shape[1])
    for _ in range(n):
        logits, _ = model(idx)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        yield int(next_id[0, 0])


def validate_request(body: bytes, block_size: int) -> tuple[dict | None, str | None]:
    """Parse and check a request body. Returns (request, None) or (None, error message)."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"body is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "body must be a JSON object"
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return None, "'prompt' must be a non-empty string"
    if len(prompt) >= block_size:
        return None, f"'prompt' has {len(prompt)} characters; the limit is {block_size - 1}"
    n = data.get("max_new_tokens", 32)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return None, "'max_new_tokens' must be a positive integer"
    return {"prompt": prompt, "max_new_tokens": n}, None


class InferenceServer:
    """GET /health, POST /generate, POST /stream over stdlib http.server.

    ``token_stream(model, idx, max_new_tokens)`` yields token ids; the default is
    the no-cache greedy loop above. Plug in a cached one from room 1.
    """

    def __init__(self, model, tokenizer, host: str = "127.0.0.1", port: int = 0,
                 token_stream: Callable | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.token_stream = token_stream or greedy_token_stream
        self._lock = threading.Lock()  # one generation at a time keeps latency predictable
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), make_handler(self))
        self.port = self._httpd.server_address[1]  # port=0 asked the OS for a free one
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()  # returns once serve_forever has exited
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------ the work
    def health(self) -> dict:
        return {
            "status": "ok",
            "model": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "block_size": self.model.cfg.block_size,
        }

    def tokens_for(self, request: dict) -> Iterator[int]:
        idx = torch.tensor([self.tokenizer.encode(request["prompt"])], dtype=torch.long)
        with self._lock:
            yield from self.token_stream(self.model, idx, request["max_new_tokens"])

    def generate(self, request: dict) -> dict:
        t0 = time.perf_counter()
        ids = list(self.tokens_for(request))
        return {
            "text": self.tokenizer.decode(ids),
            "tokens": len(ids),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


def make_handler(server: InferenceServer) -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the trial output quiet
            pass

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length > 0 else b""

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, server.health())
            else:
                self._json(404, {"error": f"no such route: {self.path}"})

        def do_POST(self) -> None:
            if self.path not in ("/generate", "/stream"):
                self._json(404, {"error": f"no such route: {self.path}"})
                return
            request, error = validate_request(self._body(), server.model.cfg.block_size)
            if error is not None:
                self._json(400, {"error": error})
                return
            if self.path == "/generate":
                self._json(200, server.generate(request))
                return
            # /stream: newline-delimited JSON. HTTP/1.0 with no Content-Length means
            # "read until the connection closes", so each line can go out as it is made.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            for tok in server.tokens_for(request):
                line = json.dumps({"token": server.tokenizer.decode([tok])}) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b'{"done": true}\n')
            self.wfile.flush()

    return Handler
