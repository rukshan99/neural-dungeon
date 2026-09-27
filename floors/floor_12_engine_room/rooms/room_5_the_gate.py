"""ROOM 12.5 - THE GATE

    At the end of the Engine Room, a door to the outside. Petitioners will
    knock on it who have never seen a tensor and never will. They will knock
    with JSON, and they will expect an answer before they lose interest.

A model becomes a service the moment something else can call it over the
network. This room builds that service on the standard library alone
(``http.server.ThreadingHTTPServer``), because the shape of the thing matters
more than the framework:

    GET  /health    -> 200 {"status": "ok", "model": ..., ...}
    POST /generate  <- {"prompt": str, "max_new_tokens": int}
                    -> 200 {"text": str, "tokens": int, "latency_ms": float}
    POST /stream    <- same body
                    -> 200, newline-delimited JSON: one {"token": str} line per
                       generated token, then {"done": true}
    anything wrong  -> 400 {"error": "what was wrong"}   (never a stack trace)
    unknown path    -> 404 {"error": ...}

Streaming exists because time-to-first-token is what a human feels. The stream
must actually stream: write and flush each line as the token is made. With
HTTP/1.0 (http.server's default) and no Content-Length, the end of the body is
the close of the connection, so a client can read line by line as they arrive.
Real servers use chunked transfer encoding or server-sent events for the same
effect.

Generation itself is a generator: ``greedy_token_stream`` yields one token id at
a time, so /generate and /stream share one code path. Plug in room 1's cached
version later; the default here is the plain no-cache loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch


def greedy_token_stream(model, idx: torch.Tensor, max_new_tokens: int) -> Iterator[int]:
    """Yield one greedy token id at a time for the (1, T) prompt ``idx``. No cache needed.

    Stop after ``max_new_tokens`` or when the sequence reaches block_size. Wrap in
    ``torch.no_grad()``.
    """
    raise NotImplementedError("greedy_token_stream() is unwritten")


def validate_request(body: bytes, block_size: int) -> tuple[dict | None, str | None]:
    """Parse a /generate or /stream body. Returns ``(request, None)`` or ``(None, error)``.

    Accept a JSON object with a non-empty string ``prompt`` shorter than block_size
    characters and a positive int ``max_new_tokens`` (default 32; bool is not an int).
    Every other input, including bytes that are not JSON at all, yields an error
    string suitable for a 400 body. Never raise.
    """
    raise NotImplementedError("validate_request() is unwritten")


class InferenceServer:
    """The gate. ``start()`` serves in a daemon thread; ``stop()`` shuts it down cleanly.

    ``port=0`` asks the OS for a free port: after ``start()`` read the real one from
    ``httpd.server_address[1]`` into ``self.port``. ``url`` is ``http://host:port``.
    ``token_stream(model, idx, max_new_tokens)`` yields token ids; default
    ``greedy_token_stream``. Serialize generation with a ``threading.Lock`` so two
    knocks at once do not fight over the CPU (the batcher is how you would do better).
    The response ``text`` is the generated text only, not the prompt.
    """

    def __init__(self, model, tokenizer, host: str = "127.0.0.1", port: int = 0,
                 token_stream: Callable | None = None):
        raise NotImplementedError("InferenceServer.__init__() is unwritten")

    def start(self) -> None:
        raise NotImplementedError("InferenceServer.start() is unwritten")

    def stop(self) -> None:
        raise NotImplementedError("InferenceServer.stop() is unwritten")

    @property
    def url(self) -> str:
        raise NotImplementedError("InferenceServer.url is unwritten")


def make_handler(server: InferenceServer) -> type:
    """Build and return a ``BaseHTTPRequestHandler`` subclass bound to ``server``.

    Implement ``do_GET`` and ``do_POST`` for the routes in the module docstring.
    Read the body with ``self.rfile.read(int(self.headers.get("Content-Length") or 0))``.
    Override ``log_message`` to keep the trial output quiet. Send JSON with a
    Content-Type header and, for non-streaming replies, a Content-Length.
    """
    raise NotImplementedError("make_handler() is unwritten")
