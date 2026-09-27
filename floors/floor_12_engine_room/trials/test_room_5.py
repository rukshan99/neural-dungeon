"""TRIAL 12.5 - THE GATE

The door opens on a free port. A petitioner knocks with JSON, twice with the
same words, once mid-sentence to see the answer arrive a token at a time, and
once with nonsense to see how politely they are turned away.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts.tiny_gpt import load_pretrained  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_5_the_gate")

torch.manual_seed(125)
MODEL, TOK, _ = load_pretrained()
CFG = MODEL.cfg
PROMPT = "Below them, the "
N_NEW = 8


@pytest.fixture(scope="module")
def gate():
    server = room.InferenceServer(MODEL, TOK, host="127.0.0.1", port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def _post(url, body: bytes):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw)
        except json.JSONDecodeError:
            pytest.fail(f"HTTP {err.code} came back without a JSON body: {raw[:200]!r}. Errors are JSON too.")


def _reference_text(prompt, n):
    idx = torch.tensor([TOK.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        out = MODEL.generate(idx, n, temperature=0)
    return TOK.decode(out[0, idx.shape[1]:])


# ------------------------------------------------------------------ the pieces
def test_the_token_stream_yields_greedy_tokens_one_at_a_time():
    idx = torch.tensor([TOK.encode(PROMPT)], dtype=torch.long)
    stream = room.greedy_token_stream(MODEL, idx, N_NEW)
    first = next(stream)
    assert isinstance(first, int), f"Yield plain Python ints, got {type(first).__name__}."
    rest = list(stream)
    assert TOK.decode([first] + rest) == _reference_text(PROMPT, N_NEW), "The stream must reproduce greedy decoding."
    almost_full = torch.zeros(1, CFG.block_size - 2, dtype=torch.long)
    assert len(list(room.greedy_token_stream(MODEL, almost_full, 10))) == 2, "Stop at block_size."


def test_validate_request_turns_nonsense_into_polite_errors():
    ok, err = room.validate_request(json.dumps({"prompt": "hi", "max_new_tokens": 3}).encode(), CFG.block_size)
    assert err is None and ok == {"prompt": "hi", "max_new_tokens": 3}, f"A good request should pass: {ok}, {err}"
    ok, err = room.validate_request(json.dumps({"prompt": "hi"}).encode(), CFG.block_size)
    assert err is None and ok["max_new_tokens"] == 32, "max_new_tokens defaults to 32."
    for bad in [b"{not json", b"[]", json.dumps({}).encode(), json.dumps({"prompt": ""}).encode(),
                json.dumps({"prompt": 5}).encode(), json.dumps({"prompt": "x", "max_new_tokens": 0}).encode(),
                json.dumps({"prompt": "x", "max_new_tokens": "3"}).encode(),
                json.dumps({"prompt": "x", "max_new_tokens": True}).encode(),
                json.dumps({"prompt": "y" * CFG.block_size}).encode()]:
        ok, err = room.validate_request(bad, CFG.block_size)
        assert ok is None and isinstance(err, str) and err, f"{bad[:40]!r} should be rejected with a message, got {ok}, {err!r}"


# ------------------------------------------------------------------ the door
def test_the_gate_opens_on_a_free_port_and_reports_health(gate):
    assert isinstance(gate.port, int) and gate.port > 0, f"After start() the bound port must be known; port is {gate.port!r}."
    assert gate.url == f"http://127.0.0.1:{gate.port}"
    status, body = _get(gate.url + "/health")
    assert status == 200
    assert body.get("status") == "ok", f"/health should say status ok: {body}"
    assert "model" in body, f"/health should name the model: {body}"


def test_generate_is_deterministic_and_matches_greedy_decoding(gate):
    payload = json.dumps({"prompt": PROMPT, "max_new_tokens": N_NEW}).encode()
    status, first = _post(gate.url + "/generate", payload)
    assert status == 200, f"/generate returned {status}: {first}"
    assert set(first) >= {"text", "tokens", "latency_ms"}, f"/generate keys: {sorted(first)}"
    assert first["tokens"] == N_NEW, f"asked for {N_NEW} tokens, got {first['tokens']}"
    assert first["text"] == _reference_text(PROMPT, N_NEW), (
        f"text should be the GENERATED text only ({_reference_text(PROMPT, N_NEW)!r}), got {first['text']!r}"
    )
    assert isinstance(first["latency_ms"], (int, float)) and first["latency_ms"] > 0
    _, second = _post(gate.url + "/generate", payload)
    assert second["text"] == first["text"], "Same prompt, same greedy answer. Twice."


def test_stream_arrives_one_token_at_a_time_and_adds_up(gate):
    payload = json.dumps({"prompt": PROMPT, "max_new_tokens": N_NEW}).encode()
    req = urllib.request.Request(gate.url + "/stream", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status == 200
        lines = [line for line in resp.read().decode("utf-8").splitlines() if line.strip()]
    chunks = []
    for line in lines:
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError:
            pytest.fail(f"Every stream line must be one JSON object; got {line!r}")
    assert len(chunks) > 2, f"Expected one chunk per token plus a done marker; got {len(chunks)} chunks."
    assert chunks[-1] == {"done": True}, f"The last line must be {{\"done\": true}}, got {chunks[-1]}"
    tokens = [c["token"] for c in chunks[:-1]]
    assert len(tokens) == N_NEW, f"{N_NEW} tokens were requested; the stream carried {len(tokens)}."
    _, whole = _post(gate.url + "/generate", payload)
    assert "".join(tokens) == whole["text"], (
        f"The streamed tokens {''.join(tokens)!r} must concatenate to the non-streamed text {whole['text']!r}."
    )


def test_the_stream_really_streams_instead_of_buffering_until_the_end():
    """A token stream that, after its first token, waits for the client to have READ the first line.

    A server that flushes each line as it is made releases it at once. A server that buffers the
    body until the generator finishes can never release it: the wait times out and the test fails.
    """
    released = threading.Event()
    seen_release: list[bool] = []
    ids = TOK.encode("ab")

    def gated_stream(model, idx, max_new_tokens):
        yield ids[0]
        seen_release.append(released.wait(timeout=5.0))
        yield ids[1]

    server = room.InferenceServer(MODEL, TOK, host="127.0.0.1", port=0, token_stream=gated_stream)
    server.start()
    try:
        payload = json.dumps({"prompt": PROMPT, "max_new_tokens": 2}).encode()
        req = urllib.request.Request(server.url + "/stream", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            first = resp.readline()
            released.set()
            rest = resp.read()
    finally:
        released.set()
        server.stop()
    assert json.loads(first) == {"token": "a"}, f"The first line of the stream should be the first token; got {first!r}"
    assert seen_release == [True], (
        "The client did not receive the first line until the whole reply had been generated: the stream is being "
        "buffered. Write each line and flush() it as the token is produced, and send no Content-Length on /stream."
    )
    tail = [json.loads(line) for line in rest.decode("utf-8").splitlines() if line.strip()]
    assert tail == [{"token": "b"}, {"done": True}], f"After the first line the rest of the stream should be the second token and the done marker; got {tail}"


def test_nonsense_is_turned_away_with_a_400_and_a_json_error(gate):
    for body in [b"{this is not json", json.dumps({"max_new_tokens": 3}).encode(), json.dumps({"prompt": "x", "max_new_tokens": -1}).encode()]:
        status, payload = _post(gate.url + "/generate", body)
        assert status == 400, f"{body[:30]!r} should be a 400, got {status}: {payload}"
        assert isinstance(payload.get("error"), str) and payload["error"], f"400 bodies carry an 'error' string: {payload}"
    status, payload = _post(gate.url + "/stream", b"nope")
    assert status == 400 and "error" in payload, "/stream validates too."


def test_unknown_doors_are_a_404(gate):
    status, payload = _post(gate.url + "/summon", b"{}")
    assert status == 404 and "error" in payload, f"Unknown routes are 404 with a JSON error; got {status}: {payload}"
    try:
        _get(gate.url + "/nothing")
    except urllib.error.HTTPError as err:
        assert err.code == 404
    else:
        pytest.fail("GET /nothing should be a 404.")


def test_the_gate_closes_cleanly():
    server = room.InferenceServer(MODEL, TOK, host="127.0.0.1", port=0)
    server.start()
    url = server.url
    assert _get(url + "/health")[0] == 200
    server.stop()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url + "/health", timeout=2)
    server.stop()  # a second stop must be harmless
