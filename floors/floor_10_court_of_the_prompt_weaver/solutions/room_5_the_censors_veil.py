"""ROOM 10.5 - THE CENSOR'S VEIL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Output-side guardrails and a response cache:

* redact_pii: emails, Luhn-checked cards, IPv4, phones; first kind to claim a
  span keeps it; findings carry original offsets and never the value
* scan_for_secrets: credential shapes, reported by kind
* handle_completion: refusal -> fallback text; max_tokens -> bounded continuation
* ResponseCache: sha256 of the normalised request, TTL on an injected clock,
  LRU eviction; cached_complete stores only end_turn replies
* OutputPolicy: secrets block, PII is redacted, a report says what happened
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from dungeon.artifacts.llm import LLM, Completion, Message, ToolSpec

# --------------------------------------------------------------------- PII

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
IPV4_RE = re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)")
PHONE_RE = re.compile(
    r"(?<![\w.+-])"
    r"(?:\+\d{1,3}(?:[\s.-]?\d{2,4}){2,4}"
    r"|(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4})"
    r"(?![\w-])"
)

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", EMAIL_RE),
    ("card", CARD_RE),
    ("ip", IPV4_RE),
    ("phone", PHONE_RE),
]
REPLACEMENTS: dict[str, str] = {"email": "[EMAIL]", "card": "[CARD]", "ip": "[IP]", "phone": "[PHONE]"}


@dataclass(frozen=True)
class Finding:
    """One redaction: what kind, where in the ORIGINAL text, and what replaced it."""

    kind: str
    span: tuple[int, int]
    replacement: str


def luhn_valid(number: str) -> bool:
    """The Luhn check: from the right, double every second digit, fold >9 to a digit, sum % 10 == 0."""
    digits = number.replace(" ", "").replace("-", "")
    if len(digits) < 2 or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:  # every second digit from the right
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[Finding]]:
    """Replace PII with tokens. First kind to claim a span keeps it; spans never overlap."""
    taken: list[tuple[int, int]] = []
    findings: list[Finding] = []
    for kind, pattern in PII_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if kind == "card" and not luhn_valid(match.group()):
                continue  # sixteen digits are just sixteen digits
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            findings.append(Finding(kind, (start, end), REPLACEMENTS[kind]))
    findings.sort(key=lambda f: f.span)

    pieces: list[str] = []
    cursor = 0
    for finding in findings:
        start, end = finding.span
        pieces.append(text[cursor:start])
        pieces.append(finding.replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), findings


# ----------------------------------------------------------------- secrets

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "assignment",
        re.compile(
            r"\b(?:password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*['\"]?[^\s'\"]{4,}",
            re.IGNORECASE,
        ),
    ),
]


def scan_for_secrets(text: str) -> list[str]:
    """Kinds of every credential-shaped match, in order of appearance. Never the values."""
    hits: list[tuple[int, str]] = []
    for kind, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            hits.append((match.start(), kind))
    hits.sort()
    return [kind for _, kind in hits]


# ------------------------------------------------------------ stop reasons


@dataclass
class Outcome:
    text: str
    status: str  # "ok" | "refused" | "truncated"
    notes: list[str] = field(default_factory=list)


def handle_completion(
    completion: Completion,
    *,
    on_refusal_text: str,
    continue_fn: Callable[[str], Completion] | None = None,
    max_continuations: int = 2,
) -> Outcome:
    """Refusal -> fallback text. max_tokens -> bounded continuation. Otherwise the text as is."""
    reason = completion.stop_reason
    if reason == "refusal":
        # The model's words are not the reply and do not go in the notes: notes are logged.
        return Outcome(on_refusal_text, "refused", ["model refused; fallback text used"])
    if reason != "max_tokens":
        notes = [] if reason == "end_turn" else [f"unexpected stop_reason {reason!r}; treated as final"]
        return Outcome(completion.text, "ok", notes)

    text = completion.text
    notes = ["reply cut at max_tokens"]
    if continue_fn is None:
        notes.append("no continue_fn; returning the partial reply")
        return Outcome(text, "truncated", notes)

    continuations = 0
    while continuations < max_continuations:
        more = continue_fn(text)
        continuations += 1
        if more.stop_reason == "refusal":
            notes.append(f"continuation {continuations} was refused; stopping with the partial reply")
            return Outcome(text, "truncated", notes)
        text += more.text
        if more.stop_reason != "max_tokens":
            notes.append(f"completed after {continuations} continuation(s)")
            return Outcome(text, "ok", notes)
    notes.append(f"still cut after {continuations} continuation(s)")
    return Outcome(text, "truncated", notes)


# ------------------------------------------------------------------- cache


class ResponseCache:
    """Completions by request key; TTL on an injected clock; least recently used goes first."""

    def __init__(self, ttl_seconds: float, max_entries: int, clock: Callable[[], float] = time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._entries: OrderedDict[str, tuple[float, Completion]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def normalise(text: str) -> str:
        return " ".join(text.split())

    def key_for(self, messages: Sequence[Message], params: Mapping[str, object]) -> str:
        canonical = {
            "messages": [
                {
                    "role": m.role,
                    "content": self.normalise(m.content or ""),
                    # names and arguments, not ids: ids are minted per run
                    "tool_calls": [[c.name, c.arguments] for c in m.tool_calls],
                }
                for m in messages
            ],
            "params": {k: params[k] for k in sorted(params)},
        }
        blob = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Completion | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        stored_at, completion = entry
        if self.clock() - stored_at >= self.ttl_seconds:
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)  # a hit is a use
        self.hits += 1
        return completion

    def set(self, key: str, completion: Completion) -> None:
        self._entries[key] = (self.clock(), completion)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1


def cached_complete(
    llm: LLM,
    cache: ResponseCache,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None = None,
    *,
    max_tokens: int = 1024,
    **params: object,
) -> Completion:
    """llm.complete behind the cache. Extra params namespace the key and never reach the model."""
    key_params: dict[str, object] = {
        "max_tokens": max_tokens,
        "tools": sorted(t.name for t in tools or []),
        **params,
    }
    key = cache.key_for(messages, key_params)
    hit = cache.get(key)
    if hit is not None:
        return hit
    completion = llm.complete(messages, tools, max_tokens=max_tokens)
    if completion.stop_reason == "end_turn":
        cache.set(key, completion)
    return completion


# -------------------------------------------------------------------- veil

BLOCKED_TEXT = "The court has withheld this reply: it contained credentials that must not leave the room."


@dataclass
class VeilReport:
    blocked: bool = False
    secrets: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


@dataclass
class OutputPolicy:
    """Secrets first (they block), then PII (it is redacted), then a report."""

    redact: bool = True
    block_secrets: bool = True
    blocked_text: str = BLOCKED_TEXT

    def apply(self, text: str) -> tuple[str, VeilReport]:
        secrets = scan_for_secrets(text)
        if self.block_secrets and secrets:
            return self.blocked_text, VeilReport(blocked=True, secrets=secrets, findings=[])
        findings: list[Finding] = []
        if self.redact:
            text, findings = redact_pii(text)
        return text, VeilReport(blocked=False, secrets=secrets, findings=findings)
