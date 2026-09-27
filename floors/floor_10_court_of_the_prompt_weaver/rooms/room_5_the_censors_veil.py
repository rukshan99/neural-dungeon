"""ROOM 10.5 - THE CENSOR'S VEIL

    Every reply the Weaver speaks passes through a veil before it leaves the
    court. Behind the veil sits the Censor, who has read every petition and
    every herald's report and knows what the Weaver ought not to repeat. The
    Censor also keeps a drawer of answers already given, because the court is
    asked the same question rather more often than it would like.

Floor 10 so far defends the INPUT side: what reaches the model. The output
side needs defences too, because a model repeats what it has read. It will
echo a card number from the record it was asked to summarise, quote the API
key it found in a config file a tool returned, or be talked into listing
every address in the archive (a request no allowlist forbids). Whatever
reaches the model may come back out, and what comes out is also written to
your logs. So the last step before a reply leaves is a checkpoint:

* ``redact_pii`` finds emails, phone numbers, IPv4 addresses and card numbers
  (13-19 digits that pass the Luhn check; ``luhn_valid`` is yours to write)
  and replaces each with a token like ``[EMAIL]``. Every finding carries its
  ORIGINAL offsets and the replacement, never the value, so a log can say
  where without saying what. Findings never overlap.
* ``scan_for_secrets`` recognises credential shapes (API keys, AWS access key
  ids, GitHub tokens, private key headers, ``password=`` assignments) and
  reports their KINDS. The scanner's own output ends up in logs too.
* ``handle_completion`` turns ``stop_reason`` into a code path. A refusal is
  not an exception: the reply is a fallback text, and the model's words never
  leave. A reply cut at ``max_tokens`` is asked to continue, a bounded number
  of times.
* ``ResponseCache`` remembers answers by a sha256 of the normalised request,
  with a TTL on an injected clock and LRU eviction. ``cached_complete`` puts
  it in front of a model. Two traps: a sampled answer is one of many, and an
  answer cached for one user is a leak to the next.
* ``OutputPolicy`` composes the veil: credentials block the reply, PII is
  redacted, and a report says what happened.

Nothing here sleeps or touches a network. The trial injects a clock.
"""

from __future__ import annotations

import hashlib  # noqa: F401  (key_for hashes with sha256)
import json  # noqa: F401  (key_for serialises a canonical form)
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from dungeon.artifacts.llm import LLM, Completion, Message, ToolSpec

# ---------------------------------------------------------------------------
# PII: the patterns are provided; Luhn, overlaps and offsets are yours.
# Each pattern is deliberately conservative. A redactor that eats every
# number is a redactor nobody leaves switched on.
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# 13-19 digits, optionally separated by single spaces or dashes, not glued to
# other digits. Whether it is a CARD is decided by luhn_valid, not the regex.
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

# Four octets of 0-255. A version string like 3.14.256.1 has an octet too big.
IPV4_RE = re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)")

# International: '+' and a country code, then 2-4 groups of 2-4 digits.
# US style: optional +1, then 3-3-4 with spaces, dots, dashes or (parentheses).
PHONE_RE = re.compile(
    r"(?<![\w.+-])"
    r"(?:\+\d{1,3}(?:[\s.-]?\d{2,4}){2,4}"
    r"|(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4})"
    r"(?![\w-])"
)

# Order matters: the first kind to claim a span keeps it. An email with digits
# in its local part is one email, not an email and a phone number.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", EMAIL_RE),
    ("card", CARD_RE),
    ("ip", IPV4_RE),
    ("phone", PHONE_RE),
]
REPLACEMENTS: dict[str, str] = {"email": "[EMAIL]", "card": "[CARD]", "ip": "[IP]", "phone": "[PHONE]"}


@dataclass(frozen=True)
class Finding:
    """One redaction. ``span`` is ``(start, end)`` in the ORIGINAL text; the value itself is not kept."""

    kind: str
    span: tuple[int, int]
    replacement: str


def luhn_valid(number: str) -> bool:
    """True if ``number`` passes the Luhn check.

    ``number`` may contain spaces or dashes between digits; strip them first.
    Anything else that is not a digit, or fewer than two digits, is not valid.
    The check: from the RIGHTMOST digit, double every second digit (the
    second, fourth, ... from the right); if a doubled digit exceeds 9,
    subtract 9; add all digits up; valid when the sum is divisible by 10.
    ``4111 1111 1111 1111`` passes; change the last digit to 2 and it fails.
    """
    raise NotImplementedError("luhn_valid() is unwritten")


def redact_pii(text: str) -> tuple[str, list[Finding]]:
    """Replace every email, card number, IPv4 address and phone number with its token.

    Walk ``PII_PATTERNS`` in order. For each match: a ``"card"`` match counts
    only if ``luhn_valid`` says so; any match that overlaps a span already
    claimed is skipped. Record ``Finding(kind, (start, end), REPLACEMENTS[kind])``
    with offsets into the ORIGINAL ``text``. Return the redacted text (built
    by splicing the replacements in at those spans) and the findings sorted
    by start. Same input, same output, every time.
    """
    raise NotImplementedError("redact_pii() is unwritten")


# ---------------------------------------------------------------------------
# SECRETS: credential shapes. The scanner returns KINDS, never values.
# ---------------------------------------------------------------------------

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "sk-" style API keys: the prefix and a long alphanumeric run.
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    # AWS access key ids: AKIA and sixteen upper-case letters or digits.
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # GitHub tokens: ghp_/gho_/ghu_/ghs_/ghr_ and 36+ alphanumerics.
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    # PEM private key headers, any algorithm.
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # Generic assignments: password=..., api_key: "...", secret_key = ...
    (
        "assignment",
        re.compile(
            r"\b(?:password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*['\"]?[^\s'\"]{4,}",
            re.IGNORECASE,
        ),
    ),
]


def scan_for_secrets(text: str) -> list[str]:
    """The kind of every credential-shaped match in ``text``, in order of appearance.

    Use ``SECRET_PATTERNS``. Return kinds (``"api_key"``, ``"aws_access_key_id"``,
    ...), one per match, sorted by where the match starts. Never return the
    matched value: this list is what gets logged.
    """
    raise NotImplementedError("scan_for_secrets() is unwritten")


# ---------------------------------------------------------------------------
# STOP REASONS: refusal and truncation are code paths, not exceptions.
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """What the court will actually say. ``status`` is ``"ok"``, ``"refused"`` or ``"truncated"``."""

    text: str
    status: str
    notes: list[str] = field(default_factory=list)


def handle_completion(
    completion: Completion,
    *,
    on_refusal_text: str,
    continue_fn: Callable[[str], Completion] | None = None,
    max_continuations: int = 2,
) -> Outcome:
    """Turn a completion's ``stop_reason`` into an ``Outcome``.

    * ``"refusal"``: return ``Outcome(on_refusal_text, "refused", notes)``. The
      model's own text is NOT the reply and does not go in the notes either.
    * ``"max_tokens"``: the reply was cut. Starting from ``completion.text``,
      call ``continue_fn(text_so_far)`` (it returns a Completion) and append
      its text, up to ``max_continuations`` times. A continuation that stops
      with anything other than ``"max_tokens"`` finishes the reply: status
      ``"ok"``. A continuation that is itself a refusal is not appended and
      ends the attempt. Still cut when the budget is spent, or no
      ``continue_fn`` given: status ``"truncated"`` with the partial text.
    * anything else (``"end_turn"``, ...): ``Outcome(completion.text, "ok", notes)``.

    ``notes`` is free-form; use it to say what happened (how many
    continuations, why it stopped).
    """
    raise NotImplementedError("handle_completion() is unwritten")


# ---------------------------------------------------------------------------
# THE DRAWER OF ANSWERS ALREADY GIVEN
# ---------------------------------------------------------------------------


class ResponseCache:
    """Remember completions by request key, with a TTL and LRU eviction.

    ``clock()`` returns seconds; the trial injects one and moves it by hand.
    An entry is live while ``clock() - stored_at < ttl_seconds``. When the
    cache holds more than ``max_entries`` the least recently USED entry goes
    (a hit counts as use). ``hits``, ``misses`` and ``evictions`` are counters
    a dashboard would want.
    """

    def __init__(self, ttl_seconds: float, max_entries: int, clock: Callable[[], float] = time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._entries: OrderedDict[str, tuple[float, Completion]] = OrderedDict()  # key -> (stored_at, completion)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def normalise(text: str) -> str:
        """Collapse every run of whitespace to one space and strip the ends. (Provided.)"""
        return " ".join(text.split())

    def key_for(self, messages: Sequence[Message], params: Mapping[str, object]) -> str:
        """A sha256 hex digest of the request.

        Build a canonical structure: for each message its ``role``, its
        ``normalise``d content and, for assistant messages, the name and
        arguments of each tool call (not the ids: ids differ per run); plus
        ``params`` with keys sorted. Serialise it with
        ``json.dumps(..., sort_keys=True, default=str)`` and hash the UTF-8
        bytes. Two requests that differ only in whitespace share a key; two
        that differ in any role, content or param do not.
        """
        raise NotImplementedError("ResponseCache.key_for() is unwritten")

    def get(self, key: str) -> Completion | None:
        """The live completion for ``key``, or None.

        Missing: count a miss. Present but expired: delete it, count a miss.
        Live: move it to the most-recently-used end, count a hit, return it.
        """
        raise NotImplementedError("ResponseCache.get() is unwritten")

    def set(self, key: str, completion: Completion) -> None:
        """Store ``completion`` under ``key`` stamped with ``clock()``, as most recently used.

        Then evict from the least-recently-used end until at most
        ``max_entries`` remain, counting each eviction.
        """
        raise NotImplementedError("ResponseCache.set() is unwritten")


def cached_complete(
    llm: LLM,
    cache: ResponseCache,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None = None,
    *,
    max_tokens: int = 1024,
    **params: object,
) -> Completion:
    """``llm.complete`` behind ``cache``.

    Key params: ``{"max_tokens": max_tokens, "tools": sorted tool names, **params}``.
    The extra ``params`` (``user_id``, ``model``, ``temperature``...) are cache
    NAMESPACE: part of the key, never passed to the model. Anything that makes
    the right answer differ between two callers belongs in them; a cached
    answer shared across users is a data leak with excellent latency.

    On a hit return the cached completion without calling the model. On a
    miss call ``llm.complete(messages, tools, max_tokens=max_tokens)`` and
    store the result only if its ``stop_reason`` is ``"end_turn"``: a cut-off
    reply, a refusal or a tool request is not an answer worth remembering.
    """
    raise NotImplementedError("cached_complete() is unwritten")


# ---------------------------------------------------------------------------
# THE VEIL ITSELF
# ---------------------------------------------------------------------------

BLOCKED_TEXT = "The court has withheld this reply: it contained credentials that must not leave the room."


@dataclass
class VeilReport:
    """What the veil did. ``secrets`` are kinds; ``findings`` are redactions (offsets, not values)."""

    blocked: bool = False
    secrets: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


@dataclass
class OutputPolicy:
    """The last step before a reply leaves the court (and before it is logged).

    ``apply(text)``: always ``scan_for_secrets`` (the report must be honest
    even when blocking is off). If ``block_secrets`` and anything was found,
    the reply becomes ``blocked_text`` and ``blocked`` is True; nothing else
    is done, there is no text left to redact. Otherwise, if ``redact``, run
    ``redact_pii`` and keep its findings. Return ``(text, VeilReport)``.
    """

    redact: bool = True
    block_secrets: bool = True
    blocked_text: str = BLOCKED_TEXT

    def apply(self, text: str) -> tuple[str, VeilReport]:
        raise NotImplementedError("OutputPolicy.apply() is unwritten")
