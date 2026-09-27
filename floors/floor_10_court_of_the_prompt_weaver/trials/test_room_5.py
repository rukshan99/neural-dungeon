"""TRIAL 10.5 - THE CENSOR'S VEIL

Nothing leaves the court unread. Card numbers must pass Luhn before they are
cards, credentials block the reply, a refusal is a code path, a cut-off reply
is continued a bounded number of times, and no petitioner ever receives
another petitioner's cached answer. The clock is injected; nothing sleeps.
"""

from dungeon.artifacts.llm import Completion, Message, ScriptedLLM, TruncatingLLM
from dungeon.trials import load_room

room = load_room(__file__, "room_5_the_censors_veil")

# Obviously fake test values. The card numbers are the classic test numbers;
# the AWS key id is the one from AWS's own documentation.
GOOD_CARD = "4111 1111 1111 1111"  # passes Luhn
BAD_CARD = "4111 1111 1111 1112"  # sixteen digits that fail Luhn
FAKE_SK = "sk-" + "FAKE" * 6
FAKE_AKIA = "AKIAIOSFODNN7EXAMPLE"
FAKE_GHP = "ghp_" + "0123456789abcdef" * 2 + "0123"
PEM_HEADER = "-----BEGIN RSA PRIVATE KEY-----"


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def kinds(findings):
    return [f.kind for f in findings]


def continuation_from(llm):
    """A continue_fn backed by a ScriptedLLM; records what it was handed."""
    received = []

    def continue_fn(text_so_far):
        received.append(text_so_far)
        return llm.complete([Message.user("Continue exactly where you stopped:\n" + text_so_far)])

    return continue_fn, received


# ----------------------------------------------------------------------- luhn
def test_luhn_accepts_the_classic_test_numbers():
    for number in (GOOD_CARD, "4111111111111111", "79927398713", "4111-1111-1111-1111"):
        assert room.luhn_valid(number) is True, f"{number!r} passes the Luhn check; you said it does not."


def test_luhn_rejects_a_number_off_by_one():
    for number in (BAD_CARD, "79927398710", "1234 5678 9012 3456"):
        assert room.luhn_valid(number) is False, (
            f"{number!r} fails the Luhn check; you accepted it. Double every SECOND digit from the RIGHT, "
            "subtract 9 when a doubled digit exceeds 9, and the sum must be divisible by 10."
        )


def test_luhn_rejects_things_that_are_not_numbers():
    for junk in ("", "7", "abcd", "4111 1111 1111 111x"):
        assert room.luhn_valid(junk) is False, f"{junk!r} is not a digit string of two or more digits; it cannot be Luhn-valid."


# ----------------------------------------------------------------- redact_pii
def test_an_email_is_veiled_and_its_offsets_point_at_the_original():
    text = "Write to wren.of.the.tower@example.invalid before dusk."
    redacted, findings = room.redact_pii(text)
    assert redacted == "Write to [EMAIL] before dusk.", f"Expected the address replaced by [EMAIL], got {redacted!r}"
    assert kinds(findings) == ["email"], f"One email finding expected, got {findings}"
    start, end = findings[0].span
    assert text[start:end] == "wren.of.the.tower@example.invalid", (
        f"span must index the ORIGINAL text: text[{start}:{end}] is {text[start:end]!r}. Record offsets before you splice."
    )
    assert findings[0].replacement == "[EMAIL]"


def test_phone_numbers_of_several_shapes_are_veiled():
    text = "Ring (555) 123-4567 or 555.123.4567, or from abroad +44 20 7946 0958."
    redacted, findings = room.redact_pii(text)
    assert redacted == "Ring [PHONE] or [PHONE], or from abroad [PHONE].", (
        f"Three phone numbers in three shapes should all become [PHONE]; got {redacted!r}"
    )
    assert kinds(findings) == ["phone", "phone", "phone"]
    spans = [f.span for f in findings]
    assert spans == sorted(spans), f"Findings must be sorted by start: {spans}"


def test_an_ipv4_address_is_veiled_but_a_version_number_with_a_big_octet_is_not():
    text = "The herald's post is at 192.168.10.42; the scroll is version 3.14.256.1."
    redacted, findings = room.redact_pii(text)
    assert "192.168.10.42" not in redacted and "[IP]" in redacted, f"192.168.10.42 should become [IP]; got {redacted!r}"
    assert "3.14.256.1" in redacted, f"3.14.256.1 has an octet above 255; it is not an address. Got {redacted!r}"
    assert kinds(findings) == ["ip"]


def test_a_card_that_passes_luhn_is_veiled_and_one_that_fails_is_not():
    text = f"Charge {GOOD_CARD} and never {BAD_CARD}."
    redacted, findings = room.redact_pii(text)
    assert GOOD_CARD not in redacted and "[CARD]" in redacted, f"{GOOD_CARD} passes Luhn and must become [CARD]; got {redacted!r}"
    assert kinds(findings) == ["card"], f"Exactly one finding expected, the Luhn-valid card; got {kinds(findings)}"
    assert BAD_CARD in redacted, (
        f"{BAD_CARD} is sixteen digits that FAIL the Luhn check. Sixteen digits are not a card number; "
        f"leave them alone. Got {redacted!r}"
    )
    start, end = findings[0].span
    assert text[start:end] == GOOD_CARD, f"The card finding's span must cover the original digits, got {text[start:end]!r}"


def test_findings_never_overlap_and_the_first_kind_claims_the_span():
    text = "Invoices go to 5551234567@example.invalid and 4111111111111111@example.invalid today."
    redacted, findings = room.redact_pii(text)
    assert redacted == "Invoices go to [EMAIL] and [EMAIL] today.", (
        f"An email whose local part looks like a phone or card is ONE email. Got {redacted!r}"
    )
    assert kinds(findings) == ["email", "email"], f"Two email findings and nothing else; got {kinds(findings)}"
    for a, b in zip(findings, findings[1:]):
        assert a.span[1] <= b.span[0], f"Findings overlap: {a.span} and {b.span}. A span claimed once is claimed for good."


def test_clean_prose_full_of_numbers_passes_untouched():
    text = "In 1066 the court hired 3 heralds; chapter 12 says the moat is 9 feet deep and 250 long."
    redacted, findings = room.redact_pii(text)
    assert redacted == text, f"Nothing here is PII, yet you changed it: {redacted!r}"
    assert findings == [], f"Nothing here is PII, yet you found {findings}"


def test_redaction_is_deterministic_and_reconstructible():
    text = f"Card {GOOD_CARD}, mail imp@example.invalid, ring 555-123-4567, host 10.0.0.7."
    first = room.redact_pii(text)
    second = room.redact_pii(text)
    assert first == second, "Same text, same result. Redaction must be deterministic."
    redacted, findings = first
    assert kinds(findings) == ["card", "email", "phone", "ip"], f"Expected card, email, phone, ip in text order; got {kinds(findings)}"
    rebuilt, cursor = "", 0
    for f in findings:
        rebuilt += text[cursor : f.span[0]] + f.replacement
        cursor = f.span[1]
    rebuilt += text[cursor:]
    assert rebuilt == redacted, "Splicing the replacements in at the recorded spans must reproduce the redacted text."


# ----------------------------------------------------------- scan_for_secrets
def test_every_credential_shape_is_recognised_by_kind():
    text = (
        f"The key is {FAKE_SK}. AWS said {FAKE_AKIA}. GitHub gave {FAKE_GHP}.\n"
        f"{PEM_HEADER}\nMIIEFAKE\n-----END RSA PRIVATE KEY-----\n"
        'api_key = "FAKE-TRIAL-KEY-0000"'
    )
    found = room.scan_for_secrets(text)
    assert found == ["api_key", "aws_access_key_id", "github_token", "private_key", "assignment"], (
        f"Expected the five kinds in order of appearance, got {found}"
    )


def test_the_scanner_reports_kinds_not_values():
    found = room.scan_for_secrets(f"password=not-a-real-password and {FAKE_AKIA}")
    joined = " ".join(found)
    assert "not-a-real-password" not in joined and FAKE_AKIA not in joined, (
        f"The scanner's output is logged. Return kinds, never the matched values: {found}"
    )
    assert found == ["assignment", "aws_access_key_id"]


def test_ordinary_prose_about_passwords_is_not_a_secret():
    text = "The password policy is strict, the token budget is tight, and the task-force met at dusk."
    assert room.scan_for_secrets(text) == [], "Prose ABOUT credentials has no credential in it. The patterns need the value."


# ---------------------------------------------------------- handle_completion
def test_an_ordinary_reply_is_ok():
    outcome = room.handle_completion(Completion(Message.assistant("The moat is nine feet deep.")), on_refusal_text="(withheld)")
    assert outcome.status == "ok", f"end_turn is the ordinary case; status should be 'ok', got {outcome.status!r}"
    assert outcome.text == "The moat is nine feet deep."


def test_a_refusal_is_a_code_path_not_an_exception():
    refusal = Completion(Message.assistant("I cannot help with dredging the moat at night."), stop_reason="refusal")
    outcome = room.handle_completion(refusal, on_refusal_text="The court cannot answer that petition.")
    assert outcome.status == "refused", f"stop_reason='refusal' should give status 'refused', got {outcome.status!r}"
    assert outcome.text == "The court cannot answer that petition.", f"The reply is the fallback text, got {outcome.text!r}"
    assert "dredging" not in outcome.text and all("dredging" not in n for n in outcome.notes), (
        "The model's own words never leave: not in the reply, not in the notes (notes are logged)."
    )


def test_a_truncated_reply_is_continued_until_it_ends():
    first = TruncatingLLM(ScriptedLLM(["The drawbridge needs new chains before winter, and the moat wants dredging."])).complete(
        [Message.user("Report.")], max_tokens=8
    )
    assert first.stop_reason == "max_tokens"  # the fixture cut it; that is the situation under test
    cut = first.text
    tail = ScriptedLLM([
        {"text": "before winter, and the moat ", "stop_reason": "max_tokens"},
        {"text": "wants dredging.", "stop_reason": "end_turn"},
    ])
    continue_fn, received = continuation_from(tail)
    outcome = room.handle_completion(first, on_refusal_text="(withheld)", continue_fn=continue_fn, max_continuations=3)
    assert outcome.status == "ok", f"The second continuation ended with end_turn, so the reply is complete: expected 'ok', got {outcome.status!r}"
    assert outcome.text == cut + "before winter, and the moat wants dredging.", f"Continuations are appended in order; got {outcome.text!r}"
    assert len(tail.calls) == 2, f"Two continuations were needed; you made {len(tail.calls)} call(s)."
    assert received == [cut, cut + "before winter, and the moat "], (
        "continue_fn must be handed the text so far, so the model knows where to pick up."
    )


def test_continuations_are_bounded():
    endless = ScriptedLLM([{"text": f"piece {i} ", "stop_reason": "max_tokens"} for i in range(5)])
    continue_fn, _ = continuation_from(endless)
    start = Completion(Message.assistant("piece 0 "), stop_reason="max_tokens")
    outcome = room.handle_completion(start, on_refusal_text="(withheld)", continue_fn=continue_fn, max_continuations=2)
    assert len(endless.calls) == 2, f"max_continuations=2 means two continuation calls, you made {len(endless.calls)}. Bounded is the point."
    assert outcome.status == "truncated", f"Still cut after the budget: status 'truncated', got {outcome.status!r}"
    assert outcome.text == "piece 0 piece 0 piece 1 ", f"The partial reply, with what was fetched, is still returned: got {outcome.text!r}"


def test_no_continue_fn_means_truncated_with_the_partial_text():
    start = Completion(Message.assistant("The drawbridge needs"), stop_reason="max_tokens")
    outcome = room.handle_completion(start, on_refusal_text="(withheld)")
    assert outcome.status == "truncated", f"Nothing to continue with: status 'truncated', got {outcome.status!r}"
    assert outcome.text == "The drawbridge needs", "Return what the model did say; the caller decides what to do with a partial."


def test_a_refusal_during_continuation_never_reaches_the_reply():
    tail = ScriptedLLM([{"text": "I will not continue that.", "stop_reason": "refusal"}])
    continue_fn, _ = continuation_from(tail)
    start = Completion(Message.assistant("The moat is "), stop_reason="max_tokens")
    outcome = room.handle_completion(start, on_refusal_text="(withheld)", continue_fn=continue_fn)
    assert outcome.status == "truncated", f"A refused continuation leaves the reply cut: expected 'truncated', got {outcome.status!r}"
    assert outcome.text == "The moat is ", f"The refusal's words are not appended: got {outcome.text!r}"
    assert len(tail.calls) == 1, "After a refusal, stop asking."


# -------------------------------------------------------------- ResponseCache
def moat_question():
    return [Message.system("You are the court's surveyor."), Message.user("How deep is the moat?")]


def test_identical_requests_cost_one_model_call():
    llm = ScriptedLLM(["Nine feet, and colder than it looks."] * 3)
    cache = room.ResponseCache(ttl_seconds=60.0, max_entries=8, clock=FakeClock())
    first = room.cached_complete(llm, cache, moat_question())
    second = room.cached_complete(llm, cache, moat_question())
    assert len(llm.calls) == 1, f"The same request twice is one model call, you made {len(llm.calls)}."
    assert first.text == second.text == "Nine feet, and colder than it looks."
    assert (cache.hits, cache.misses) == (1, 1), f"One miss then one hit; counters say hits={cache.hits}, misses={cache.misses}"


def test_whitespace_does_not_change_the_key():
    cache = room.ResponseCache(ttl_seconds=60.0, max_entries=8, clock=FakeClock())
    tidy = cache.key_for([Message.user("How deep is the moat?")], {})
    messy = cache.key_for([Message.user("  How   deep is\nthe moat?  ")], {})
    assert tidy == messy, "Collapse runs of whitespace and strip before hashing: the same question in a different layout is the same question."
    assert len(tidy) == 64 and all(c in "0123456789abcdef" for c in tidy), f"A sha256 hex digest is 64 hex characters, got {tidy!r}"


def test_content_role_and_params_all_change_the_key():
    cache = room.ResponseCache(ttl_seconds=60.0, max_entries=8, clock=FakeClock())
    base = cache.key_for([Message.user("How deep is the moat?")], {"max_tokens": 100})
    assert cache.key_for([Message.user("How wide is the moat?")], {"max_tokens": 100}) != base, "Different content, different key."
    assert cache.key_for([Message.system("How deep is the moat?")], {"max_tokens": 100}) != base, "Same text in a different role is a different request."
    assert cache.key_for([Message.user("How deep is the moat?")], {"max_tokens": 200}) != base, "Params are part of the request; hash them too."


def test_entries_expire_after_ttl_on_the_injected_clock():
    clock = FakeClock()
    cache = room.ResponseCache(ttl_seconds=60.0, max_entries=8, clock=clock)
    answer = Completion(Message.assistant("Nine feet."))
    cache.set("moat", answer)
    clock.advance(59.9)
    assert cache.get("moat") is answer, "59.9s into a 60s TTL the entry is still live."
    clock.advance(0.1)
    assert cache.get("moat") is None, "At 60s the entry has expired: a miss, not a stale answer."
    assert len(cache) == 0, "An expired entry is removed when it is found expired."
    assert (cache.hits, cache.misses) == (1, 1), f"One hit then one miss; counters say hits={cache.hits}, misses={cache.misses}"


def test_the_least_recently_used_entry_is_evicted_first():
    cache = room.ResponseCache(ttl_seconds=600.0, max_entries=2, clock=FakeClock())
    a, b, c = (Completion(Message.assistant(x)) for x in "abc")
    cache.set("a", a)
    cache.set("b", b)
    assert cache.get("a") is a  # a is now the most recently used
    cache.set("c", c)
    assert len(cache) == 2 and cache.evictions == 1, f"max_entries=2: adding a third evicts one; len={len(cache)}, evictions={cache.evictions}"
    assert cache.get("b") is None, "b was the least recently USED (a was read after b was written), so b goes. LRU, not FIFO."
    assert cache.get("a") is a and cache.get("c") is c


def test_answers_are_never_shared_across_users():
    """The documented caveat: anything that makes the right answer differ per caller belongs in the key."""
    llm = ScriptedLLM(["Your ledger shows three torches.", "Your ledger shows no torches at all."])
    cache = room.ResponseCache(ttl_seconds=60.0, max_entries=8, clock=FakeClock())
    ask = [Message.user("How many torches are on my ledger?")]
    wren = room.cached_complete(llm, cache, ask, user_id="wren")
    tam = room.cached_complete(llm, cache, ask, user_id="tam")
    assert len(llm.calls) == 2, f"Two users, two model calls: you made {len(llm.calls)}. A cached answer served to the wrong user is a data leak."
    assert wren.text != tam.text
    assert cache.key_for(ask, {"user_id": "wren"}) != cache.key_for(ask, {"user_id": "tam"})
    again = room.cached_complete(llm, cache, ask, user_id="wren")
    assert again.text == wren.text and len(llm.calls) == 2, "The same user asking again is a hit."


def test_unfinished_replies_are_not_cached():
    llm = ScriptedLLM([Completion(Message.assistant("Nine f"), stop_reason="max_tokens"), "Nine feet.", "Nine feet."])
    cache = room.ResponseCache(ttl_seconds=60.0, max_entries=8, clock=FakeClock())
    cut = room.cached_complete(llm, cache, moat_question(), max_tokens=2)
    assert cut.stop_reason == "max_tokens" and len(cache) == 0, "A reply cut at max_tokens is not an answer; do not remember it."
    whole = room.cached_complete(llm, cache, moat_question(), max_tokens=2)
    assert whole.text == "Nine feet." and len(llm.calls) == 2, "After a miss on the cut reply, the model is asked again."
    room.cached_complete(llm, cache, moat_question(), max_tokens=2)
    assert len(llm.calls) == 2, "The finished reply IS cached: the third request is a hit."


# --------------------------------------------------------------- OutputPolicy
def test_the_veil_lets_clean_text_through():
    text, report = room.OutputPolicy().apply("The moat is nine feet deep and the bucket still leaks.")
    assert text == "The moat is nine feet deep and the bucket still leaks."
    assert report.blocked is False and report.secrets == [] and report.findings == []


def test_the_veil_redacts_pii_before_the_reply_leaves():
    reply = f"The treasurer, imp@example.invalid, paid with {GOOD_CARD}."
    text, report = room.OutputPolicy().apply(reply)
    assert text == "The treasurer, [EMAIL], paid with [CARD].", f"PII must be redacted on the way out; got {text!r}"
    assert report.blocked is False
    assert kinds(report.findings) == ["email", "card"]
    assert GOOD_CARD not in text and "4111" not in " ".join(str(f) for f in report.findings), (
        "Neither the reply nor the report may carry the card number."
    )


def test_the_veil_blocks_a_reply_that_contains_credentials():
    reply = f"The config says the key is {FAKE_AKIA}; mail imp@example.invalid if it stops working."
    text, report = room.OutputPolicy().apply(reply)
    assert report.blocked is True, "A reply containing a credential is withheld, not redacted: a key with one character hidden is still a leak."
    assert text == room.BLOCKED_TEXT, f"The blocked reply is BLOCKED_TEXT, got {text!r}"
    assert FAKE_AKIA not in text
    assert report.secrets == ["aws_access_key_id"]
    assert report.findings == [], "Nothing was redacted: there is no text left to redact."


def test_the_veil_can_be_configured():
    reply = f"Mail imp@example.invalid; the key is {FAKE_AKIA}."
    text, report = room.OutputPolicy(redact=False, block_secrets=False).apply(reply)
    assert text == reply, "With both switches off the veil is transparent."
    assert report.secrets == ["aws_access_key_id"] and report.blocked is False, (
        "Even when not blocking, the report says what it saw: an honest report is the point of having one."
    )
    text, report = room.OutputPolicy(redact=True, block_secrets=False).apply(reply)
    assert text == f"Mail [EMAIL]; the key is {FAKE_AKIA}." and kinds(report.findings) == ["email"], (
        f"redact on, blocking off: the email goes, the key (reported, not blocked) stays. Got {text!r}"
    )
