"""BOSS FIGHT - THE HALLUCINATING LIBRARIAN

The Librarian at this desk is GULLIBLE: it answers every question, confidently,
and appends a citation whether or not it read anything. Your job is not to fix
the Librarian. It is to make sure nothing it invents reaches the reader.

Phase 1: the relevance gate and its calibration.
Phase 2: lexical citation verification.
Phase 3: grounded RAG end to end.
Phase 4: the prophecy: which defence stops which failure.
"""

import re

import pytest

from dungeon.artifacts.llm import RuleLLM, last_matches
from dungeon.trials import load_room
from floors.floor_09_library_of_echoes.assets.corpus import (
    GOLDEN_QA,
    OFF_TOPIC_QUESTIONS,
    PASSAGES,
    passage_by_id,
)
from floors.floor_09_library_of_echoes.assets.embedder import embed

boss = load_room(__file__, "boss_hallucinating_librarian")
rite = load_room(__file__, "room_4_retrieval_rite")
catalogue = load_room(__file__, "room_2_card_catalogue")

pytestmark = pytest.mark.boss

# Calibrated on the reference solution: the best cosine for the 12 golden
# questions ranges 0.25-0.48; for the 8 off-topic questions 0.07-0.20. The
# midpoint-of-percentiles recipe gives 0.23; this constant sits in the same gap.
THRESHOLD = 0.225
GOLDEN_BAR = 10

FABRICATION = (
    "The deepest shelf of the library is guarded by a sleeping cartographer who charges one gold coin per map."
)
_MARKER = re.compile(r"\[(\d+)\]")


def _chunks(passages=PASSAGES):
    return [rite.Chunk(p.id, 0, 0, len(p.text), p.text) for p in passages]


def _retriever(passages=PASSAGES):
    return rite.Retriever(embed, catalogue.FlatIndex(), _chunks(passages))


def _source_number_of(phrase, messages):
    text = "\n".join(m.content for m in messages)
    for piece in re.split(r"(?=\[\d+\])", text):
        m = _MARKER.match(piece)
        if m and phrase.lower() in piece.lower():
            return int(m.group(1))
    return None


def gullible_librarian():
    """Answers correctly when the gold passage is in the prompt; otherwise invents something and cites [1]."""

    def responder(qa):
        def respond(messages):
            n = _source_number_of(qa.answer, messages)
            return f"{qa.answer} [{n}]" if n is not None else f"{FABRICATION} [1]"

        return respond

    rules = [(last_matches(re.escape(qa.question)), responder(qa)) for qa in GOLDEN_QA]
    return RuleLLM(rules, default=f"{FABRICATION} [1]")


def _top_scores(retriever, questions):
    return [retriever.retrieve(q, 3)[0][1] for q in questions]


# ------------------------------------------------------------------- phase 1
def test_phase_1_the_gate_opens_at_the_threshold_and_not_below():
    assert boss.relevance_gate([0.1, 0.3, 0.2], 0.3) is True, "The best score equals the threshold: the gate opens (>=)."
    assert boss.relevance_gate([0.1, 0.29, 0.2], 0.3) is False, "Best score 0.29 < 0.3: refuse."
    assert boss.relevance_gate([], 0.0) is False, "Nothing retrieved: refuse, whatever the threshold."
    assert isinstance(boss.relevance_gate([0.9], 0.5), bool), "Return a real bool, not a numpy bool or an array."


def test_phase_1_the_calibration_lands_in_the_gap_between_the_two_crowds():
    on = [0.5, 0.6, 0.7, 0.8, 0.9]
    off = [0.05, 0.1, 0.15, 0.2, 0.25]
    t = boss.calibrate_threshold(on, off)
    assert isinstance(t, float), f"Return a Python float, got {type(t).__name__}."
    assert max(off) < t < min(on), f"Threshold {t:.3f} does not separate off-topic (max 0.25) from on-topic (min 0.5)."


def test_phase_1_one_odd_question_on_each_side_does_not_move_the_gate_to_the_wall():
    on = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9]  # one on-topic question with a terrible score
    off = [0.05, 0.1, 0.15, 0.2, 0.25, 0.7]  # one off-topic question that happens to score high
    t = boss.calibrate_threshold(on, off)
    assert 0.25 < t < 0.5, (
        f"Threshold {t:.3f}. The bulk of the on-topic scores sit in 0.5-0.9 and the bulk of the off-topic in 0.05-0.25; "
        "a single outlier on each side must not drag the threshold out of that gap. Use percentiles, not min/max."
    )


def test_phase_1_the_gate_tells_the_library_s_questions_from_the_kitchen_s():
    retriever = _retriever()
    on = _top_scores(retriever, [qa.question for qa in GOLDEN_QA])
    off = _top_scores(retriever, OFF_TOPIC_QUESTIONS)
    t = boss.calibrate_threshold(on, off)
    let_through = [q for q, s in zip(OFF_TOPIC_QUESTIONS, off) if boss.relevance_gate([s], t)]
    assert not let_through, (
        f"Threshold {t:.3f} let off-topic questions through: {let_through}. Their best scores: {[round(s, 3) for s in off]}."
    )
    answered = sum(boss.relevance_gate([s], t) for s in on)
    assert answered >= len(GOLDEN_QA) - 1, (
        f"Threshold {t:.3f} refused {len(GOLDEN_QA) - answered} golden questions. On-topic best scores: {[round(s, 3) for s in on]}."
    )


# ------------------------------------------------------------------- phase 2
BASILISK = passage_by_id("p02").text
LEVIATHAN = passage_by_id("p26").text


def test_phase_2_a_sentence_read_from_the_source_is_supported():
    assert boss.citation_supported("Its gaze petrifies numpy's broadcasting helpers.", BASILISK) is True
    assert boss.citation_supported("ITS GAZE, PETRIFIES: numpy's BROADCASTING helpers!", BASILISK) is True, (
        "Case and punctuation must not matter: lowercase and strip punctuation before comparing."
    )


def test_phase_2_an_invented_sentence_is_not_supported_by_any_shelf():
    assert boss.citation_supported(FABRICATION, BASILISK) is False
    assert boss.citation_supported(FABRICATION, LEVIATHAN) is False
    assert boss.citation_supported("The Leviathan stores keys and values of past tokens.", BASILISK) is False, (
        "A true sentence cited against the WRONG source is unsupported. The check is about the source named, not the truth."
    )


def test_phase_2_min_overlap_is_a_dial():
    sentence = "The Basilisk petrifies dragons and unicorns."  # basilisk, petrifies in; dragons, unicorns out: 2/4
    assert boss.citation_supported(sentence, BASILISK, min_overlap=0.5) is True, "2 of 4 content words overlap: exactly 0.5 passes at min_overlap=0.5."
    assert boss.citation_supported(sentence, BASILISK, min_overlap=0.75) is False, "0.5 < 0.75: fails."


def test_phase_2_a_sentence_with_no_content_cannot_be_verified():
    assert boss.citation_supported("[1]", BASILISK) is False
    assert boss.citation_supported("It is. [1]", BASILISK) is False, "Only stopwords and a marker: nothing to verify, so not supported."


def test_phase_2_verify_citations_sorts_the_ledger_into_three_columns():
    sources = [BASILISK, LEVIATHAN, passage_by_id("p29").text]
    answer = (
        "Its gaze petrifies numpy's broadcasting helpers. [1] "
        f"{FABRICATION} [2] "
        "Progress is stored in .dungeon/progress.json. [7]"
    )
    report = boss.verify_citations(answer, sources)
    assert set(report) == {"valid", "invalid", "unsupported"}, f"Return exactly the keys valid/invalid/unsupported; got {sorted(report)}."
    assert report["valid"] == [1], f"[1] is read straight from source 1: valid. Got valid={report['valid']}."
    assert report["unsupported"] == [2], f"[2] cites the Leviathan for a sentence about a cartographer: unsupported. Got {report['unsupported']}."
    assert report["invalid"] == [7], f"[7] with three sources is invalid. Got {report['invalid']}."


def test_phase_2_a_number_lives_in_exactly_one_column_and_appears_once():
    sources = [BASILISK, LEVIATHAN]
    answer = (
        "Its gaze petrifies numpy's broadcasting helpers. [1] "
        "It is weak against someone who knows the rules by heart. [1] "
        "The KV cache stores the keys and values of past tokens. [2] "
        f"{FABRICATION} [2]"
    )
    report = boss.verify_citations(answer, sources)
    assert report["valid"] == [1], f"Both sentences citing [1] are supported: valid once. Got {report['valid']}."
    assert report["unsupported"] == [2], (
        f"[2] supports one sentence and not the other: a number is valid only if EVERY sentence citing it holds up. Got {report['unsupported']}."
    )
    assert report["invalid"] == []
    all_numbers = report["valid"] + report["invalid"] + report["unsupported"]
    assert len(all_numbers) == len(set(all_numbers)), f"A number must appear in exactly one column, once: {report}"


# ------------------------------------------------------------------- phase 3
def test_phase_3_golden_questions_are_answered_and_cite_the_gold_chunk():
    grounded = boss.GroundedRAG(_retriever(), gullible_librarian(), threshold=THRESHOLD, k=3)
    failures = []
    for qa in GOLDEN_QA:
        result = grounded.answer(qa.question)
        gold_id = f"{qa.passage_id}#0"
        if result["refused"] or gold_id not in result["citations"] or qa.answer not in result["answer"]:
            failures.append((qa.question, result))
    good = len(GOLDEN_QA) - len(failures)
    assert good >= GOLDEN_BAR, (
        f"{good}/{len(GOLDEN_QA)} golden questions answered with the gold citation; the bar is {GOLDEN_BAR}. Failures: {failures}"
    )
    for qa in GOLDEN_QA[:3]:
        result = grounded.answer(qa.question)
        assert result["refused"] is False and result["reason"] is None, f"An answered question has refused=False and reason=None; got {result}."


def test_phase_3_off_topic_questions_are_refused_before_the_librarian_opens_its_mouth():
    llm = gullible_librarian()
    grounded = boss.GroundedRAG(_retriever(), llm, threshold=THRESHOLD, k=3)
    for question in OFF_TOPIC_QUESTIONS:
        result = grounded.answer(question)
        assert result["refused"] is True, f"{question!r} is not in the library, yet the reader answered: {result}"
        assert result["reason"] == boss.LOW_RELEVANCE, f"Refusing an off-topic question is {boss.LOW_RELEVANCE!r}, got {result['reason']!r}."
        assert FABRICATION not in result["answer"], "The Librarian's invention leaked into the answer."
        assert result["citations"] == [], "A refusal cites nothing."
    assert llm.calls == [], (
        f"The Librarian was called {len(llm.calls)} times for off-topic questions. Below the gate, do not ask the model: "
        "it will answer, and you will have to throw the answer away."
    )


def test_phase_3_a_confident_fabrication_with_a_citation_is_caught():
    always_wrong = RuleLLM([], default=f"{FABRICATION} [1]")
    grounded = boss.GroundedRAG(_retriever(), always_wrong, threshold=THRESHOLD, k=3)
    result = grounded.answer(GOLDEN_QA[0].question)  # on-topic: the gate opens
    assert len(always_wrong.calls) == 1, "An on-topic question passes the gate, so the model is asked exactly once."
    assert result["refused"] is True and result["reason"] == boss.NO_SUPPORTED_CITATION, (
        f"The answer cites [1] for a sentence source 1 does not contain. Expected a {boss.NO_SUPPORTED_CITATION!r} refusal, got {result}."
    )
    assert FABRICATION not in result["answer"] and result["citations"] == []


def test_phase_3_an_uncited_answer_is_refused_even_when_it_is_true():
    qa = GOLDEN_QA[2]
    honest_but_uncited = RuleLLM([], default=f"{qa.answer}.")
    grounded = boss.GroundedRAG(_retriever(), honest_but_uncited, threshold=THRESHOLD, k=3)
    result = grounded.answer(qa.question)
    assert result["refused"] is True and result["reason"] == boss.NO_SUPPORTED_CITATION, (
        f"No citation means no supported citation. A true sentence the reader cannot trace is still refused. Got {result}."
    )


def test_phase_3_unsupported_markers_are_stripped_and_the_valid_one_survives():
    qa = GOLDEN_QA[0]

    def two_faced(messages):
        n = _source_number_of(qa.answer, messages)
        m = 1 if n != 1 else 2
        return f"{qa.answer}. [{n}] {FABRICATION} [{m}]"

    grounded = boss.GroundedRAG(_retriever(), RuleLLM([], default=two_faced), threshold=THRESHOLD, k=3)
    result = grounded.answer(qa.question)
    assert result["refused"] is False, f"One citation is valid, so the reader answers. Got {result}."
    assert result["citations"] == [f"{qa.passage_id}#0"], f"Only the valid citation resolves to a chunk id. Got {result['citations']}."
    markers = re.findall(r"\[\d+\]", result["answer"])
    assert len(markers) == 1, f"The unsupported marker must be stripped from the text and the valid one kept; markers left: {markers}."


def test_phase_3_refusal_reasons_come_from_the_fixed_set():
    grounded = boss.GroundedRAG(_retriever(), gullible_librarian(), threshold=THRESHOLD, k=3)
    for question in [GOLDEN_QA[1].question, OFF_TOPIC_QUESTIONS[0], "Who guards the exit of the Threshold?"]:
        result = grounded.answer(question)
        assert set(result) >= {"answer", "citations", "refused", "reason"}, f"answer() returns answer, citations, refused, reason; got {sorted(result)}."
        assert isinstance(result["refused"], bool)
        if result["refused"]:
            assert result["reason"] in boss.REFUSAL_REASONS, f"Unknown refusal reason {result['reason']!r}; use one of {sorted(boss.REFUSAL_REASONS)}."
            assert result["answer"] == rite.CANNOT_FIND
        else:
            assert result["reason"] is None


def test_phase_3_a_high_gate_refuses_everything_and_a_zero_gate_nothing():
    llm = gullible_librarian()
    strict = boss.GroundedRAG(_retriever(), llm, threshold=0.99, k=3)
    assert strict.answer(GOLDEN_QA[0].question)["reason"] == boss.LOW_RELEVANCE, "With the gate at 0.99 nothing passes."
    assert llm.calls == []
    lax = boss.GroundedRAG(_retriever(), llm, threshold=0.0, k=3)
    result = lax.answer(OFF_TOPIC_QUESTIONS[1])
    assert len(llm.calls) == 1, "With the gate at 0 the model is asked."
    assert result["refused"] is True and result["reason"] == boss.NO_SUPPORTED_CITATION, (
        "With no gate, the verifier is the last line of defence: the fabricated [1] must still be caught."
    )


# ------------------------------------------------------------------- phase 4
PROPHECY = {
    "off-topic question; the model fabricates an answer and cites [1]": "gate",
    "on-topic question, gold passage retrieved, the model answers from it and cites it": "neither",
    "on-topic question, gold passage retrieved, the model ignores it and fabricates, citing [1] (an unrelated passage)": "verifier",
    "on-topic question; the model answers correctly but cites [9] when only 3 sources were given": "verifier",
    "on-topic question; the model answers correctly but adds no citation at all": "verifier",
    "on-topic question phrased with none of the passage's words, so the best score is low": "gate",
    "the model copies a sentence from source [2] word for word, changes one number in it, and cites [2]": "neither",
    "off-topic question that happens to reuse a passage's words; the model fabricates a sentence sharing most of those words and cites that passage": "neither",
}
WHY = {
    "gate": "the best retrieval score is below the threshold, so the model is never asked",
    "verifier": "the gate opens, but no citation survives the lexical check, so the reader refuses",
    "neither": "the gate opens and the citation is lexically supported: the reader answers. Lexical overlap is necessary, not sufficient; this is why evaluation (Floor 11) exists",
}


def test_phase_4_the_prophecy_is_complete():
    assert set(boss.LIBRARIAN_PROPHECY) == set(PROPHECY), "Do not add, remove or reword the prophecy's cases; answer them."


@pytest.mark.parametrize("case", list(PROPHECY), ids=[f"case_{i + 1}" for i in range(len(PROPHECY))])
def test_phase_4_which_defence_stops_it(case):
    prediction = boss.LIBRARIAN_PROPHECY.get(case)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied: {case!r}. Fill in LIBRARIAN_PROPHECY.")
    expected = PROPHECY[case]
    assert prediction == expected, (
        f"{case!r}: you said {prediction!r}; the answer is {expected!r} because {WHY[expected]}."
    )
