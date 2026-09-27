"""TRIAL 9.4 - THE RETRIEVAL RITE

A faithful librarian sits at the desk: it answers a question only when the
passage holding the answer is in front of it, and otherwise says it cannot
find it. Whether it answers is therefore a verdict on YOUR retrieval and prompt.
"""

import re

import pytest

from dungeon.artifacts.llm import RuleLLM, last_matches
from dungeon.trials import load_room
from floors.floor_09_library_of_echoes.assets.corpus import GOLDEN_QA, PASSAGES
from floors.floor_09_library_of_echoes.assets.embedder import embed

room = load_room(__file__, "room_4_retrieval_rite")
catalogue = load_room(__file__, "room_2_card_catalogue")

# Calibrated on the reference solution: the toy embedder puts the gold passage
# at rank 1 for all 12 golden questions. The bar leaves room for a different but
# still correct index implementation.
RETRIEVAL_BAR = 10
ANSWER_BAR = 10

_MARKER = re.compile(r"\[(\d+)\]")


def _chunks(passages=PASSAGES):
    """One chunk per passage: the simplest shelving there is."""
    return [room.Chunk(p.id, 0, 0, len(p.text), p.text) for p in passages]


def _retriever(passages=PASSAGES):
    return room.Retriever(embed, catalogue.FlatIndex(), _chunks(passages))


def _source_number_of(phrase, messages):
    """Which numbered source in the prompt contains `phrase`? None if absent."""
    text = "\n".join(m.content for m in messages)
    pieces = re.split(r"(?=\[\d+\])", text)
    for piece in pieces:
        m = _MARKER.match(piece)
        if m and phrase.lower() in piece.lower():
            return int(m.group(1))
    return None


def faithful_librarian():
    """Answers only from the sources in front of it; cites the one it used."""

    def responder(qa):
        def respond(messages):
            n = _source_number_of(qa.answer, messages)
            return f"{qa.answer} [{n}]" if n is not None else room.CANNOT_FIND

        return respond

    rules = [(last_matches(re.escape(qa.question)), responder(qa)) for qa in GOLDEN_QA]
    return RuleLLM(rules, default=room.CANNOT_FIND)


# ------------------------------------------------------------------ Retriever
def test_the_retriever_returns_chunks_with_scores_best_first():
    retriever = _retriever()
    results = retriever.retrieve("Who guards the exit of the Threshold?", 5)
    assert len(results) == 5, f"k=5 but {len(results)} results came back."
    for item in results:
        assert isinstance(item, tuple) and len(item) == 2, "retrieve() returns (Chunk, score) pairs."
        chunk, score = item
        assert isinstance(chunk, room.Chunk), f"The first element must be the Chunk itself, not {type(chunk).__name__}."
        assert isinstance(score, float), f"The score should be a Python float, got {type(score).__name__}."
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), f"Results must be best first; got scores {scores}."


def test_retrieval_at_3_finds_the_gold_passage_for_most_golden_questions():
    retriever = _retriever()
    misses = []
    for qa in GOLDEN_QA:
        top = [c.doc_id for c, _ in retriever.retrieve(qa.question, 3)]
        if qa.passage_id not in top:
            misses.append((qa.question, qa.passage_id, top))
    hits = len(GOLDEN_QA) - len(misses)
    assert hits >= RETRIEVAL_BAR, (
        f"retrieval@3 found the gold passage for {hits}/{len(GOLDEN_QA)} questions; the bar is {RETRIEVAL_BAR}. Misses: {misses}. "
        "Are you embedding the query with the same embedder as the chunks, and resolving ids back to the right chunks?"
    )


# --------------------------------------------------------------- build_prompt
def test_the_prompt_opens_with_the_three_instructions():
    retrieved = _retriever().retrieve("What does the Overfit Hydra grow?", 3)
    messages = room.build_prompt("What does the Overfit Hydra grow?", retrieved)
    assert messages and messages[0].role == "system", f"The first message must be the system instruction, got role {messages[0].role!r}."
    system = messages[0].content.lower()
    for phrase, why in [
        ("only from the numbered sources", "the model must not answer from memory"),
        ("[n]", "citations must be machine-readable markers"),
        ("cannot find", "the model must be told what to say when the answer is absent"),
    ]:
        assert phrase in system, f"The system instruction lacks {phrase!r} ({why}). SYSTEM_INSTRUCTION has all three."


def test_the_prompt_numbers_every_source_and_ends_with_the_question():
    question = "What does the Overfit Hydra grow?"
    retrieved = _retriever().retrieve(question, 4)
    messages = room.build_prompt(question, retrieved)
    joined = "\n".join(m.content for m in messages)
    for n, (chunk, _) in enumerate(retrieved, start=1):
        assert f"[{n}] {chunk.text}" in joined, (
            f"Source {n} should appear as '[{n}] <chunk text>' on its own line. Could not find it for chunk {chunk.id}."
        )
    assert "[5]" not in joined, "Four sources were retrieved; there should be no [5]."
    assert messages[-1].role == "user", f"The last message carries the question and must be from the user, got {messages[-1].role!r}."
    assert question in messages[-1].content, "The question must be in the last message."
    assert joined.rfind(question) > joined.rfind("[4]"), "Sources first, then the question."


# ------------------------------------------------------------ parse_citations
def test_parse_citations_reads_the_markers_in_order_without_repeats():
    assert room.parse_citations("The gaze petrifies helpers [2]. It is weak to the rules [1], as [2] says.") == [2, 1]
    assert room.parse_citations("Two sources agree [1, 3].") == [1, 3], "Comma lists inside one bracket count too."
    assert room.parse_citations("No markers here.") == []
    assert room.parse_citations("[10] beats [9]") == [10, 9], "Multi-digit markers are numbers, not digits."
    assert all(isinstance(n, int) for n in room.parse_citations("[1][2]")), "Return ints."


# ---------------------------------------------------------------- RAGPipeline
def test_the_rite_answers_golden_questions_and_cites_the_gold_chunk():
    pipeline = room.RAGPipeline(_retriever(), faithful_librarian(), k=3)
    failures = []
    for qa in GOLDEN_QA:
        result = pipeline.answer(qa.question)
        gold_id = f"{qa.passage_id}#0"
        if qa.answer not in result["answer"] or gold_id not in result["citations"]:
            failures.append((qa.question, result["answer"], result["citations"]))
    good = len(GOLDEN_QA) - len(failures)
    assert good >= ANSWER_BAR, (
        f"{good}/{len(GOLDEN_QA)} golden questions were answered with the gold citation; the bar is {ANSWER_BAR}. "
        f"Failures: {failures}. The librarian only answers when the gold passage is in its prompt and only cites "
        "the [n] it read it under; the citation must resolve to that chunk's id."
    )


def test_citations_resolve_to_real_chunk_ids_among_the_sources():
    pipeline = room.RAGPipeline(_retriever(), faithful_librarian(), k=3)
    result = pipeline.answer(GOLDEN_QA[0].question)
    assert set(result) >= {"answer", "citations", "sources"}, f"answer() must return answer, citations and sources; got keys {sorted(result)}."
    assert len(result["sources"]) == 3 and all(isinstance(c, room.Chunk) for c in result["sources"]), (
        "sources must be the 3 retrieved Chunk objects, in [1]..[3] order."
    )
    source_ids = {c.id for c in result["sources"]}
    assert result["citations"], "The faithful librarian cited a source; citations should not be empty."
    assert set(result["citations"]) <= source_ids, (
        f"Citations {result['citations']} are not all ids of the retrieved sources {sorted(source_ids)}. "
        "Resolve [n] to sources[n - 1].id."
    )


def test_the_rite_respects_k():
    llm = faithful_librarian()
    pipeline = room.RAGPipeline(_retriever(), llm, k=3)
    result = pipeline.answer(GOLDEN_QA[1].question, k=5)
    assert len(result["sources"]) == 5, f"k=5 was requested; {len(result['sources'])} sources came back."
    prompt = "\n".join(m.content for m in llm.calls[-1]["messages"])
    assert "[5]" in prompt and "[6]" not in prompt, "The prompt should number exactly the 5 retrieved sources."
    result3 = pipeline.answer(GOLDEN_QA[1].question)
    assert len(result3["sources"]) == 3, "Without an override, the pipeline's default k applies."


def test_a_missing_book_gets_an_honest_refusal_not_a_guess():
    qa = GOLDEN_QA[4]  # the Babel Golem
    without_gold = [p for p in PASSAGES if p.id != qa.passage_id]
    pipeline = room.RAGPipeline(_retriever(without_gold), faithful_librarian(), k=3)
    result = pipeline.answer(qa.question)
    assert "cannot find" in result["answer"].lower(), (
        f"The gold passage is not in the library, so the faithful librarian should say it cannot find it. Got: {result['answer']!r}"
    )
    assert result["citations"] == [], f"A refusal cites nothing; got {result['citations']}."


def test_markers_pointing_outside_the_sources_are_dropped_quietly():
    llm = RuleLLM([], default="Confidently stated [7]. Also [2].")
    pipeline = room.RAGPipeline(_retriever(), llm, k=3)
    result = pipeline.answer("What is the loot of Floor 9?")
    assert result["citations"] == [result["sources"][1].id], (
        f"[7] points outside 3 sources and must be dropped; [2] resolves to the second source. Got {result['citations']}."
    )


@pytest.mark.parametrize("k", [1, 2])
def test_small_k_still_numbers_from_one(k):
    retrieved = _retriever().retrieve("Where is progress stored?", k)
    joined = "\n".join(m.content for m in room.build_prompt("Where is progress stored?", retrieved))
    assert "[1] " in joined and f"[{k + 1}]" not in joined
