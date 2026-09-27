"""TRIAL 11.3 - THE JUDGE

Prompts that carry the rubric, a parser that survives decoration, a swap that
exposes the judge's favourite corner, and agreement with humans measured
honestly.
"""

import math

import pytest

from dungeon.artifacts.llm import Message, ScriptedLLM
from dungeon.trials import load_room
from floors.floor_11_proving_grounds.assets.judges import (
    LABELLED_ANSWERS,
    LABELLED_QUESTION,
    position_biased_judge,
    scoring_judge,
    tag,
)

room = load_room(__file__, "room_3_the_judge")

RUBRIC_CRITERIA = ["States the four broadcasting rules", "Mentions working on tuples", "Is concise"]
QUESTION = "How do you get past the Broadcasting Basilisk?"
ANSWER = "Right-align the shapes, pad with 1s, and demand equal-or-1 at every position."


def _rubric():
    return room.Rubric(criteria=list(RUBRIC_CRITERIA), scale=(1, 5))


def _text(messages, role):
    return "\n".join(m.content for m in messages if m.role == role)


# ---------------------------------------------------------------- judge_prompt
def test_the_judge_prompt_carries_the_rubric_the_scale_and_the_format():
    messages = room.judge_prompt(QUESTION, ANSWER, _rubric())
    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages), "Return a list of Message objects."
    assert messages[0].role == "system" and messages[-1].role == "user", "System message first, user message last."
    system = _text(messages, "system")
    user = _text(messages, "user")
    for criterion in RUBRIC_CRITERIA:
        assert criterion in system, f"The criterion {criterion!r} is missing from the system message. The judge cannot apply a rubric it was not shown."
    assert "1" in system and "5" in system, "State both ends of the scale."
    assert "score" in system and "rationale" in system and "JSON" in system.upper(), (
        'Ask for JSON with "score" and "rationale" keys; a free-text verdict is not machine-readable.'
    )
    assert QUESTION in user and ANSWER in user, "The user message must contain the question and the answer being judged."


# ------------------------------------------------------------- parse_judgement
@pytest.mark.parametrize(
    "reply",
    [
        '{"score": 4, "rationale": "Clear and mostly complete."}',
        '```json\n{"score": 4, "rationale": "Clear and mostly complete."}\n```',
        '```\n{"score": 4, "rationale": "Clear and mostly complete."}\n```',
        'Certainly! Here is my judgement:\n{"score": 4, "rationale": "Clear and mostly complete."}\nLet me know if you need more.',
        '{"rationale": "Clear and mostly complete.", "score": "4"}',
        '{"score": 4.0, "rationale": "Clear and mostly complete."}',
    ],
    ids=["bare", "fenced-json", "fenced-plain", "prose-wrapped", "string-score", "float-score"],
)
def test_the_parser_sees_through_every_decoration(reply):
    score, rationale = room.parse_judgement(reply)
    assert score == 4 and type(score) is int, f"Expected score 4 as an int from {reply!r}; got {score!r} ({type(score).__name__})."
    assert "mostly complete" in rationale, f"The rationale went missing: {rationale!r}."


def test_the_parser_refuses_a_reply_with_no_score():
    with pytest.raises(ValueError):
        room.parse_judgement("The answer was fine, I suppose.")


# -------------------------------------------------------------------- llm_judge
def test_llm_judge_returns_a_judgement():
    llm = ScriptedLLM(['```json\n{"score": 3, "rationale": "Partly right."}\n```'])
    judgement = room.llm_judge(llm, QUESTION, ANSWER, _rubric())
    assert isinstance(judgement, room.Judgement)
    assert judgement.score == 3 and judgement.rationale == "Partly right.", f"Got {judgement!r}."
    assert len(llm.calls) == 1, "One answer, one call."
    assert ANSWER in _text(llm.calls[0]["messages"], "user"), "The judge was not shown the answer."


def test_llm_judge_rejects_a_score_off_the_scale():
    llm = ScriptedLLM(['{"score": 9, "rationale": "Enthusiastic."}'])
    with pytest.raises(ValueError):
        room.llm_judge(llm, QUESTION, ANSWER, _rubric())


def test_the_scoring_judge_is_parsed_on_every_one_of_thirty_answers():
    judge = scoring_judge()
    scores = [room.llm_judge(judge, LABELLED_QUESTION, answer, _rubric()).score for answer, _ in LABELLED_ANSWERS]
    assert all(1 <= s <= 5 for s in scores), f"Every score must be on the 1..5 scale; got {scores}."
    assert len(set(scores)) > 1, "The judge gave every answer the same score; the parse is probably returning a default."


# ------------------------------------------------------------- pairwise_prompt
def test_the_pairwise_prompt_follows_the_contract():
    a, b = "the first candidate", "the second candidate"
    messages = room.pairwise_prompt(QUESTION, a, b)
    assert messages[0].role == "system" and messages[-1].role == "user"
    user = messages[-1].content
    assert "Answer A:" in user and "Answer B:" in user, "Label the candidates with the exact lines 'Answer A:' and 'Answer B:'."
    assert user.index("Answer A:") < user.index("Answer B:"), "A comes before B."
    assert user.split("Answer B:", 1)[1].strip() == b, "Nothing may follow candidate B in the user message; put instructions in the system message."
    assert a in user.split("Answer B:", 1)[0].split("Answer A:", 1)[1], "Candidate a must sit between the two labels."
    assert QUESTION in user
    assert "A" in messages[0].content and "B" in messages[0].content, "Tell the judge to reply with A or B."


def test_parse_choice_finds_a_lone_letter():
    assert room.parse_choice("B") == "B"
    assert room.parse_choice("  A\n") == "A"
    assert room.parse_choice("Verdict: B.") == "B"
    with pytest.raises(ValueError):
        room.parse_choice("Both are fine.")


# ------------------------------------------------------------------- debiasing
def test_the_judge_has_a_favourite_corner_and_swapping_exposes_it():
    judge = position_biased_judge()
    worse, better = tag("The answer is probably shapes.", 4), tag("Right-align and pad with ones, then compare dims.", 5)
    naive = room.pairwise_judge_naive(judge, QUESTION, worse, better)
    assert naive == "A", f"Sanity check on the fixture: the naive judge should pick the first candidate on a close pair; it said {naive!r}."
    judge = position_biased_judge()
    verdict = room.pairwise_judge_debiased(judge, QUESTION, worse, better)
    assert verdict == "tie", (
        f"The judge said A in both orders, i.e. it contradicted itself about which TEXT is better. "
        f"That is a tie, not a winner; you returned {verdict!r}."
    )
    assert len(judge.calls) == 2, f"Debiasing is exactly two calls: (a, b) and (b, a). You made {len(judge.calls)}."


def test_the_debiased_judge_swaps_the_candidates_on_the_second_call():
    judge = position_biased_judge()
    a, b = tag("alpha", 5), tag("beta", 5)
    room.pairwise_judge_debiased(judge, QUESTION, a, b)
    first = _text(judge.calls[0]["messages"], "user")
    second = _text(judge.calls[1]["messages"], "user")
    assert first.index(a) < first.index(b), "In the first call a should be Answer A."
    assert second.index(b) < second.index(a), "In the second call b should be Answer A. The judge must see both orders."


def test_a_clear_winner_survives_the_swap_in_either_corner():
    judge = position_biased_judge()
    bad, great = tag("Fight it with a sword.", 2), tag("Apply the four rules on tuples; never build the arrays.", 8)
    assert room.pairwise_judge_debiased(judge, QUESTION, bad, great) == "B", "b is clearly better: both orders agree, the verdict is 'B'."
    assert room.pairwise_judge_debiased(judge, QUESTION, great, bad) == "A", (
        "a is clearly better here. If you returned 'B', you forgot to translate the second call's verdict back: "
        "an 'A' from the swapped call means b won."
    )


# --------------------------------------------------------------------- kappa
def test_kappa_is_one_for_perfect_agreement_and_zero_at_chance():
    assert math.isclose(room.cohens_kappa([1, 2, 3, 1, 2], [1, 2, 3, 1, 2]), 1.0), "Identical labels: kappa is 1.0."
    k = room.cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1])
    assert math.isclose(k, 0.0, abs_tol=1e-12), (
        f"p_o = 0.5 and p_e = 0.5*0.5 + 0.5*0.5 = 0.5: agreement exactly at chance is kappa 0.0; you said {k:.4f}."
    )


def test_kappa_goes_negative_for_systematic_disagreement():
    k = room.cohens_kappa([0, 1, 0, 1], [1, 0, 1, 0])
    assert math.isclose(k, -1.0), f"Always disagreeing with balanced labels is kappa -1.0; you said {k:.4f}."


def test_kappa_by_hand():
    a = [1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
    b = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    # p_o = 7/10. a has six 1s, b has five: p_e = 0.6*0.5 + 0.4*0.5 = 0.5. kappa = 0.2/0.5 = 0.4.
    k = room.cohens_kappa(a, b)
    assert math.isclose(k, 0.4), f"p_o = 0.7, p_e = 0.5, kappa = 0.4; you said {k:.4f}."


def test_kappa_with_one_label_only_is_one():
    assert room.cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0, "Both raters always say 1: p_e is 1; define kappa as 1.0 here rather than dividing by zero."


# ------------------------------------------------------------------ spearman
def test_average_ranks_share_the_ranks_they_span():
    ranks = room.average_ranks([10, 20, 20, 30])
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0], f"[10, 20, 20, 30] ranks as [1, 2.5, 2.5, 4]; you said {list(ranks)}."
    ranks = room.average_ranks([5, 5, 5])
    assert list(ranks) == [2.0, 2.0, 2.0], f"Three-way tie: everyone gets the middle rank 2.0; you said {list(ranks)}."


def test_spearman_ignores_the_shape_of_a_monotone_curve():
    x = [1, 2, 3, 4, 5]
    assert math.isclose(room.spearman(x, [1, 4, 9, 16, 25]), 1.0), "Any increasing function of x has Spearman 1.0 with x."
    assert math.isclose(room.spearman(x, [25, 16, 9, 4, 1]), -1.0), "Any decreasing function has Spearman -1.0."


def test_spearman_by_hand_with_a_tie():
    # ranks of x: [1, 2.5, 2.5, 4]; ranks of y: [1, 2, 3, 4]; Pearson of those = 4.5 / sqrt(4.5 * 5).
    got = room.spearman([1, 2, 2, 3], [1, 2, 3, 4])
    expected = 4.5 / math.sqrt(22.5)
    assert math.isclose(got, expected, rel_tol=1e-9), (
        f"With average ranks for the tie, Spearman is {expected:.6f}; you said {got:.6f}. "
        "If you got 0.8 you gave the tied values ranks 2 and 3 instead of 2.5 and 2.5."
    )


def test_spearman_of_a_constant_is_zero_not_nan():
    assert room.spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


# ---------------------------------------------------------- agreement_report
def test_the_judge_agrees_with_humans_well_above_chance():
    judge = scoring_judge()
    rubric = _rubric()
    judge_scores = [room.llm_judge(judge, LABELLED_QUESTION, answer, rubric).score for answer, _ in LABELLED_ANSWERS]
    human_scores = [human for _, human in LABELLED_ANSWERS]
    report = room.agreement_report(judge_scores, human_scores)
    for key in ("n", "kappa", "spearman", "exact_agreement", "mean_abs_diff"):
        assert key in report, f"agreement_report is missing {key!r}."
    assert report["n"] == 30
    assert math.isclose(report["exact_agreement"], 0.7), f"The judge matches the human on 21 of 30 answers; you report {report['exact_agreement']:.3f}."
    assert math.isclose(report["kappa"], 0.625, abs_tol=1e-9), (
        f"p_o = 0.7 and p_e = 0.2 (five equally common human scores), so kappa = 0.5/0.8 = 0.625; you report {report['kappa']:.4f}."
    )
    assert math.isclose(report["spearman"], 0.9242531766, abs_tol=1e-6), (
        f"Spearman with average ranks on this set is 0.92425; you report {report['spearman']:.5f}."
    )
    assert math.isclose(report["mean_abs_diff"], 0.3), f"Nine disagreements of one point over 30 answers: mean |diff| = 0.3; you report {report['mean_abs_diff']:.3f}."


def test_agreement_report_refuses_unpaired_scores():
    with pytest.raises(ValueError):
        room.agreement_report([1, 2, 3], [1, 2])
