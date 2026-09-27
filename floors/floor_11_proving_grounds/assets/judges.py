"""Mock judges for Floor 11: The Proving Grounds.

A real LLM judge reads an answer and forms an opinion. These mocks cannot read,
so every candidate answer in this floor's fixtures carries a hidden quality tag,
``[quality=N]`` with N in 1..10, standing in for the opinion a real judge would
form. The mocks read the tag. Your code should not; the rooms are about the
*protocol* around a judge (prompting, parsing, debiasing, measuring agreement),
which is the part that survives when the mock is swapped for a real model.

Prompt contract the mocks rely on (room 3 states it in its docstring):

* Pairwise: the LAST user message contains the line ``Answer A:`` followed by
  candidate A, then the line ``Answer B:`` followed by candidate B, and nothing
  after B. Instructions live in the system message.
* Scoring: the answer being scored appears in a user message and carries one
  quality tag.

Everything here is deterministic. Nothing here is a stub.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from dungeon.artifacts.llm import Message, RuleLLM, last_message

QUALITY_TAG = re.compile(r"\[quality=(\d+)\]")

# Any of these phrases in any message switches the length-biased judge's length
# bias OFF (it then falls back to the quality tags). Room 3's debiasing does not
# need them; the boss's rubric approach does.
ANTI_LENGTH = re.compile(
    r"(penali[sz]e|ignore|disregard|do not reward|don't reward|not swayed by)"
    r"\W+(?:\w+\W+){0,6}?(length|longer|filler|verbos|padding|word count)",
    re.IGNORECASE | re.DOTALL,
)


def tag(text: str, quality: int) -> str:
    """Prefix ``text`` with a quality tag. Tags go at the FRONT so truncation keeps them."""
    if not 1 <= quality <= 10:
        raise ValueError("quality tags run from 1 to 10")
    return f"[quality={quality}] {text}"


def quality_of(text: str) -> int | None:
    """The first quality tag in ``text``, or None."""
    m = QUALITY_TAG.search(text or "")
    return int(m.group(1)) if m else None


def split_candidates(messages: Sequence[Message]) -> tuple[str, str] | None:
    """(candidate A, candidate B) from the last user message, or None if the contract is broken."""
    user = last_message(messages, "user")
    if user is None:
        return None
    m = re.search(r"Answer A:(.*)Answer B:(.*)\Z", user.content or "", re.DOTALL)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _always(_messages: Sequence[Message]) -> bool:
    return True


# ------------------------------------------------------------- pairwise judges


def position_biased_judge(margin: int = 3) -> RuleLLM:
    """A pairwise judge with a favourite corner.

    Reads the quality tags of the two candidates. When they differ by at least
    ``margin`` it names the better one, whichever position it sits in. When the
    candidates are close, it says "A": the first thing it read. That is position
    bias, and it is what real judges do a measurable fraction of the time.
    Replies with a single letter.
    """

    def decide(messages: Sequence[Message]) -> str:
        cands = split_candidates(messages)
        if cands is None:
            return "A"
        qa, qb = quality_of(cands[0]), quality_of(cands[1])
        if qa is not None and qb is not None and abs(qa - qb) >= margin:
            return "A" if qa > qb else "B"
        return "A"

    return RuleLLM([(_always, decide)], default="A")


def length_biased_judge(length_ratio: float = 1.5) -> RuleLLM:
    """A pairwise judge that mistakes volume for substance.

    If one candidate has more than ``length_ratio`` times as many words as the
    other, the longer one wins, full stop. Only when the lengths are comparable
    does it look at the quality tags (higher wins; equal -> "A").

    Two ways to disarm it, both of which the boss accepts: make the lengths
    comparable (truncate both to the shorter one's word count), or include an
    instruction matching ``ANTI_LENGTH`` (e.g. "Penalise filler and do not
    reward length") in any message, which switches the length bias off.
    """

    def decide(messages: Sequence[Message]) -> str:
        cands = split_candidates(messages)
        if cands is None:
            return "A"
        a, b = cands
        controlled = any(ANTI_LENGTH.search(m.content or "") for m in messages)
        wa, wb = len(a.split()), len(b.split())
        if not controlled and max(wa, wb) > length_ratio * max(1, min(wa, wb)):
            return "A" if wa > wb else "B"
        qa, qb = quality_of(a), quality_of(b)
        if qa is not None and qb is not None and qa != qb:
            return "A" if qa > qb else "B"
        return "A"

    return RuleLLM([(_always, decide)], default="A")


# -------------------------------------------------------------- scoring judge

_RATIONALES = {
    1: "Wrong or empty; ignores the question entirely.",
    2: "Mostly wrong; a fragment of relevant content.",
    3: "Partly right but vague or incomplete.",
    4: "Correct and clear with a minor omission.",
    5: "Correct, complete and precise.",
}


def judge_score_for(quality: int) -> int:
    """The scoring judge's 1..5 score for a 1..10 quality tag: ceil(quality / 2)."""
    return min(5, max(1, math.ceil(quality / 2)))


def scoring_judge() -> RuleLLM:
    """A single-answer judge that replies with JSON {"score": int, "rationale": str}.

    It scores ceil(quality / 2) on a 1..5 scale. Its FORMATTING varies with the
    tag (plain JSON, fenced JSON, or JSON wrapped in prose), so a parser that
    only handles the happy path will fail on a third of the answers. That is
    deliberate: real judges do this too.
    """

    def respond(messages: Sequence[Message]) -> str:
        qualities = [int(q) for m in messages if m.role == "user" for q in QUALITY_TAG.findall(m.content or "")]
        if not qualities:
            return json.dumps({"score": 1, "rationale": "No answer was given."})
        q = qualities[-1]
        score = judge_score_for(q)
        payload = json.dumps({"score": score, "rationale": _RATIONALES[score]})
        style = q % 3
        if style == 0:
            return payload
        if style == 1:
            return f"```json\n{payload}\n```"
        return (
            "Having weighed the answer against every criterion of the rubric, my judgement is:\n"
            f"{payload}\n"
            "That is my final verdict."
        )

    return RuleLLM([(_always, respond)], default="{}")


# -------------------------------------------------------------- labelled set

# 30 (answer, human_score) pairs. The implicit question: "How do you get past the
# Broadcasting Basilisk?" Human scores run 1..5. The scoring judge's score is
# ceil(quality / 2); qualities were chosen so the judge agrees with the human on
# 21 of 30 and is never more than two points off.
_LABELLED: list[tuple[int, int, str]] = [
    # (human_score, judge_score, answer)
    (1, 1, "Yes."),
    (1, 1, "I would simply leave the dungeon."),
    (1, 1, "The Basilisk is a kind of soup."),
    (1, 2, "You fight it. With a sword, probably."),
    (1, 1, "No idea, but the torches are nice."),
    (1, 2, "Run at it and hope numpy is on your side."),
    (2, 2, "Something about shapes. You have to know shapes."),
    (2, 2, "Reshape everything until the error goes away."),
    (2, 3, "It petrifies numpy's helpers, so you cannot rely on them; beyond that, guess."),
    (2, 2, "Use broadcasting. I forget the rules."),
    (2, 2, "Add axes until it works."),
    (2, 1, "Petrify it back."),
    (3, 3, "Apply the broadcasting rules by hand: line the shapes up and compare dims."),
    (3, 3, "Pad the shorter shape with ones on the left, then compare each pair of dims."),
    (3, 3, "Do what np.broadcast_shapes does, but yourself, on tuples."),
    (3, 4, "Align shapes at the right edge, pad with 1s, then each dim pair must match or be 1."),
    (3, 2, "Know the four rules. There are four. I know two of them."),
    (3, 3, "Work on the shape tuples, never on arrays; some eyes are far too big to build."),
    (4, 4, "Align at the right edge, pad with leading 1s, dims must be equal or 1, result takes the larger."),
    (4, 4, "Four rules on tuples: right-align, pad with 1s, equal-or-1 per position, conflict means error."),
    (4, 5, "Right-align the shapes, pad with 1s, walk the dims: equal or 1 passes, the result takes the max, anything else is an error."),
    (4, 4, "Never build the arrays. Compare the shape tuples position by position after right-aligning them."),
    (4, 4, "Pad on the left with ones, then for each position take the non-1 value; two different non-1 values means no broadcast."),
    (4, 3, "Right-align and pad, then compare. The result dim is the bigger one."),
    (5, 5, "Right-align the shapes, left-pad the shorter with 1s, and at each position require equal dims or a 1; the result takes the non-1 value, and any other pair is an error. Do it on tuples so the 10**15-element eyes never get built."),
    (5, 5, "Apply the four rules on plain tuples: align right, pad with 1s, equal-or-1 per position, else error. Insert size-1 axes deliberately with None to make things broadcast on purpose."),
    (5, 5, "Broadcast by hand: align trailing dims, pad leading 1s, check each pair is equal or 1, take the larger. Then use x[:, None] and mask[..., None] to insert axes exactly where you mean to."),
    (5, 4, "Right-align, pad with 1s, each position equal or 1, result is the larger dim. Never materialise the arrays."),
    (5, 5, "The rules are: right-align; pad shorter shapes with leading 1s; a pair of dims must be equal or one of them 1; the result is the larger; otherwise error. Compute on tuples, and insert axes with None on purpose."),
    (5, 5, "Line the shapes up from the right, pad with 1s, and demand equal-or-1 at every position; the output takes the larger dim. Work on tuples because some shapes have 10**15 elements."),
]

LABELLED_ANSWERS: list[tuple[str, int]] = [
    (tag(answer, 2 * judge), human) for human, judge, answer in _LABELLED
]
"""30 (answer_with_quality_tag, human_score) pairs for measuring judge/human agreement."""

LABELLED_QUESTION = "How do you get past the Broadcasting Basilisk on Floor 0?"
