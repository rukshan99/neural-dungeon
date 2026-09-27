"""Golden evaluation cases for Floor 11, plus the champions that answer them.

The cases are plain dicts (id, input, expected, tags) so that this module does
not depend on the learner's ``EvalCase`` class; room 4's trial turns them into
``EvalCase`` objects. Three tags, eight cases each:

* ``arithmetic``: loot prices. Scored by ``contains`` on the number.
* ``lore``: facts about the dungeon. Scored by ``contains`` on a keyword.
* ``shapes``: Floor 0 shape questions. Scored by ``contains`` on the tuple.

The champion is a ``RuleLLM`` behind a ``system_fn(prompt) -> str`` face, the
same shape as any system you would evaluate for real. It answers 21 of 24
correctly (one wrong per tag) so that scores are informative rather than 1.0.
"""

from __future__ import annotations

from collections.abc import Callable

from dungeon.artifacts.llm import Message, RuleLLM

GOLDEN_CASES: list[dict] = [
    # ---------------------------------------------------------- arithmetic
    {"id": "arith_01", "input": "A healing potion costs 12 gold. What do 3 potions cost?", "expected": "36", "tags": ("arithmetic",)},
    {"id": "arith_02", "input": "A torch costs 3 gold and a rope costs 7 gold. What do 2 torches and 1 rope cost?", "expected": "13", "tags": ("arithmetic",)},
    {"id": "arith_03", "input": "The Basilisk's hoard holds 250 gold, split evenly among 5 adventurers. How much does each get?", "expected": "50", "tags": ("arithmetic",)},
    {"id": "arith_04", "input": "A shield costs 45 gold. You have 30 gold. How much more do you need?", "expected": "15", "tags": ("arithmetic",)},
    {"id": "arith_05", "input": "Arrows cost 2 gold each. How many arrows can you buy with 25 gold?", "expected": "12", "tags": ("arithmetic",)},
    {"id": "arith_06", "input": "A map costs 8 gold and the merchant offers a 25% discount. What do you pay?", "expected": "6", "tags": ("arithmetic",)},
    {"id": "arith_07", "input": "You sell 4 gorgon scales at 11 gold each and buy a lantern for 9 gold. How much gold do you leave with?", "expected": "35", "tags": ("arithmetic",)},
    {"id": "arith_08", "input": "A torch costs 5 gold on this floor and prices double on every floor below. What does it cost three floors down?", "expected": "40", "tags": ("arithmetic",)},
    # ---------------------------------------------------------------- lore
    {"id": "lore_01", "input": "Which floor number is The Threshold?", "expected": "0", "tags": ("lore",)},
    {"id": "lore_02", "input": "What creature guards The Threshold?", "expected": "Basilisk", "tags": ("lore",)},
    {"id": "lore_03", "input": "What does the Broadcasting Basilisk petrify?", "expected": "broadcasting", "tags": ("lore",)},
    {"id": "lore_04", "input": "Which floor number is The Proving Grounds?", "expected": "11", "tags": ("lore",)},
    {"id": "lore_05", "input": "Whose law is the Gorgon of the Proving Grounds named after?", "expected": "Goodhart", "tags": ("lore",)},
    {"id": "lore_06", "input": "According to the Gorgon's curse, what does a measure become when you optimise it?", "expected": "target", "tags": ("lore",)},
    {"id": "lore_07", "input": "How many broadcasting rules does The Threshold teach?", "expected": "4", "tags": ("lore",)},
    {"id": "lore_08", "input": "Which single incantation is permitted in the Einsum Oubliette?", "expected": "einsum", "tags": ("lore",)},
    # -------------------------------------------------------------- shapes
    {"id": "shape_01", "input": "What is the shape of np.zeros((2, 3)) @ np.zeros((3, 4))?", "expected": "(2, 4)", "tags": ("shapes",)},
    {"id": "shape_02", "input": "What is the shape of np.zeros((5,))[:, None]?", "expected": "(5, 1)", "tags": ("shapes",)},
    {"id": "shape_03", "input": "What is the shape of np.zeros((2, 3, 4)).sum(axis=1)?", "expected": "(2, 4)", "tags": ("shapes",)},
    {"id": "shape_04", "input": "What is the shape of np.zeros((3, 1, 4)) + np.zeros((5, 4))?", "expected": "(3, 5, 4)", "tags": ("shapes",)},
    {"id": "shape_05", "input": "What is the shape of np.zeros((6,)).reshape(2, -1)?", "expected": "(2, 3)", "tags": ("shapes",)},
    {"id": "shape_06", "input": "What is the shape of np.zeros((2, 3, 4)).transpose(2, 0, 1)?", "expected": "(4, 2, 3)", "tags": ("shapes",)},
    {"id": "shape_07", "input": "Are the shapes (3,) and (4,) compatible for broadcasting? Answer 'compatible' or 'incompatible'.", "expected": "incompatible", "tags": ("shapes",)},
    {"id": "shape_08", "input": "What is the shape of np.zeros((4, 3)).T?", "expected": "(3, 4)", "tags": ("shapes",)},
]

# One rule per question. Three are deliberately wrong (arith_05, lore_06, shape_04).
_CHAMPION_RULES: list[tuple[str, str]] = [
    (r"3 potions", "Three potions come to 36 gold."),
    (r"2 torches and 1 rope", "That is 6 + 7 = 13 gold."),
    (r"split evenly among 5", "Each adventurer gets 50 gold."),
    (r"shield costs 45", "You need 15 more gold."),
    (r"arrows can you buy", "You can buy 13 arrows."),  # wrong: 25 // 2 is 12
    (r"25% discount", "You pay 6 gold."),
    (r"gorgon scales", "44 in, 9 out: you leave with 35 gold."),
    (r"prices double", "5, 10, 20, 40: it costs 40 gold three floors down."),
    (r"floor number is The Threshold", "The Threshold is Floor 0."),
    (r"creature guards The Threshold", "The Broadcasting Basilisk guards it."),
    (r"Basilisk petrify", "It petrifies numpy's broadcasting helpers."),
    (r"floor number is The Proving Grounds", "The Proving Grounds are Floor 11."),
    (r"Gorgon .* named after", "Goodhart's law."),
    (r"measure become", "It becomes a number people argue about."),  # wrong: a target
    (r"how many broadcasting rules", "There are 4 rules."),
    (r"Einsum Oubliette", "Only einsum is permitted there."),
    (r"\(2, 3\)\) @", "The result has shape (2, 4)."),
    (r"\(5,\)\)\[:, None\]", "That gives shape (5, 1)."),
    (r"sum\(axis=1\)", "Axis 1 disappears: (2, 4)."),
    (r"\(3, 1, 4\)\) \+", "Those do not broadcast: (3, 4)."),  # wrong: (3, 5, 4)
    (r"reshape\(2, -1\)", "Six elements in two rows: (2, 3)."),
    (r"transpose\(2, 0, 1\)", "Old axis 2 goes first: (4, 2, 3)."),
    (r"\(3,\) and \(4,\)", "They are incompatible."),
    (r"\(4, 3\)\)\.T", "Transposed: (3, 4)."),
]

CHAMPION_WRONG_IDS = {"arith_05", "lore_06", "shape_04"}


def champion_llm() -> RuleLLM:
    """The honest champion as a RuleLLM. Default reply has no digits and no keywords."""
    return RuleLLM(_CHAMPION_RULES, default="The champion shrugs and says nothing useful.")


def champion() -> Callable[[str], str]:
    """The honest champion as a ``system_fn(prompt) -> str``. Right on 21 of the 24 golden cases."""
    llm = champion_llm()

    def answer(prompt: str) -> str:
        return llm.complete([Message.user(prompt)]).text

    return answer


def degraded_champion() -> Callable[[str], str]:
    """Yesterday's champion after a bad deploy: it lost the ability to count gold.

    Every question mentioning gold now gets a vague answer. Seven cases that
    passed before (all the arithmetic except arith_05, which was already wrong)
    now fail; the other tags are untouched. Use it to test regression detection.
    """
    honest = champion()

    def answer(prompt: str) -> str:
        if "gold" in prompt.lower():
            return "Some gold. A reasonable amount of gold."
        return honest(prompt)

    return answer
