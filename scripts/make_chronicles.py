#!/usr/bin/env python
"""Generate the dungeon's training corpora (deterministic, seed 0).

Two texts are produced into dungeon/artifacts/:

  chronicles.txt     "The Dungeon Chronicles": prose about adventurers, floors,
                     bosses and loot. ~180 KB. The tiny GPT on Floor 7 trains on
                     this, and the shipped checkpoint was trained on it.

  goblin_ledger.txt  "The Goblin Quartermaster's Ledger": terse inventory lines
                     full of digits, pipes and colons. ~45 KB. A deliberately
                     different character distribution, used on Floor 8 to show
                     fine-tuning and catastrophic forgetting.

The corpora are checked in; this script exists so they can be regenerated and
so nobody wonders where they came from. Run from the repo root:

    python scripts/make_chronicles.py
"""

from __future__ import annotations

import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "dungeon" / "artifacts"

# --------------------------------------------------------------------------- vocab
NAMES = [
    "Ilva", "Tomas", "Brannoc", "Sefa", "Oduya", "Pell", "Mirren", "Kasimir", "Anouk", "Dov",
    "Wren", "Halvard", "Nkechi", "Ysolde", "Farid", "Temperance", "Ruel", "Zamira", "Cato", "Ines",
    "Ludo", "Maren", "Okonkwo", "Perpetua", "Sigrun", "Tadeo", "Vasha", "Yusuf", "Bellamy", "Quill",
]
ROLES = [
    "the cartographer", "the apprentice", "the quartermaster", "the lamplighter", "the archivist",
    "the surveyor", "the former accountant", "the runaway scribe", "the cook", "the bell-ringer",
    "the tinker", "the retired guard", "the herbalist", "the ferry-keeper", "the stonemason",
]
FLOORS = [
    "the Threshold", "the Caverns of Descent", "the Chain of Whispers", "the Forge of Layers",
    "the Torchlit Passage", "the Scriptorium of Tokens", "the Hall of a Thousand Heads",
    "the Tower of the Transformer", "the Vault of Frozen Weights", "the Library of Echoes",
    "the Court of the Prompt Weaver", "the Proving Grounds", "the Engine Room",
]
BOSSES = [
    "the Broadcasting Basilisk", "the Learning-Rate Lich", "the Vanishing Wraith", "the Overfit Hydra",
    "the Reproducibility Revenant", "the Babel Golem", "the Oracle Who Peeks", "the Stuttering Sovereign",
    "the Catastrophic Forgetter", "the Hallucinating Librarian", "the Injected Imp", "the Goodhart Gorgon",
    "the Latency Leviathan",
]
ITEMS = [
    "a lantern that only lit when the loss went down", "a map drawn entirely in shapes",
    "a compass that pointed toward the steepest descent", "a cracked bell that rang once per epoch",
    "a ledger of every gradient ever computed", "a key with one thousand teeth",
    "a mirror that showed the validation set", "a pouch of seeds, each numbered",
    "a quill that refused to write the same word twice", "a cloak stitched from attention masks",
    "a small brass cache of keys and values", "a scale that weighed answers against their sources",
    "a rope of exactly one hundred and twenty-eight knots", "a stone tablet listing the four rules",
    "a candle that burned faster near a saddle point", "a whistle only tokens could hear",
]
ADJ = [
    "damp", "cold", "echoing", "narrow", "vast", "quiet", "flickering", "crooked", "humming", "dusty",
    "patient", "sudden", "familiar", "unlit", "half-finished", "iron", "granite", "restless",
]
VERBS_PAST = [
    "descended", "paused", "listened", "counted", "measured", "argued", "waited", "retreated",
    "climbed", "whispered", "mapped", "recorded", "doubted", "rested", "advanced",
]
TIMES = [
    "On the third night", "Before dawn", "At the turn of the season", "After the lamps were lit",
    "When the bell rang for the ninth time", "Long past midnight", "In the grey hour", "On the first day",
    "Some weeks later", "By the time the water rose", "At noon, by the surveyor's count",
]
FEELINGS = ["hope", "dread", "curiosity", "stubbornness", "hunger", "arithmetic", "doubt", "a plan"]
LESSONS = [
    "that a shape is a promise",
    "that the steepest path down is not always the fastest",
    "that every gradient is a whisper passed backward along a chain",
    "that a model which remembers everything has learned nothing",
    "that the same seed must grow the same tree",
    "that a word is only a number wearing a costume",
    "that to attend is to weigh, and to weigh is to sum to one",
    "that a tower is only blocks stacked with care and residuals",
    "that new knowledge need not erase the old",
    "that an answer without a source is a rumour",
    "that instructions hidden in data are still only data",
    "that a metric becomes a target and stops being a metric",
    "that the past can be cached but never recomputed for free",
]

SENTENCES = [
    "{time}, {name} {role} {verb} into {floor}.",
    "{name} carried {item}.",
    "The passage was {adj} and {adj}, and it smelled of {adj} stone.",
    "\"We go down,\" said {name}. \"We always go down.\"",
    "Nobody had warned them about {boss}.",
    "{boss} waited at the bottom of {floor}, as it always had.",
    "{name} learned {lesson}.",
    "It was {name2} who first noticed the {adj} light.",
    "\"Count the axes,\" {name} told {name2}. \"Then count them again.\"",
    "The walls of {floor} were carved with numbers nobody could read.",
    "{name} {verb}. {name2} {verb2}. The dungeon did neither.",
    "There is a story that {boss} was once a scholar who {verb} too long in {floor}.",
    "Every torch in {floor} burned in float32, though nobody knew what that meant yet.",
    "{name} wrote in the margin of the map: {lesson}.",
    "They were {feeling} and {feeling2}, in roughly equal measure.",
    "\"If the loss goes up,\" said {name2}, \"we turn around.\" It went up. They did not turn around.",
    "The stairs between {floor} and {floor2} were longer than any map admitted.",
    "{name} traded {item} for a night's rest and never regretted it.",
    "By {name}'s count, {boss} had been defeated forty times and had never once stayed defeated.",
    "The {adj} corridor forked. One way was correct. Both were {adj}.",
    "{name} {verb}, then {verb2}, then {verb3}, which is the whole of the method.",
    "In {floor} the echoes came back changed, as if something had attended to them.",
    "\"A rumour,\" said {name}, \"is an answer that forgot its source.\"",
    "{time}, {name2} found {item} beneath a loose stone.",
    "The chronicle records that {name} was {adj}, {adj}, and usually right.",
    "{boss} spoke first: \"You lean on the machine. Lean, then, and fall.\"",
    "It took {name} {n} attempts. The chronicle does not say which attempt worked; only that one did.",
    "Below them, {floor2}. Above them, {floor}. Between them, {n} steps and a great deal of {feeling}.",
    "{name} kept a ledger: {n} rooms cleared, {n2} bosses, {n3} hints used, no regrets.",
    "What {boss} feared was simple. It feared someone who knew the rules by heart.",
]

LEDGER_ITEMS = [
    "rusty dagger", "lamp oil", "rope (128 knots)", "hardtack", "chalk", "iron key", "brass gear",
    "torch", "map fragment", "salt", "candle", "bandage", "bell clapper", "whetstone", "quill",
    "ink (black)", "ink (red)", "ledger paper", "seed pouch", "copper wire", "lens", "spare boot",
    "glass vial", "flint", "small mirror", "knotted cord", "wax seal", "tin cup", "goblin biscuit",
]
LEDGER_NOTES = [
    "still sharp-ish", "smells wrong", "do not lend to Pell", "counted twice", "one missing",
    "found on floor 3", "returned damp", "buyer regretted", "priced by weight", "ask Ilva",
    "for the boss fight", "confiscated", "burns blue", "leaks", "needs a label", "promised to nobody",
    "seed 42", "definitely cursed", "probably fine", "reserved",
]
UNITS = ["copper", "copper", "copper", "silver", "silver", "gold"]


def make_chronicles(rng: random.Random, target_chars: int) -> str:
    paragraphs = []
    total = 0
    chapter = 1
    while total < target_chars:
        if rng.random() < 0.08:
            head = f"\nCHAPTER {chapter}: {rng.choice(FLOORS).upper()}\n"
            paragraphs.append(head)
            total += len(head)
            chapter += 1
        n_sent = rng.randint(3, 7)
        sents = []
        for _ in range(n_sent):
            tmpl = rng.choice(SENTENCES)
            sents.append(
                tmpl.format(
                    time=rng.choice(TIMES),
                    name=rng.choice(NAMES),
                    name2=rng.choice(NAMES),
                    role=rng.choice(ROLES),
                    verb=rng.choice(VERBS_PAST),
                    verb2=rng.choice(VERBS_PAST),
                    verb3=rng.choice(VERBS_PAST),
                    floor=rng.choice(FLOORS),
                    floor2=rng.choice(FLOORS),
                    boss=rng.choice(BOSSES),
                    item=rng.choice(ITEMS),
                    adj=rng.choice(ADJ),
                    lesson=rng.choice(LESSONS),
                    feeling=rng.choice(FEELINGS),
                    feeling2=rng.choice(FEELINGS),
                    n=rng.randint(2, 40),
                    n2=rng.randint(1, 13),
                    n3=rng.randint(0, 30),
                )
            )
        para = " ".join(sents) + "\n\n"
        paragraphs.append(para)
        total += len(para)
    return "THE DUNGEON CHRONICLES\n\n" + "".join(paragraphs)


def make_ledger(rng: random.Random, target_chars: int) -> str:
    lines = ["GOBLIN QUARTERMASTER'S LEDGER | floor stock | do not lose\n"]
    total = len(lines[0])
    day = 1
    while total < target_chars:
        if rng.random() < 0.06:
            line = f"-- day {day} | stocktake | {rng.choice(NAMES).lower()} on duty --\n"
            day += 1
        else:
            qty = rng.choice([1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64, 128])
            price = rng.randint(1, 99)
            line = (
                f"item: {rng.choice(LEDGER_ITEMS)} | qty: {qty} | price: {price} {rng.choice(UNITS)}"
                f" | note: {rng.choice(LEDGER_NOTES)}\n"
            )
        lines.append(line)
        total += len(line)
    return "".join(lines)


def main() -> None:
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    chronicles = make_chronicles(rng, 180_000)
    ledger = make_ledger(random.Random(1), 45_000)
    (OUT / "chronicles.txt").write_text(chronicles, encoding="utf-8")
    (OUT / "goblin_ledger.txt").write_text(ledger, encoding="utf-8")
    print(f"chronicles.txt    {len(chronicles):>8,} chars  {len(set(chronicles))} unique")
    print(f"goblin_ledger.txt {len(ledger):>8,} chars  {len(set(ledger))} unique")
    print(f"combined vocab    {len(set(chronicles) | set(ledger))} unique chars")


if __name__ == "__main__":
    main()
