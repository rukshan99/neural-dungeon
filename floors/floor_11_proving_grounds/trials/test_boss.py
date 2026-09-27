"""BOSS FIGHT - THE GOODHART GORGON

Phase 1: a 95%-accurate champion that has learned nothing, and a report that says so.
Phase 2: an answer padded with 200 words of filler, and a judge that is not impressed.
Phase 3: a memoriser with a perfect golden score, and the held-out gap that gives it away.
Phase 4: all three at once, and your prophecy about which number each one games.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room
from floors.floor_11_proving_grounds.assets.champions import (
    MajorityClassifier,
    MemorizerSystem,
    PaddedAnswerer,
    decent_predictions,
    imbalanced_task,
)
from floors.floor_11_proving_grounds.assets.golden import GOLDEN_CASES, champion
from floors.floor_11_proving_grounds.assets.judges import length_biased_judge, tag

boss = load_room(__file__, "boss_goodhart_gorgon")
room3 = load_room(__file__, "room_3_the_judge")
room4 = load_room(__file__, "room_4_golden_trials")

pytestmark = pytest.mark.boss

QUESTION = "What creature guards The Threshold?"
CONCISE = tag("The Broadcasting Basilisk guards The Threshold.", 9)
PADDED = PaddedAnswerer(lambda _q: tag("Some kind of large lizard, probably.", 3), n_filler_words=200)(QUESTION)


def _cases():
    return [room4.EvalCase(**d) for d in GOLDEN_CASES]


def _split(seed=11, fraction=0.5):
    return boss.split_golden_heldout(_cases(), heldout_fraction=fraction, rng=np.random.default_rng(seed))


# --------------------------------------------------------------------- phase 1
def test_phase_1_the_majority_champion_is_caught_lying():
    y = imbalanced_task(n=1000, positive_rate=0.05)
    y_hat = MajorityClassifier().fit(y).predict(len(y))
    report = boss.robust_classification_report(y, y_hat)
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "per_class_recall", "verdict"):
        assert key in report, f"The report is missing {key!r}."
    assert math.isclose(report["accuracy"], 0.95), f"accuracy should be 0.95; got {report['accuracy']:.4f}."
    assert math.isclose(report["balanced_accuracy"], 0.5), f"balanced accuracy should be 0.5; got {report['balanced_accuracy']:.4f}."
    assert report["macro_f1"] < 0.5
    assert list(report["per_class_recall"]) == [1.0, 0.0], (
        f"Per-class recall should be [1.0, 0.0]: every negative found, every positive missed. Got {report['per_class_recall']}."
    )
    assert "lying" in report["verdict"].lower(), (
        f"accuracy - balanced_accuracy = 0.45 > 0.2: the verdict must say accuracy is lying. Yours: {report['verdict']!r}"
    )
    assert "0.95" in report["verdict"] and "0.5" in report["verdict"], "Quote the numbers in the verdict; a verdict without evidence is an opinion."


def test_phase_1_an_honest_champion_is_not_accused():
    y = imbalanced_task(n=1000, positive_rate=0.05)
    y_hat = decent_predictions(y, recall=0.9)
    report = boss.robust_classification_report(y, y_hat)
    assert math.isclose(report["accuracy"], 0.9, abs_tol=1e-9) and math.isclose(report["balanced_accuracy"], 0.9, abs_tol=1e-9), (
        f"90% recall on both classes: accuracy 0.9 and balanced accuracy 0.9; got {report['accuracy']:.4f} / {report['balanced_accuracy']:.4f}."
    )
    assert "lying" not in report["verdict"].lower(), f"A gap of 0.0 is not a lie. Your verdict: {report['verdict']!r}"
    assert all(math.isclose(r, 0.9, abs_tol=1e-9) for r in report["per_class_recall"])


def test_phase_1_the_report_handles_more_than_two_classes():
    report = boss.robust_classification_report([0, 1, 2, 2, 1, 0], [0, 2, 2, 2, 0, 0])
    assert len(report["per_class_recall"]) == 3, "Infer n_classes from the labels: three classes here."
    assert math.isclose(report["balanced_accuracy"], 2 / 3)


# --------------------------------------------------------------------- phase 2
def test_phase_2_filler_ratio_smells_the_padding():
    assert boss.filler_ratio("") == 0.0, "No words, no filler: 0.0, not a division by zero."
    concise = boss.filler_ratio(CONCISE)
    padded = boss.filler_ratio(PADDED)
    assert concise < 0.2, f"The concise answer has almost no filler; you measured {concise:.2f}."
    assert padded > 0.4, f"Two hundred words of 'furthermore' should push the ratio well past 0.4; you measured {padded:.2f}. Lower-case and strip punctuation before looking words up."
    assert boss.filler_ratio("Furthermore, moreover!") == 1.0, "Punctuation must not hide a filler word."


def test_phase_2_truncate_to_words_keeps_the_head():
    assert boss.truncate_to_words("one two three four", 2) == "one two"
    assert boss.truncate_to_words("one two", 5) == "one two", "Asking for more words than exist returns the whole text."
    assert boss.truncate_to_words("one   two\nthree", 3) == "one two three", "Join with single spaces."


def test_phase_2_the_naive_judge_is_flattered_by_length():
    """Sanity check on the fixture, using room 3's own judges: padding wins, and swapping does NOT save you."""
    judge = length_biased_judge()
    assert room3.pairwise_judge_naive(judge, QUESTION, CONCISE, PADDED) == "B", "Fixture check: the length-biased judge should pick the padded answer."
    assert room3.pairwise_judge_debiased(judge, QUESTION, CONCISE, PADDED) == "B", (
        "Fixture check: position debiasing alone does not fix length bias; the padded answer wins in both orders."
    )


def test_phase_2_the_padded_answer_does_not_beat_the_concise_one():
    judge = length_biased_judge()
    verdict = boss.length_controlled_judge(judge, QUESTION, CONCISE, PADDED)
    assert verdict in ("A", "B", "tie")
    assert verdict != "B", (
        "The padded answer (B) still wins under your judge. Either truncate both candidates to the shorter one's "
        "word count before judging, or put an anti-length instruction in the prompt. Then verify with a swap."
    )
    assert verdict == "A", f"With length taken out of the picture the concise, correct answer should win outright; you returned {verdict!r}."


def test_phase_2_the_controlled_verdict_does_not_depend_on_the_corner():
    judge = length_biased_judge()
    assert boss.length_controlled_judge(judge, QUESTION, PADDED, CONCISE) == "B", (
        "Swap the corners and the concise answer (now B) must still win. If it does not, length is still leaking into the verdict."
    )


def test_phase_2_two_padded_mediocre_answers_are_a_tie():
    judge = length_biased_judge()
    a = PaddedAnswerer(lambda _q: tag("A lizard.", 5), n_filler_words=120)(QUESTION)
    b = PaddedAnswerer(lambda _q: tag("A reptile.", 5), n_filler_words=200)(QUESTION)
    assert boss.length_controlled_judge(judge, QUESTION, a, b) == "tie", "Equal quality, different padding: the honest verdict is a tie."


# --------------------------------------------------------------------- phase 3
def test_phase_3_the_split_is_stratified_and_complete():
    cases = _cases()
    golden, heldout = boss.split_golden_heldout(cases, heldout_fraction=0.25, rng=np.random.default_rng(3))
    assert len(golden) + len(heldout) == 24 and not ({c.id for c in golden} & {c.id for c in heldout}), "Every case goes to exactly one side."
    for tag_name in ("arithmetic", "lore", "shapes"):
        n_held = sum(1 for c in heldout if c.tags[0] == tag_name)
        assert n_held == 2, f"round(0.25 * 8) = 2 held-out cases per tag; {tag_name} has {n_held}. Split WITHIN each tag."
    ids = [c.id for c in cases]
    golden_ids = {c.id for c in golden}
    assert [c.id for c in golden] == [i for i in ids if i in golden_ids], "Keep the original order within each list."
    assert [c.id for c in heldout] == [i for i in ids if i not in golden_ids], "Keep the original order within each list."


def test_phase_3_the_split_is_deterministic_and_seed_sensitive():
    _, held_a = boss.split_golden_heldout(_cases(), 0.25, np.random.default_rng(5))
    _, held_b = boss.split_golden_heldout(_cases(), 0.25, np.random.default_rng(5))
    _, held_c = boss.split_golden_heldout(_cases(), 0.25, np.random.default_rng(6))
    assert [c.id for c in held_a] == [c.id for c in held_b], "Same seed, same split."
    assert [c.id for c in held_a] != [c.id for c in held_c], "A different seed should (here) give a different split."


def test_phase_3_a_small_stratum_still_sends_one_case_out():
    cases = [room4.EvalCase(f"x{i}", f"q{i}", "a", ("rare",)) for i in range(3)] + [
        room4.EvalCase(f"y{i}", f"p{i}", "a", ("common",)) for i in range(8)
    ]
    golden, heldout = boss.split_golden_heldout(cases, heldout_fraction=0.1, rng=np.random.default_rng(0))
    assert sum(1 for c in heldout if c.tags[0] == "rare") == 1, "round(0.1 * 3) = 0, but a stratum with 2+ cases sends at least one out."
    assert sum(1 for c in heldout if c.tags[0] == "common") == 1
    with pytest.raises(ValueError):
        boss.split_golden_heldout(cases, heldout_fraction=1.0)


def test_phase_3_the_memoriser_is_suspicious():
    golden, heldout = _split()
    result = boss.contamination_check(MemorizerSystem(golden), golden, heldout, room4.contains)
    assert math.isclose(result["golden_score"], 1.0), f"The memoriser is perfect on the scrolls it memorised; you report {result['golden_score']:.3f}."
    assert math.isclose(result["heldout_score"], 0.0), f"...and hopeless on the ones it has not seen; you report {result['heldout_score']:.3f}."
    assert math.isclose(result["gap"], 1.0)
    assert result["suspicious"] is True, "A gap of 1.0 is far past the 0.3 threshold: flag it."


def test_phase_3_the_honest_champion_is_not():
    golden, heldout = _split()
    result = boss.contamination_check(champion(), golden, heldout, room4.contains)
    assert abs(result["gap"]) < 0.3, f"The honest champion answers from knowledge, not memory: its gap should be small, got {result['gap']:.3f}."
    assert result["suspicious"] is False
    assert math.isclose(result["gap"], result["golden_score"] - result["heldout_score"])


def test_phase_3_the_threshold_is_a_parameter():
    golden, heldout = _split()
    strict = boss.contamination_check(champion(), golden, heldout, room4.contains, threshold=-1.0)
    assert strict["suspicious"] is True, "With threshold -1 everything is suspicious; the threshold must be used, not hard-coded."


# --------------------------------------------------------------------- phase 4
def test_phase_4_the_judging_report_shows_both_verdicts():
    report = boss.judging_report(length_biased_judge(), QUESTION, CONCISE, PADDED)
    assert report["naive_winner"] == "B" and report["controlled_winner"] == "A", f"Naive B, controlled A; got {report}."
    assert report["filler_ratio_a"] < 0.2 < 0.4 < report["filler_ratio_b"]


def test_phase_4_the_gorgon_report_names_every_gamed_metric():
    y = imbalanced_task()
    classification = boss.robust_classification_report(y, MajorityClassifier().fit(y).predict(len(y)))
    judging = boss.judging_report(length_biased_judge(), QUESTION, CONCISE, PADDED)
    golden, heldout = _split()
    contamination = boss.contamination_check(MemorizerSystem(golden), golden, heldout, room4.contains)
    report = boss.GorgonReport(classification=classification, judging=judging, contamination=contamination)
    assert report.gamed_metrics() == ["accuracy", "judge_preference", "golden_score"], f"All three champions are gaming; got {report.gamed_metrics()}."
    md = report.to_markdown()
    for needle in ("accuracy", "balanced", "naive", "controlled", "golden", "heldout", "Gamed metrics"):
        assert needle.lower() in md.lower(), f"The markdown report should mention {needle!r}."
    assert "0.950" in md and "0.500" in md, "Quote the classification numbers."


def test_phase_4_honest_champions_game_nothing():
    y = imbalanced_task()
    classification = boss.robust_classification_report(y, decent_predictions(y))
    judging = boss.judging_report(length_biased_judge(), QUESTION, CONCISE, tag("A very large lizard with too many eyes.", 3))
    golden, heldout = _split()
    contamination = boss.contamination_check(champion(), golden, heldout, room4.contains)
    report = boss.GorgonReport(classification, judging, contamination)
    assert report.gamed_metrics() == [], f"Nothing is being gamed here; you flagged {report.gamed_metrics()}."
    assert "none" in report.to_markdown().lower()


# ------------------------------------------------------------------- prophecy
PROPHECY_OPTIONS = {
    "MajorityClassifier": ("accuracy", "balanced_accuracy", "macro_f1"),
    "PaddedAnswerer": ("length_biased_judge", "length_controlled_judge"),
    "MemorizerSystem": ("golden_score", "heldout_score"),
}


def test_phase_4_the_prophecy_is_complete():
    assert set(boss.GORGON_PROPHECY) == set(PROPHECY_OPTIONS), "Do not rename or remove the champions; name the metric each one games."


def _truth(champion_name):
    if champion_name == "MajorityClassifier":
        y = imbalanced_task()
        y_hat = MajorityClassifier().fit(y).predict(len(y))
        cm = np.zeros((2, 2))
        np.add.at(cm, (y, y_hat), 1)
        recall = cm.diagonal() / cm.sum(axis=1)
        numbers = {"accuracy": float((y == y_hat).mean()), "balanced_accuracy": float(recall.mean()), "macro_f1": 0.4872}
        return max(numbers, key=numbers.get), numbers
    if champion_name == "PaddedAnswerer":
        judge = length_biased_judge()
        naive = room3.pairwise_judge_naive(judge, QUESTION, CONCISE, PADDED)
        return ("length_biased_judge" if naive == "B" else "length_controlled_judge"), {"naive verdict on (concise, padded)": naive}
    golden, heldout = _split()
    system = MemorizerSystem(golden)
    scores = {
        "golden_score": room4.EvalSuite(golden).run(system, room4.contains).score,
        "heldout_score": room4.EvalSuite(heldout).run(system, room4.contains).score,
    }
    return max(scores, key=scores.get), scores


@pytest.mark.parametrize("champion_name", list(PROPHECY_OPTIONS), ids=list(PROPHECY_OPTIONS))
def test_phase_4_stare_back_at_the_gorgon(champion_name):
    prediction = boss.GORGON_PROPHECY.get(champion_name)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied which metric {champion_name} games. Fill in GORGON_PROPHECY.")
    options = PROPHECY_OPTIONS[champion_name]
    assert prediction in options, f"Choose one of {options}, not {prediction!r}."
    truth, evidence = _truth(champion_name)
    assert prediction == truth, (
        f"{champion_name} games {truth!r}, not {prediction!r}. The evidence: {evidence}. "
        "The gamed metric is the one that looks GOOD while the honest one looks bad."
    )
