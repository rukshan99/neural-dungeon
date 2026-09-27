"""TRIAL 11.4 - GOLDEN TRIALS

Twenty-four sealed scrolls, one champion, one ledger. Then yesterday's ledger
against today's, to find out what broke.
"""

import json
import math

import pytest

from dungeon.artifacts.llm import ScriptedLLM
from dungeon.trials import load_room
from floors.floor_11_proving_grounds.assets.golden import (
    CHAMPION_WRONG_IDS,
    GOLDEN_CASES,
    champion,
    degraded_champion,
)

room = load_room(__file__, "room_4_golden_trials")


def _cases():
    return [room.EvalCase(**d) for d in GOLDEN_CASES]


def _case(expected="42", tags=("arithmetic",)):
    return room.EvalCase(id="c1", input="What is the answer?", expected=expected, tags=tags)


# ------------------------------------------------------------------- scorers
def test_exact_match_forgives_whitespace_and_nothing_else():
    assert room.exact_match(_case("42"), " 42\n") == 1.0
    assert room.exact_match(_case("42"), "The answer is 42.") == 0.0, "exact_match is exact: the whole output must equal the expected text."


def test_contains_finds_the_expected_text_anywhere():
    assert room.contains(_case("Basilisk"), "It was the basilisk, obviously.") == 1.0, "contains ignores case."
    assert room.contains(_case("(2, 4)"), "shape (2, 4).") == 1.0
    assert room.contains(_case("(2, 4)"), "shape (2, 3).") == 0.0


def test_numeric_tolerance_reads_the_first_number():
    scorer = room.numeric_tolerance(tol=0.01)
    assert scorer(_case("3.14"), "About 3.141, give or take.") == 1.0
    assert scorer(_case("3.14"), "About 3.2.") == 0.0, "3.2 is more than 0.01 away from 3.14."
    assert scorer(_case("3.14"), "No idea.") == 0.0, "No number in the output scores 0, not a crash."
    assert scorer(_case("36"), "Three potions come to 36 gold.") == 1.0


def test_judge_scorer_maps_the_scale_onto_zero_one():
    llm = ScriptedLLM(['{"score": 4, "rationale": "good"}', '{"score": 1, "rationale": "bad"}'])
    scorer = room.judge_scorer(llm, room.Rubric(["Correct"], (1, 5)))
    assert math.isclose(scorer(_case(), "some answer"), 0.75), "A 4 on a 1..5 scale is (4-1)/(5-1) = 0.75."
    assert math.isclose(scorer(_case(), "another"), 0.0), "The bottom of the scale is 0.0."


# --------------------------------------------------------------------- suite
def test_the_champion_scores_twenty_one_of_twenty_four():
    report = room.EvalSuite(_cases()).run(champion(), room.contains)
    assert isinstance(report, room.EvalReport)
    assert len(report.per_case) == 24, f"One result per case: expected 24, got {len(report.per_case)}."
    assert math.isclose(report.score, 21 / 24), f"The champion is right on 21 of 24 scrolls, so the score is {21 / 24:.4f}; you report {report.score:.4f}."
    failed = {r.id for r in report.per_case if not r.passed}
    assert failed == CHAMPION_WRONG_IDS, f"The champion fails exactly {sorted(CHAMPION_WRONG_IDS)}; your report says {sorted(failed)}."
    assert report.n_passed == 21


def test_results_keep_the_case_order_and_the_evidence():
    report = room.EvalSuite(_cases()).run(champion(), room.contains)
    assert [r.id for r in report.per_case] == [d["id"] for d in GOLDEN_CASES], "Results must be in case order."
    first = report.per_case[0]
    assert first.expected == "36" and "36" in first.output and first.score == 1.0 and first.passed, (
        f"Keep the evidence: expected, the system's actual output, the score and pass/fail. Got {first!r}."
    )
    assert first.tags == ("arithmetic",)


def test_scores_are_aggregated_per_tag():
    report = room.EvalSuite(_cases()).run(champion(), room.contains)
    assert set(report.by_tag) == {"arithmetic", "lore", "shapes"}, f"One entry per tag; got {sorted(report.by_tag)}."
    for tag_name, value in report.by_tag.items():
        assert math.isclose(value, 7 / 8), f"The champion misses one scroll per tag, so every tag scores 0.875; {tag_name} reports {value:.4f}."


def test_a_champion_that_faints_mid_trial_scores_zero_not_a_crash():
    def fainting(prompt):
        if "potion" in prompt:
            raise RuntimeError("the champion faints")
        return champion()(prompt)

    report = room.EvalSuite(_cases()).run(fainting, room.contains)
    fainted = report.per_case[0]
    assert fainted.score == 0.0 and not fainted.passed, "A crashing case scores 0."
    assert "ERROR" in fainted.output and "RuntimeError" in fainted.output, f"Record the error as the output so it can be read later; got {fainted.output!r}."
    assert math.isclose(report.score, 20 / 24), "The other 23 cases must still run."


def test_a_scorer_that_jams_is_recorded_too_not_raised():
    def jamming_scorer(case, output):
        if case.id == "lore_01":
            raise ValueError("the scales jammed")
        return room.contains(case, output)

    report = room.EvalSuite(_cases()).run(champion(), jamming_scorer)
    jammed = next(r for r in report.per_case if r.id == "lore_01")
    assert jammed.score == 0.0 and not jammed.passed, "A scorer that raises scores that case 0, like a system that raises."
    assert "ERROR" in jammed.output and "ValueError" in jammed.output, f"Record the scorer's error as the output too; got {jammed.output!r}."
    assert math.isclose(report.score, 20 / 24), "The other 23 cases still run and are scored normally."


def test_the_suite_refuses_duplicate_ids():
    with pytest.raises(ValueError):
        room.EvalSuite([_case(), _case()])


def test_the_pass_threshold_is_respected():
    llm = ScriptedLLM(['{"score": 4, "rationale": "good"}'] * 24)
    report = room.EvalSuite(_cases()).run(champion(), room.judge_scorer(llm, room.Rubric(["Correct"], (1, 5))), pass_threshold=0.8)
    assert report.n_passed == 0 and math.isclose(report.score, 0.75), "A 0.75 score does not pass a 0.8 threshold, but it still counts toward the mean."


# ------------------------------------------------------------------ markdown
def test_the_markdown_ledger_names_every_scroll():
    report = room.EvalSuite(_cases()).run(champion(), room.contains)
    md = report.to_markdown()
    assert isinstance(md, str)
    for d in GOLDEN_CASES:
        assert d["id"] in md, f"The case {d['id']} is missing from the markdown table."
    assert "0.875" in md, "State the overall score to three decimals."
    assert "21/24" in md or "21 / 24" in md, "State how many passed."
    table_rows = [line for line in md.splitlines() if line.startswith("|")]
    assert len(table_rows) >= 24 + 2, "Use a markdown table (rows starting with '|') with one row per case."
    for tag_name in ("arithmetic", "lore", "shapes"):
        assert tag_name in md, f"The per-tag table should mention {tag_name}."


def test_pipes_inside_cells_are_escaped():
    report = room.EvalSuite([_case("a|b")]).run(lambda _p: "a|b", room.exact_match)
    md = report.to_markdown()
    assert "a\\|b" in md, "A '|' inside a cell breaks the table; escape it as '\\|'."


# ---------------------------------------------------------------- round-trip
def test_the_ledger_survives_a_trip_through_json(tmp_path):
    report = room.EvalSuite(_cases()).run(champion(), room.contains)
    path = room.save_report(report, tmp_path / "baseline.json")
    assert path.is_file()
    json.loads(path.read_text())  # must be real JSON
    loaded = room.load_report(path)
    assert loaded == report, (
        "load_report(save_report(r)) must equal r. Common culprits: tags came back as lists instead of tuples, "
        "or by_tag was not saved."
    )
    assert loaded.per_case[0].tags == ("arithmetic",)


# --------------------------------------------------------------- regressions
def test_yesterdays_champion_forgot_how_to_count_gold():
    baseline = room.EvalSuite(_cases()).run(champion(), room.contains)
    today = room.EvalSuite(_cases()).run(degraded_champion(), room.contains)
    regressions = room.compare_to_baseline(today, baseline, tolerance=0.02)
    assert all(isinstance(r, room.Regression) for r in regressions)
    case_ids = [r.case_id for r in regressions if r.kind == "case"]
    expected = ["arith_01", "arith_02", "arith_03", "arith_04", "arith_06", "arith_07", "arith_08"]
    assert case_ids == expected, (
        f"Seven arithmetic scrolls passed yesterday and fail today: {expected}. You flagged {case_ids}. "
        "arith_05 was already failing, so it is not a regression."
    )
    overall = [r for r in regressions if r.kind == "overall"]
    assert len(overall) == 1, "The overall score fell from 0.875 to 0.583: flag it once."
    assert math.isclose(overall[0].before, 21 / 24) and math.isclose(overall[0].after, 14 / 24)
    assert overall[0].case_id is None
    for r in regressions:
        assert r.message, "Every regression needs a human-readable message."


def test_an_unchanged_champion_has_no_regressions():
    baseline = room.EvalSuite(_cases()).run(champion(), room.contains)
    again = room.EvalSuite(_cases()).run(champion(), room.contains)
    assert room.compare_to_baseline(again, baseline) == [], "Same system, same scores: nothing to flag."


def test_a_drop_within_tolerance_is_noise_not_a_regression():
    cases = _cases()
    baseline = room.EvalSuite(cases).run(champion(), room.contains)
    honest = champion()

    def slightly_worse(prompt):
        # Break exactly one case that passed yesterday (arith_01).
        return "no comment" if "3 potions" in prompt else honest(prompt)

    today = room.EvalSuite(cases).run(slightly_worse, room.contains)
    regs = room.compare_to_baseline(today, baseline, tolerance=0.05)
    kinds = [r.kind for r in regs]
    assert kinds == ["case"], f"One case regressed (a 1/24 = 0.042 drop, inside the 0.05 tolerance): expected ['case'], got {kinds}."
    assert regs[0].case_id == "arith_01"
