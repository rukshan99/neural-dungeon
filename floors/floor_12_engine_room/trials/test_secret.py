"""SECRET - THE SPECULATIVE SPARK

A repetitive chronicle, decoded exactly as greedy would, in fewer forward
passes than there are tokens.
"""

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts.tiny_gpt import load_pretrained  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

spark = load_room(__file__, "secret_speculative_spark")

pytestmark = pytest.mark.secret

torch.manual_seed(1299)
torch.set_num_threads(1)
MODEL, TOK, _ = load_pretrained()
CFG = MODEL.cfg
N_NEW = 40
PASS_BUDGET = 0.6  # measured: 12 passes for 40 tokens on the Revenant prompt

# Sentences from the chronicles, each followed by the first half of itself: the
# model completes the repetition, and prompt lookup can see it coming.
SENTENCES = [
    "What the Reproducibility Revenant feared was simple.",
    "They were hope and curiosity, in roughly equal measure.",
]


def _repetitive(sentence: str) -> torch.Tensor:
    prompt = sentence + " " + sentence[: len(sentence) // 2]
    return torch.tensor([TOK.encode(prompt)], dtype=torch.long)


# ------------------------------------------------------------------ propose_from_prompt
def test_the_draft_copies_what_followed_the_last_ngram():
    assert spark.propose_from_prompt([1, 2, 3, 4, 5, 1, 2, 3], n=3, k=5) == [4, 5, 1, 2, 3], (
        "The last 3-gram [1, 2, 3] appeared at the start; what followed it is [4, 5, 1, 2, 3]."
    )
    assert spark.propose_from_prompt([1, 2, 3, 4, 5, 1, 2, 3], n=3, k=2) == [4, 5], "k caps the draft."
    assert spark.propose_from_prompt([1, 2, 3, 4, 5, 6, 7, 8, 9], n=3, k=5) == [], "No earlier occurrence: no draft."
    assert spark.propose_from_prompt([1, 2, 3], n=3, k=5) == [], "The n-gram must not match itself."
    assert spark.propose_from_prompt([1, 2], n=3, k=5) == [], "Fewer than n tokens: nothing to look up."


def test_the_most_recent_occurrence_wins_and_the_draft_may_be_short():
    ids = [7, 8, 9, 1, 7, 8, 9, 2, 7, 8, 9]
    assert spark.propose_from_prompt(ids, n=3, k=5) == [2, 7, 8, 9], (
        "Two earlier occurrences of [7, 8, 9]: use the most recent one (followed by 2, 7, 8, 9), "
        "and stop when the context runs out."
    )


# ------------------------------------------------------------------ speculative_generate
@pytest.mark.parametrize("sentence", SENTENCES, ids=["revenant", "hope_and_curiosity"])
def test_speculation_is_exactly_greedy_in_fewer_passes(sentence):
    idx = _repetitive(sentence)
    with torch.no_grad():
        ref = MODEL.generate(idx, N_NEW, temperature=0)
        got, stats = spark.speculative_generate(MODEL, idx, N_NEW, n=3, k=5)
    assert got.shape == ref.shape, f"Expected {tuple(ref.shape)} (prompt + {N_NEW} tokens), got {tuple(got.shape)}."
    if not torch.equal(got, ref):
        first = int((got[0] != ref[0]).nonzero()[0]) - idx.shape[1]
        pytest.fail(
            f"Token {first} differs: greedy wrote {TOK.decode(ref[0, idx.shape[1]:])!r}, you wrote "
            f"{TOK.decode(got[0, idx.shape[1]:])!r}. Accept only while draft[j] == argmax at position T-1+j, "
            "then append the model's own token."
        )
    assert set(stats) >= {"forward_passes", "proposed", "accepted", "generated"}, f"stats keys: {sorted(stats)}"
    assert stats["generated"] == N_NEW
    assert stats["forward_passes"] < N_NEW, (
        f"{stats['forward_passes']} forward passes for {N_NEW} tokens: no better than plain greedy. "
        "Verify the whole draft with ONE pass over prompt + draft."
    )
    assert stats["forward_passes"] <= PASS_BUDGET * N_NEW, (
        f"{stats['forward_passes']} passes for {N_NEW} tokens (accepted {stats['accepted']} of {stats['proposed']} proposed). "
        f"On this repetitive chronicle the reference needs about 12; the budget is {int(PASS_BUDGET * N_NEW)}."
    )
    assert stats["accepted"] <= stats["proposed"]


def test_without_repetition_the_spark_still_speaks_greedy():
    idx = torch.tensor([TOK.encode("Kasimir retreated. ")], dtype=torch.long)
    with torch.no_grad():
        ref = MODEL.generate(idx, 25, temperature=0)
        got, stats = spark.speculative_generate(MODEL, idx, 25, n=3, k=5)
    assert torch.equal(got, ref), "Whatever the drafts do, the output must be greedy's output."
    assert stats["forward_passes"] <= 25, "Never more passes than tokens: an empty draft is still one token per pass."


def test_the_spark_respects_block_size():
    idx = torch.tensor([TOK.encode("Rest, then climb, then measure. " * 3)], dtype=torch.long)
    room_left = CFG.block_size - idx.shape[1]
    with torch.no_grad():
        got, stats = spark.speculative_generate(MODEL, idx, room_left + 20, n=3, k=5)
        ref = MODEL.generate(idx, room_left, temperature=0)
    assert got.shape[1] == CFG.block_size, f"The output must stop at block_size ({CFG.block_size}); it is {got.shape[1]} long."
    assert torch.equal(got, ref) and stats["generated"] == room_left
