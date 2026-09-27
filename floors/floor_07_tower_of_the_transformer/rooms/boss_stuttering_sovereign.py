"""BOSS - THE STUTTERING SOVEREIGN

                    _.-=-._
                  .'  ___  '.          "The chronicle records that Bellamy
                 /  .'   '.  \\          was vast, vast, and usually right.
                |  |  o o  |  |         The chronicle records that Bellamy
                |  |   ^   |  |         was vast, vast, and usually right.
                 \\  '.___.'  /          The chronicle records that Bellamy
                  '._______.'           was vast, vast, and usually r-"
                 .-'|     |'-.
                /   |_____|   \\        The Sovereign at the top of the tower
               |   / _____ \\   |       knows the chronicles almost by heart.
               |  | |     | |  |       Ask it to speak and it picks the most
               |__|_|     |_|__|       likely character, then the most likely
                                       after that, and falls into the same
                                       loop every time.

WEAKNESS: the decoding strategy. Not the weights: the weights are fine. A
tower that is *sure* loops under greedy decoding; a tower that is made too
*unsure* (high temperature) babbles. The Sovereign's court judges speech by
BOTH measures, so random noise cannot win:

    diversity:  distinct 4-grams / total 4-grams  >= 0.6
    fluency:    mean per-token NLL under the model <= 1.0

Greedy scores about 0.23 on diversity. Uniform noise scores 1.0 on diversity
and about 11 on NLL. Sensible sampling sits at 0.8-0.95 and 0.15-0.6.

The fight has four phases:

  Phase 1  distinct_ngram_ratio, repetition_penalty, block_repeated_ngrams -
           the instruments of anti-repetition.
  Phase 2  mean_nll - the fluency judge: teacher-forced NLL of a sequence,
           in windows of block_size.
  Phase 3  speak_without_stuttering - 300 characters that satisfy both judges
           on three seeds. Combine what you built: your room 5 sampler, a
           penalty or a block, a temperature.
  Phase 4  SOVEREIGN_PROPHECY - predict how four strategies fare before the
           court measures them.

Run:  dungeon fight 7
"""

from __future__ import annotations

import torch

from .room_5_the_voice import sample_next  # noqa: F401  (your sampler; use it in phase 3)


def distinct_ngram_ratio(ids, n: int) -> float:
    """Number of distinct n-grams divided by the total number of n-grams in a 1-D sequence.

    ``ids`` may be a list of ints or a 1-D tensor. A sequence with fewer
    than n tokens has no n-grams: return 0.0.
    """
    raise NotImplementedError("distinct_ngram_ratio() is unwritten")


def repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style penalty on tokens already generated, row by row.

    logits (B, V), generated_ids (B, T). For each token that appears in
    row b's history: a positive logit is divided by ``penalty``, a negative
    one is multiplied by it (both push it down). Return a NEW tensor; leave
    ``logits`` untouched. penalty=1.0 changes nothing.
    """
    raise NotImplementedError("repetition_penalty() is unwritten")


def block_repeated_ngrams(logits: torch.Tensor, generated_ids: torch.Tensor, n: int) -> torch.Tensor:
    """Set to -inf every token that would complete an n-gram already present in the row.

    The prefix is the row's last n-1 tokens. Every earlier position where
    that same prefix occurred names one banned continuation: the token that
    followed it. A history shorter than n bans nothing. With n=1 the prefix
    is empty and every token already generated is banned. Return a NEW
    tensor of shape (B, V).
    """
    raise NotImplementedError("block_repeated_ngrams() is unwritten")


def mean_nll(model, ids) -> float:
    """Mean negative log-likelihood per predicted token of ``ids`` under ``model``, teacher-forced.

    ``ids`` is a 1-D sequence of length L >= 2. Every token ids[1:] is a
    target; ids[0] is only context. Longer than block_size is scored in
    consecutive non-overlapping windows: window w feeds ids[wB : wB + B] and
    is scored against ids[wB + 1 : wB + B + 1] (B = block_size). Return
    total NLL / (L - 1) as a float. No gradients, eval mode.
    """
    raise NotImplementedError("mean_nll() is unwritten")


def speak_without_stuttering(model, tokenizer, prompt: str, n_tokens: int = 300, seed: int = 0) -> str:
    """Return ``prompt`` followed by exactly ``n_tokens`` generated characters.

    Seed a ``torch.Generator`` with ``seed`` and decode with a strategy that
    passes both judges: distinct-4-gram ratio >= 0.6 and mean NLL <= 1.0.
    Crop the context to ``model.cfg.block_size`` each step.
    """
    raise NotImplementedError("speak_without_stuttering() is unwritten")


# ---------------------------------------------------------------------------
# PHASE 4: THE PROPHECY
# For each strategy, predict the court's verdict on 200 generated characters:
#   "stutters"  distinct-4-gram ratio < 0.6
#   "babbles"   diverse enough, but mean NLL > 1.0
#   "speaks"    passes both
# Predict first. The trial decodes with each strategy and measures.
# ---------------------------------------------------------------------------
SOVEREIGN_PROPHECY: dict[str, str | None] = {
    "greedy": None,
    "temperature_0.7_top_k_40": None,
    "temperature_2.0": None,
    "greedy_no_repeat_4gram": None,
}
