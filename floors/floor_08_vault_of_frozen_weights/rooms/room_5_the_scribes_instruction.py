"""ROOM 8.5 - THE SCRIBE'S INSTRUCTION

    Past the Forgetting, a scribe's desk. A stack of questions on the left, a
    stack of answers on the right, and a Chronicler that only knows how to
    continue whatever text it is shown. Teach it that a question is a text
    whose continuation is an answer.

Supervised fine-tuning (SFT) is not a new kind of training. It is pretraining
continued on text of a particular shape: dialogues rendered with a fixed
*chat template*, so that "continue this text" and "answer this question"
become the same task. Three ideas, each with a trial:

THE TEMPLATE. Real tokenizers have special tokens for this (``<|im_start|>``,
``[INST]``). The Chronicler's has 72 characters and none of them is special,
so the template is plain text (``#`` is not in its alphabet; ``-`` is):

    --- User:\\n<question>\\n\\n--- Assistant:\\n<answer>\\n\\n

It must be identical, character for character, at training and inference
time. The model learns p(answer | exactly this prefix). A missing newline at
inference is a prefix it has never seen.

LOSS MASKING. Every token of the rendered text is an *input*; only the answer
tokens and the end tag that stops it are *targets*. Give every other position
the label ``IGNORE_INDEX = -100`` and ``F.cross_entropy`` skips it (that is
PyTorch's default ``ignore_index``). Without the mask the gradient also
spends itself learning to write questions, which nobody will ask it to do.

RIGHT-PADDING. Examples differ in length; a batch is a rectangle. Pad on the
right with any id and label the pads -100. In a causal model a pad *after*
the real tokens influences nothing that is scored.

Evaluate two ways: the mean loss over held-out *answer* tokens, and reading
what greedy decoding writes after the assistant tag. Both matter.
"""

from __future__ import annotations

import re  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

from dungeon.artifacts.tiny_gpt import CharTokenizer

USER_TAG = "--- User:\n"
ASSISTANT_TAG = "--- Assistant:\n"
END_TAG = "\n\n"
IGNORE_INDEX = -100
ROLES: tuple[str, str] = ("user", "assistant")

PRICE_QUESTION = "Quote the price of {item}."  # no '?' in the Chronicler's alphabet either
PRICE_ANSWER = "{price} {unit}"


def render_chat(turns: list[tuple[str, str]]) -> str:
    """Render ``[(role, text), ...]`` as plain text: each turn is ``TAG + text + END_TAG``.

    ``role`` is ``"user"`` (USER_TAG) or ``"assistant"`` (ASSISTANT_TAG); raise
    ValueError for anything else. An empty list renders as ``""``.
    """
    raise NotImplementedError("render_chat() is unwritten")


def render_prompt(user_text: str) -> str:
    """The text the model continues at inference: one user turn, then ASSISTANT_TAG.

    ``render_chat([("user", q)]) + ASSISTANT_TAG``. It ends with the assistant
    tag so that the model's continuation *is* the assistant's answer. Must equal
    what ``build_sft_example`` puts before the response, or training and
    inference disagree about the prefix.
    """
    raise NotImplementedError("render_prompt() is unwritten")


def build_sft_example(
    prompt_text: str,
    response_text: str,
    tokenizer: CharTokenizer,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One training example: ``(input_ids, labels)``, both 1-D int64 of the same length T.

    ``input_ids`` encodes ``render_prompt(prompt_text) + response_text + END_TAG``.
    ``labels[t]`` is the *next* token ``input_ids[t + 1]`` when that target
    belongs to the response (or the end tag), and ``IGNORE_INDEX`` when the
    target is still part of the prompt. The last position has no target:
    ``labels[T - 1] = IGNORE_INDEX``. So the supervised labels, read in order,
    decode to exactly ``response_text + END_TAG``.

    Truncation policy (document it, then follow it): if the example is longer
    than ``max_len``, drop tokens from the FRONT, i.e. from the prompt side, so
    the response and its end tag survive whole. Whatever prompt tokens remain
    are still masked. If ``response_text + END_TAG`` alone has ``max_len`` or
    more tokens there is nothing left to condition on: raise ValueError.
    """
    raise NotImplementedError("build_sft_example() is unwritten")


def collate_sft(examples: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of ``(input_ids, labels)`` into ``(B, T)`` tensors, T = the longest example.

    Pads in ``input_ids`` are ``pad_id``; pads in ``labels`` are ``IGNORE_INDEX``.
    Any id works as ``pad_id`` because a pad is never a target and sits after
    every real token (causal attention cannot look forward).
    """
    raise NotImplementedError("collate_sft() is unwritten")


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the supervised positions only.

    ``logits`` is ``(B, T, V)``, ``labels`` is ``(B, T)`` with ``IGNORE_INDEX``
    where there is nothing to learn. ``F.cross_entropy(logits.reshape(-1, V),
    labels.reshape(-1), ignore_index=IGNORE_INDEX)``: the mean is over the
    positions that count, not over B * T.
    """
    raise NotImplementedError("sft_loss() is unwritten")


def supervised_fraction(labels: torch.Tensor) -> float:
    """Fraction of positions in ``labels`` that are not ``IGNORE_INDEX``, as a Python float."""
    raise NotImplementedError("supervised_fraction() is unwritten")


def make_instruction_pairs(ledger_text: str, n: int) -> list[tuple[str, str]]:
    """``[(prompt, response), ...]`` for the first ``n`` item lines of the goblin ledger, in order.

    Item lines look like ``item: rusty dagger | qty: 3 | price: 12 copper | note: leaks``.
    Header and ``-- day N ... --`` lines are skipped. Each pair is
    ``(PRICE_QUESTION.format(item=...), PRICE_ANSWER.format(price=..., unit=...))``,
    e.g. ``("Quote the price of rusty dagger.", "12 copper")``. If ``n`` exceeds
    the number of item lines, return all of them. Deterministic: no randomness.
    Every character of every pair must be in the tokenizer's alphabet (the
    ledger's are; a ``?`` would not be).
    """
    raise NotImplementedError("make_instruction_pairs() is unwritten")


def finetune_sft(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    batch_size: int = 16,
    max_len: int = 128,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Train the trainable parameters of ``model`` on the response tokens of ``pairs``.

    Each step: draw ``batch_size`` pair indices with ``torch.randint(...,
    generator=generator)``, ``build_sft_example`` each, ``collate_sft`` with
    ``pad_id=tokenizer.unk_id``, ``logits, _ = model(input_ids)``, ``loss =
    sft_loss(logits, labels)``, AdamW step over ``requires_grad`` parameters
    only (Room 3's rule). Train mode during the steps, previous mode restored
    after. Return the per-step losses as Python floats.
    """
    raise NotImplementedError("finetune_sft() is unwritten")


def evaluate_sft(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    max_len: int = 128,
    batch_size: int = 32,
) -> float:
    """Mean loss per supervised token over ALL pairs (sum of token losses / number of targets).

    Under ``torch.no_grad()``, eval mode, mode restored. Process ``pairs`` in
    chunks of ``batch_size``; use ``F.cross_entropy(..., reduction="sum")`` and
    count the labels that are not ``IGNORE_INDEX`` so that every answer token
    weighs the same regardless of how the chunks fell.
    """
    raise NotImplementedError("evaluate_sft() is unwritten")


def answer(model: nn.Module, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    """Greedy continuation of ``render_prompt(prompt)``, stopped at END_TAG.

    Encode the rendered prompt, then up to ``max_new_tokens`` times: run the
    model on the last ``model.cfg.block_size`` ids, take ``argmax`` of the last
    position's logits, append. Stop as soon as the generated text ends with
    END_TAG and return the text WITHOUT the end tag (never the prompt). No
    grad, eval mode, mode restored.
    """
    raise NotImplementedError("answer() is unwritten")
