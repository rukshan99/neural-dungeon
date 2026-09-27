"""ROOM 8.5 - THE SCRIBE'S INSTRUCTION  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

SFT is pretraining continued on dialogues rendered with a fixed template. The
only genuinely new mechanics are the mask (labels of -100 over the prompt so
the loss trains the answer alone) and the padding (right, labelled -100). The
template itself is plain text because the Chronicler's 72-character alphabet
has no special tokens; ``#`` is not in it, ``-`` is.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from dungeon.artifacts.tiny_gpt import CharTokenizer

USER_TAG = "--- User:\n"
ASSISTANT_TAG = "--- Assistant:\n"
END_TAG = "\n\n"
IGNORE_INDEX = -100
ROLES: tuple[str, str] = ("user", "assistant")

PRICE_QUESTION = "Quote the price of {item}."  # no '?' in the Chronicler's alphabet either
PRICE_ANSWER = "{price} {unit}"

_TAGS = {"user": USER_TAG, "assistant": ASSISTANT_TAG}
_ITEM_LINE = re.compile(r"^item: (?P<item>.+?) \| qty: (?P<qty>\d+) \| price: (?P<price>\d+) (?P<unit>[a-z]+) \| note: (?P<note>.*)$")


def render_chat(turns: list[tuple[str, str]]) -> str:
    """Each turn is ``TAG + text + END_TAG``; unknown roles are an error, not a guess."""
    out = []
    for role, text in turns:
        if role not in _TAGS:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
        out.append(_TAGS[role] + text + END_TAG)
    return "".join(out)


def render_prompt(user_text: str) -> str:
    """One user turn followed by the assistant tag: the model's continuation is the answer."""
    return render_chat([("user", user_text)]) + ASSISTANT_TAG


def build_sft_example(
    prompt_text: str,
    response_text: str,
    tokenizer: CharTokenizer,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(input_ids, labels)``: next-token labels over the response, -100 over the prompt.

    Truncation drops tokens from the front (the prompt side) so the answer and
    its end tag survive whole; the surviving prompt tokens stay masked.
    """
    prompt_ids = tokenizer.encode(render_prompt(prompt_text))
    response_ids = tokenizer.encode(response_text + END_TAG)
    if len(response_ids) >= max_len:
        raise ValueError(
            f"The response ({len(response_ids)} tokens with END_TAG) fills max_len={max_len}: "
            "nothing would be left of the prompt to condition on."
        )
    ids = prompt_ids + response_ids
    n_prompt = len(prompt_ids)
    if len(ids) > max_len:
        dropped = len(ids) - max_len
        ids = ids[dropped:]
        n_prompt -= dropped  # >= 1, because the response alone fits with room to spare
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    # labels[t] is the target of position t, i.e. ids[t + 1]; supervised iff that target is a response token.
    labels[n_prompt - 1 : -1] = input_ids[n_prompt:]
    return input_ids, labels


def collate_sft(examples: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest example: ``pad_id`` in the inputs, ``IGNORE_INDEX`` in the labels."""
    longest = max(len(ids) for ids, _labels in examples)
    input_ids = torch.full((len(examples), longest), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), longest), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, lab) in enumerate(examples):
        input_ids[row, : len(ids)] = ids
        labels[row, : len(lab)] = lab
    return input_ids, labels


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over the positions whose label is not ``IGNORE_INDEX``."""
    vocab = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=IGNORE_INDEX)


def supervised_fraction(labels: torch.Tensor) -> float:
    """How much of the batch the loss actually sees."""
    return (labels != IGNORE_INDEX).float().mean().item()


def make_instruction_pairs(ledger_text: str, n: int) -> list[tuple[str, str]]:
    """Price questions and their answers, read off the first ``n`` item lines of the ledger."""
    pairs: list[tuple[str, str]] = []
    for line in ledger_text.splitlines():
        match = _ITEM_LINE.match(line)
        if match is None:
            continue  # the header and the "-- day N --" separators
        pairs.append((PRICE_QUESTION.format(item=match["item"]), PRICE_ANSWER.format(price=match["price"], unit=match["unit"])))
        if len(pairs) >= n:
            break
    return pairs


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
    """Room 3's loop with masked labels: only the answer tokens produce gradient."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Nothing to train: every parameter is frozen.")
    opt = torch.optim.AdamW(params, lr=lr)
    was_training = model.training
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        ix = torch.randint(0, len(pairs), (batch_size,), generator=generator)
        examples = [build_sft_example(*pairs[i], tokenizer, max_len) for i in ix.tolist()]
        input_ids, labels = collate_sft(examples, pad_id=tokenizer.unk_id)
        logits, _ = model(input_ids)
        loss = sft_loss(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.train(was_training)
    return losses


@torch.no_grad()
def evaluate_sft(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    max_len: int = 128,
    batch_size: int = 32,
) -> float:
    """Token-weighted mean loss over every supervised position of every pair."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(pairs), batch_size):
        examples = [build_sft_example(p, r, tokenizer, max_len) for p, r in pairs[start : start + batch_size]]
        input_ids, labels = collate_sft(examples, pad_id=tokenizer.unk_id)
        logits, _ = model(input_ids)
        vocab = logits.size(-1)
        total += F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum").item()
        count += int((labels != IGNORE_INDEX).sum())
    model.train(was_training)
    return total / max(count, 1)


@torch.no_grad()
def answer(model: nn.Module, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    """Greedy decoding after the assistant tag, cut at the first END_TAG."""
    was_training = model.training
    model.eval()
    idx = torch.tensor([tokenizer.encode(render_prompt(prompt))], dtype=torch.long)
    generated: list[int] = []
    text = ""
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size :])
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        generated.append(int(next_id))
        text = tokenizer.decode(generated)
        if text.endswith(END_TAG):
            text = text[: -len(END_TAG)]
            break
    model.train(was_training)
    return text
