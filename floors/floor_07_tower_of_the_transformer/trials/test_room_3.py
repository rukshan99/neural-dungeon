"""TRIAL 7.3 - THE TOWER

The GPT is checked for its parts, its weight tying, its parameter count, its
initialisation, its loss, and finally it must wear the Chronicler's shipped
checkpoint and produce the reference's loss to five decimal places.
"""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_3_the_tower")

torch.manual_seed(73)

PAYLOAD = torch.load(tiny_gpt.CHECKPOINT, map_location="cpu")
CHRONICLER_CFG = tiny_gpt.GPTConfig(**PAYLOAD["config"])
TOKENIZER = tiny_gpt.CharTokenizer(PAYLOAD["chars"])
SMALL = tiny_gpt.GPTConfig(vocab_size=11, block_size=16, n_layer=2, n_head=2, n_embd=16)


def _fixed_batch(block_size=128, batch_size=8, seed=3):
    text = tiny_gpt.read_corpus("chronicles")
    data = torch.tensor(TOKENIZER.encode(text), dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=g)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


# ---------------------------------------------------------------- structure
def test_the_tower_has_its_named_parts():
    model = room.GPT(SMALL)
    assert isinstance(getattr(model, "wte", None), nn.Embedding) and model.wte.weight.shape == (11, 16), "wte: nn.Embedding(vocab_size, n_embd)."
    assert isinstance(getattr(model, "wpe", None), nn.Embedding) and model.wpe.weight.shape == (16, 16), "wpe: nn.Embedding(block_size, n_embd)."
    assert isinstance(getattr(model, "drop", None), nn.Dropout), "drop: nn.Dropout(cfg.dropout) applied to the summed embeddings."
    assert isinstance(getattr(model, "blocks", None), nn.ModuleList) and len(model.blocks) == 2, "blocks: an nn.ModuleList of n_layer Blocks."
    assert getattr(model, "ln_f", None) is not None and model.ln_f.weight.shape == (16,), "ln_f: the final LayerNorm over n_embd."
    assert isinstance(getattr(model, "lm_head", None), nn.Linear), "lm_head: nn.Linear(n_embd, vocab_size, bias=False)."
    assert model.lm_head.bias is None, "lm_head has no bias. Its weight is the embedding matrix, which has no bias either."
    assert (model.lm_head.in_features, model.lm_head.out_features) == (16, 11)


def test_the_lm_head_and_the_token_embedding_are_one_tensor():
    model = room.GPT(SMALL)
    assert model.lm_head.weight is model.wte.weight, (
        "Weight tying means the SAME Parameter object: `self.lm_head.weight = self.wte.weight`. "
        "Equal values are not enough; a copy would learn separately and double the embedding parameters."
    )


def test_the_chronicler_has_802560_non_embedding_parameters():
    model = room.GPT(CHRONICLER_CFG)
    n = model.num_params(non_embedding=True)
    assert n == 802_560, (
        f"num_params(non_embedding=True) should be 802,560 for the Chronicler's config, got {n:,}. "
        "Count each parameter once (a tied weight is one tensor), then subtract wpe. Per block: 12D^2 + 13D."
    )
    total = model.num_params(non_embedding=False)
    assert total == 818_944, f"With the position embedding included the count is 818,944, got {total:,}."


# --------------------------------------------------------------- initialisation
def test_a_fresh_tower_is_initialised_like_gpt2():
    torch.manual_seed(0)
    model = room.GPT(CHRONICLER_CFG)
    n_layer = CHRONICLER_CFG.n_layer
    resid_std = 0.02 / math.sqrt(2 * n_layer)
    assert abs(model.wte.weight.std().item() - 0.02) < 0.003, f"wte should be N(0, 0.02); its std is {model.wte.weight.std().item():.4f}."
    assert abs(model.wpe.weight.std().item() - 0.02) < 0.003, f"wpe should be N(0, 0.02); its std is {model.wpe.weight.std().item():.4f}."
    for name, p in model.named_parameters():
        if name.endswith(".bias") and "ln" not in name:
            assert torch.all(p == 0), f"Linear biases start at zero; {name} does not."
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m is not model.lm_head:
            std = m.weight.std().item()
            if name.endswith("attn.c_proj") or name.endswith("mlp.proj"):
                assert abs(std - resid_std) / resid_std < 0.15, (
                    f"{name}.weight writes into the residual stream, so its std should be 0.02/sqrt(2*n_layer) = {resid_std:.4f}; got {std:.4f}."
                )
            else:
                assert abs(std - 0.02) / 0.02 < 0.15, f"{name}.weight should be N(0, 0.02); its std is {std:.4f}."


def test_a_newborn_tower_is_maximally_uncertain():
    torch.manual_seed(1)
    model = room.GPT(CHRONICLER_CFG).eval()
    x, y = _fixed_batch()
    with torch.no_grad():
        _, loss = model(x, y)
    expected = math.log(CHRONICLER_CFG.vocab_size)
    assert abs(loss.item() - expected) < 0.2, (
        f"A freshly initialised model should be close to guessing uniformly: loss about ln(V) = {expected:.3f}; "
        f"got {loss.item():.3f}. Weights that are too large (std 1 instead of 0.02) make it confidently wrong from birth."
    )


# ---------------------------------------------------------------------- forward
def test_forward_returns_logits_and_no_loss_without_targets():
    model = room.GPT(SMALL).eval()
    idx = torch.randint(0, 11, (3, 9))
    with torch.no_grad():
        logits, loss = model(idx)
    assert logits.shape == (3, 9, 11), f"logits must be (B, T, vocab_size) = (3, 9, 11); got {tuple(logits.shape)}."
    assert loss is None, "Without targets, the second return value is None."


def test_the_loss_is_mean_cross_entropy_over_every_position():
    model = room.GPT(SMALL).eval()
    idx = torch.randint(0, 11, (3, 9))
    targets = torch.randint(0, 11, (3, 9))
    with torch.no_grad():
        logits, loss = model(idx, targets)
    want = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    assert torch.isclose(loss, want, atol=1e-6), (
        f"loss should be F.cross_entropy over ALL B*T positions at once (flatten to (B*T, V) and (B*T,)): "
        f"expected {want.item():.6f}, got {loss.item():.6f}."
    )


def test_positions_are_told_apart():
    model = room.GPT(SMALL).eval()
    idx = torch.full((1, 6), 4)  # the same token six times
    with torch.no_grad():
        logits, _ = model(idx)
    spread = (logits[0, 0] - logits[0, 5]).abs().max().item()
    assert spread > 1e-4, (
        "Six copies of the same token produced identical logits at positions 0 and 5. Without the position "
        "embedding the model cannot tell where it is: x = wte(idx) + wpe(arange(T))."
    )


def test_too_long_a_sequence_is_refused():
    model = room.GPT(SMALL)
    with pytest.raises(ValueError):
        model(torch.zeros(1, SMALL.block_size + 1, dtype=torch.long))


def test_dropout_sleeps_in_eval_and_wakes_in_train():
    cfg = tiny_gpt.GPTConfig(vocab_size=11, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.5)
    model = room.GPT(cfg)
    idx = torch.randint(0, 11, (2, 8))
    model.eval()
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx)
    assert torch.equal(a, b), "In eval() mode the model must be deterministic: dropout is off."
    model.train()
    with torch.no_grad():
        c, _ = model(idx)
        d, _ = model(idx)
    assert not torch.equal(c, d), "In train() mode with dropout=0.5 two forward passes should differ. Is cfg.dropout wired into nn.Dropout?"


def test_a_duck_typed_config_is_welcome():
    cfg = SimpleNamespace(vocab_size=11, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0, bias=True)
    model = room.GPT(cfg).eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, 11, (1, 5)))
    assert logits.shape == (1, 5, 11), "Any object with the config's attributes should do; do not require the GPTConfig class."


# --------------------------------------------------------------- the checkpoint
def test_the_tower_wears_the_chroniclers_checkpoint_and_speaks_with_its_voice():
    mine = room.GPT(CHRONICLER_CFG)
    try:
        mine.load_state_dict(PAYLOAD["state_dict"], strict=True)
    except RuntimeError as exc:
        pytest.fail(
            "The Chronicler's checkpoint does not fit your tower: " + str(exc).splitlines()[0]
            + " Names must follow wte, wpe, blocks.N.ln_1/attn.c_attn/attn.c_proj/ln_2/mlp.fc/mlp.proj, ln_f, lm_head."
        )
    mine.eval()
    reference, _, _ = tiny_gpt.load_pretrained()
    x, y = _fixed_batch()
    with torch.no_grad():
        my_logits, my_loss = mine(x, y)
        ref_logits, ref_loss = reference(x, y)
    assert abs(my_loss.item() - ref_loss.item()) < 1e-5, (
        f"With the Chronicler's weights loaded, your loss on a fixed batch is {my_loss.item():.6f}; "
        f"the reference gets {ref_loss.item():.6f}. Same weights, different architecture: check the forward "
        "order (embeddings, drop, blocks, ln_f, lm_head) and that the position ids are arange(T)."
    )
    assert (my_logits - ref_logits).abs().max().item() < 1e-4, "Logits differ from the reference's by more than 1e-4."
    assert ref_loss.item() < 0.5, "Dungeon self-check: the reference Chronicler should have low loss on its own chronicles."
