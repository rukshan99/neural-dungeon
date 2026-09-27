"""SECRET - THE REPLAY WELL

Two more charms against forgetting: an exact replay schedule, and elastic weight
consolidation, which makes the weights that mattered to the old text stiff.
"""

import copy
import math

import pytest

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn

well = load_room(__file__, "secret_the_replay_well")

pytestmark = pytest.mark.secret

STEPS = 40
LR = 1e-3
LAM = 1e5  # the Fisher diagonal averages ~1e-6 on this model, so lambda has to be large to bite


@pytest.fixture(scope="module")
def chronicler():
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


@pytest.fixture(scope="module")
def corpora():
    return read_corpus("chronicles"), read_corpus("ledger")


_ENCODED: dict[str, "torch.Tensor"] = {}


def _data(tokenizer, text):
    if text not in _ENCODED:
        _ENCODED[text] = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return _ENCODED[text]


@torch.no_grad()
def _loss_on(model, tokenizer, text, n_batches=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    data = _data(tokenizer, text)
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, len(data) - 65, (16,), generator=gen)
        x = torch.stack([data[i : i + 64] for i in ix])
        y = torch.stack([data[i + 1 : i + 65] for i in ix])
        total += model(x, y)[1].item()
    model.train(was_training)
    return total / n_batches


# ------------------------------------------------------------- mixed batches
def _take(stream, n):
    it = iter(stream)
    return [next(it) for _ in range(n)]


@pytest.mark.parametrize("ratio,n,expected_old", [(0.3, 40, 12), (0.25, 40, 10), (0.5, 21, 10), (0.0, 20, 0), (1.0, 20, 20)])
def test_the_replay_ratio_is_exact_over_many_batches(chronicler, corpora, ratio, n, expected_old):
    tokenizer = chronicler[1]
    batches = _take(well.mixed_batches(corpora[1], corpora[0], ratio, tokenizer, 32, 4, torch.Generator().manual_seed(1)), n)
    sources = [b[2] for b in batches]
    assert set(sources) <= {"old", "new"}, f"Each batch is (x, y, source) with source 'old' or 'new'; saw {set(sources)}."
    n_old = sources.count("old")
    assert n_old == expected_old, (
        f"ratio={ratio} over {n} batches means exactly floor({n} * {ratio}) = {expected_old} replay batches; you produced {n_old}. "
        "Use a deterministic schedule (batch i is old when floor((i+1)*ratio) > floor(i*ratio)), not a coin flip."
    )


def test_replay_batches_really_come_from_the_old_text(chronicler, corpora):
    tokenizer = chronicler[1]
    chron, ledger = corpora
    for x, y, source in _take(well.mixed_batches(ledger, chron, 0.5, tokenizer, 32, 3, torch.Generator().manual_seed(2)), 12):
        assert tuple(x.shape) == (3, 32) and torch.equal(x[:, 1:], y[:, :-1]), "Batches must be (batch_size, block_size) shifted windows, like make_batches."
        corpus = chron if source == "old" else ledger
        for row in range(3):
            assert tokenizer.decode(x[row].tolist()) in corpus, f"A batch tagged {source!r} contains a window that is not from that corpus."


# ------------------------------------------------------------- fisher
@pytest.fixture(scope="module")
def fisher(chronicler, corpora):
    model = copy.deepcopy(chronicler[0])
    return well.fisher_diagonal(model, corpora[0], chronicler[1], n_batches=8, generator=torch.Generator().manual_seed(3))


def test_the_fisher_diagonal_has_one_nonnegative_entry_per_weight(chronicler, fisher):
    model = chronicler[0]
    names = {n for n, _p in model.named_parameters()}
    assert set(fisher) == names, "fisher_diagonal() returns {name: tensor} for every trainable parameter."
    for name, p in model.named_parameters():
        assert tuple(fisher[name].shape) == tuple(p.shape), f"fisher[{name}] must have the parameter's shape {tuple(p.shape)}."
        assert torch.all(fisher[name] >= 0), f"fisher[{name}] has negative entries; it is a mean of squared gradients."
    total = sum(f.sum().item() for f in fisher.values())
    assert 0.01 < total < 100, f"The Fisher mass on this model is ~0.7 in total; yours sums to {total:.3g}. Average grad**2 over the batches; do not sum over batches without dividing."
    assert all(p.grad is None for p in chronicler[0].parameters()), "Do not leave gradients on the model after estimating the Fisher."


# ------------------------------------------------------------- penalty
def test_the_penalty_is_zero_at_the_snapshot(chronicler, fisher):
    model = copy.deepcopy(chronicler[0])
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    pen = well.ewc_penalty(model, snap, fisher, lam=LAM)
    assert isinstance(pen, torch.Tensor) and pen.ndim == 0, "ewc_penalty returns a scalar tensor (so it can be added to the loss)."
    assert pen.item() == 0.0, f"At the snapshot every (w - w*) is zero, so the penalty is exactly 0; got {pen.item():.3e}."
    assert pen.requires_grad, "The penalty must stay attached to the graph: it is differentiated with the loss."


def test_the_penalty_grows_quadratically_and_scales_with_lambda(chronicler, fisher):
    model = copy.deepcopy(chronicler[0])
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    p1 = well.ewc_penalty(model, snap, fisher, lam=1.0).item()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    p2 = well.ewc_penalty(model, snap, fisher, lam=1.0).item()
    assert p1 > 0
    assert p2 / p1 == pytest.approx(4.0, rel=1e-3), f"Doubling the displacement should quadruple the penalty (it is quadratic); ratio was {p2 / p1:.4f}."
    p2_lam = well.ewc_penalty(model, snap, fisher, lam=10.0).item()
    assert p2_lam == pytest.approx(10 * p2, rel=1e-5), "The penalty is linear in lambda."
    expected = 0.5 * sum((fisher[n] * 0.02**2).sum().item() for n in fisher)
    assert p2 == pytest.approx(expected, rel=1e-3), f"Expected (lam/2) * sum F_i (w_i - w*_i)^2 = {expected:.4e}; got {p2:.4e}. Do not forget the 1/2."


# ------------------------------------------------------------- the well works
def test_ewc_forgets_less_than_a_plain_fine_tune(chronicler, corpora, fisher):
    model, tokenizer = chronicler
    chron, ledger = corpora
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}

    plain = copy.deepcopy(model)
    torch.manual_seed(5)
    well.finetune_with_ewc(plain, ledger, tokenizer, STEPS, LR, snap, fisher, lam=0.0, generator=torch.Generator().manual_seed(5))
    elastic = copy.deepcopy(model)
    torch.manual_seed(5)
    losses = well.finetune_with_ewc(elastic, ledger, tokenizer, STEPS, LR, snap, fisher, lam=LAM, generator=torch.Generator().manual_seed(5))

    assert len(losses) == STEPS and all(isinstance(v, float) for v in losses), "Return one task-loss float per step."
    old_plain, old_ewc = _loss_on(plain, tokenizer, chron), _loss_on(elastic, tokenizer, chron)
    new_ewc = _loss_on(elastic, tokenizer, ledger)
    assert not math.isnan(old_ewc), "The EWC run produced NaN. lambda * Fisher * displacement got out of hand; check the 1/2 and the mean over batches."
    assert old_ewc < old_plain - 0.3, (
        f"Same lr, same steps, same batches: plain fine-tune left the chronicles at {old_plain:.2f}, EWC at {old_ewc:.2f}. "
        f"EWC should hold the important weights back by a clear margin (reference: ~1.8 vs ~2.65). Is the penalty added to the loss BEFORE backward()?"
    )
    assert new_ewc < 1.5, f"EWC must still learn the ledger (loss < 1.5); got {new_ewc:.2f}. Lambda so large that nothing moves is just freezing."
