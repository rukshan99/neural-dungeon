"""TRIAL 12.2 - THE QUANTIZER'S BENCH

Every weight within half a notch of where it started, the loud row kept to
itself, and a Chronicler that still knows the chronicles at a quarter the size.
"""

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch.nn as nn  # noqa: E402

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_2_quantizers_bench")

torch.manual_seed(122)
MODEL, TOK, _ = load_pretrained()
CFG = MODEL.cfg
TEXT = read_corpus()
LOSS_DELTA = 0.05  # measured ~2e-4 on this checkpoint; the bar is generous
AGREE = 30  # greedy tokens that must agree; measured 40/40

PROMPTS = [
    "Ruel carried a mirror that showed the validation set. ",
    "Below them, the Scriptorium of Tokens. ",
    "Kasimir retreated. Kasimir waited.",
]


def _spiky_matrix() -> torch.Tensor:
    """(8, 16): quiet rows of scale ~0.05 and one row ten times louder."""
    g = torch.Generator().manual_seed(7)
    w = torch.randn(8, 16, generator=g) * 0.05
    w[3] *= 10.0
    return w


def _fixed_batch():
    ids = torch.tensor(TOK.encode(TEXT[:20000]), dtype=torch.long)
    g = torch.Generator().manual_seed(12)
    starts = torch.randint(0, len(ids) - 65, (16,), generator=g)
    x = torch.stack([ids[s:s + 64] for s in starts])
    y = torch.stack([ids[s + 1:s + 65] for s in starts])
    return x, y


# ------------------------------------------------------------------ quantize / dequantize
def test_codes_are_int8_and_scales_are_float32_columns():
    w = torch.randn(6, 10)
    q, scale = room.quantize_int8(w, per_channel=True)
    assert q.dtype == torch.int8, f"codes must be torch.int8, got {q.dtype}"
    assert q.shape == w.shape, f"codes keep the weight's shape {tuple(w.shape)}, got {tuple(q.shape)}"
    assert scale.dtype == torch.float32, f"scale must be float32, got {scale.dtype}"
    assert scale.shape == (6, 1), f"per-channel scale is one column per output row: (6, 1), got {tuple(scale.shape)}"
    q_t, scale_t = room.quantize_int8(w, per_channel=False)
    assert scale_t.shape == (1, 1), f"per-tensor scale is a single (1, 1) value, got {tuple(scale_t.shape)}"
    assert int(q.abs().max()) <= 127 and int(q_t.abs().max()) <= 127, "Symmetric int8 stays within [-127, 127]."


def test_every_weight_is_within_half_a_notch():
    w = _spiky_matrix()
    for per_channel in (True, False):
        q, scale = room.quantize_int8(w, per_channel=per_channel)
        err = (w - room.dequantize(q, scale)).abs()
        bound = scale / 2 + 1e-6  # broadcasts per row or per matrix
        assert bool((err <= bound).all()), (
            f"per_channel={per_channel}: max |w - q*scale| is {float(err.max()):.5f} but round-to-nearest "
            f"guarantees <= scale/2 = {float(scale.max()) / 2:.5f}. Are you rounding, and dequantizing with the same scale?"
        )


def test_the_loudest_weight_of_every_row_lands_on_127():
    w = _spiky_matrix()
    q, _ = room.quantize_int8(w, per_channel=True)
    peaks = q.abs().amax(dim=1)
    assert bool((peaks == 127).all()), (
        f"Per-channel absmax scaling maps each row's largest |w| to exactly 127; your rows peak at {peaks.tolist()}."
    )


def test_per_channel_keeps_the_loud_row_to_itself():
    w = _spiky_matrix()
    quiet = [r for r in range(8) if r != 3]
    q_c, s_c = room.quantize_int8(w, per_channel=True)
    q_t, s_t = room.quantize_int8(w, per_channel=False)
    err_c = (w - room.dequantize(q_c, s_c)).abs()[quiet].max()
    err_t = (w - room.dequantize(q_t, s_t)).abs()[quiet].max()
    assert err_t > 3 * err_c, (
        f"On the quiet rows per-tensor error is {float(err_t):.5f} and per-channel is {float(err_c):.5f}. "
        "One loud row should force a coarse grid on everyone under per-tensor; per-channel gives each row its own scale."
    )
    stats = room.quantization_error(w, q_t, s_t)
    assert set(stats) >= {"max_abs_error", "relative_error"}, f"quantization_error keys: {sorted(stats)}"
    assert abs(stats["max_abs_error"] - float((w - room.dequantize(q_t, s_t)).abs().max())) < 1e-6
    assert 0.0 < stats["relative_error"] < 1.0


def test_the_chroniclers_own_weights_survive_with_small_relative_error():
    for name, module in MODEL.named_modules():
        if isinstance(module, nn.Linear) and name != "lm_head":
            q, s = room.quantize_int8(module.weight, per_channel=True)
            rel = room.quantization_error(module.weight, q, s)["relative_error"]
            assert rel < 0.05, f"{name}: relative error {rel:.4f} is too large for int8 per-channel (expect ~1%)."


# ------------------------------------------------------------------ QuantizedLinear
def test_quantized_linear_stores_int8_and_answers_like_its_fp32_twin():
    linear = MODEL.blocks[0].mlp.fc
    qlin = room.QuantizedLinear(linear, per_channel=True)
    assert isinstance(qlin, nn.Module), "QuantizedLinear must be an nn.Module."
    int8_buffers = [b for b in qlin.buffers() if b.dtype == torch.int8]
    assert int8_buffers, "Store the codes as an int8 buffer (register_buffer), not as a float Parameter."
    assert int8_buffers[0].shape == linear.weight.shape
    assert qlin.in_features == linear.in_features and qlin.out_features == linear.out_features
    x = torch.randn(3, 5, linear.in_features)
    with torch.no_grad():
        ref = linear(x)
        got = qlin(x)
    assert got.shape == ref.shape and got.dtype == torch.float32
    assert torch.allclose(got, ref, atol=0.05), (
        f"QuantizedLinear output differs from the fp32 linear by up to {(got - ref).abs().max():.4f}. "
        "forward = F.linear(x, dequantize(weight_q, scale), bias). Did you keep the bias in float32?"
    )
    q_bytes = sum(b.numel() * b.element_size() for b in int8_buffers)
    w_bytes = linear.weight.numel() * linear.weight.element_size()
    assert q_bytes * 4 == w_bytes, f"int8 codes take {q_bytes} bytes; the fp32 weight takes {w_bytes}. Expected a quarter."
    float_copies = [t for t in list(qlin.buffers()) + list(qlin.parameters()) if t.dtype != torch.int8 and t.shape == linear.weight.shape]
    assert not float_copies, (
        "QuantizedLinear also keeps a full-size float tensor of the weight's shape, which is the 4x you were meant to "
        "save. Store the int8 codes and the scale only, and dequantize inside forward."
    )


def test_a_linear_without_bias_is_welcome_too():
    linear = nn.Linear(8, 4, bias=False)
    qlin = room.QuantizedLinear(linear)
    x = torch.randn(2, 8)
    with torch.no_grad():
        assert torch.allclose(qlin(x), linear(x), atol=0.05)


# ------------------------------------------------------------------ quantize_model_linears
def test_sixteen_linears_are_replaced_and_the_tied_head_is_spared():
    qmodel, n = room.quantize_model_linears(MODEL, per_channel=True)
    assert n == 16, f"4 blocks x (c_attn, c_proj, fc, proj) = 16 linears to replace; you replaced {n}."
    replaced = [name for name, m in qmodel.named_modules() if isinstance(m, room.QuantizedLinear)]
    assert len(replaced) == 16, f"The copy holds {len(replaced)} QuantizedLinear modules: {replaced}"
    assert isinstance(qmodel.lm_head, nn.Linear) and not isinstance(qmodel.lm_head, room.QuantizedLinear), (
        "lm_head must stay an fp32 nn.Linear: its weight is wte.weight."
    )
    assert qmodel.lm_head.weight is qmodel.wte.weight, "The copy broke the wte/lm_head tie. copy.deepcopy keeps it."
    assert not any(isinstance(m, room.QuantizedLinear) for m in MODEL.modules()), (
        "The ORIGINAL model was modified. Quantize a deep copy."
    )
    assert qmodel is not MODEL


def test_the_quantized_chronicler_still_knows_the_chronicles():
    x, y = _fixed_batch()
    qmodel, _ = room.quantize_model_linears(MODEL, per_channel=True)
    with torch.no_grad():
        _, loss_fp32 = MODEL(x, y)
        _, loss_q = qmodel(x, y)
    delta = float(loss_q - loss_fp32)
    assert abs(delta) < LOSS_DELTA, (
        f"fp32 loss {loss_fp32:.4f}, int8 loss {loss_q:.4f}: a delta of {delta:+.4f} means something was melted "
        f"down wrong (expected within {LOSS_DELTA})."
    )


@pytest.mark.parametrize("prompt", PROMPTS, ids=["ruel", "scriptorium", "kasimir"])
def test_greedy_generations_agree_for_the_first_tokens(prompt):
    idx = torch.tensor([TOK.encode(prompt)], dtype=torch.long)
    qmodel, _ = room.quantize_model_linears(MODEL, per_channel=True)
    with torch.no_grad():
        ref = MODEL.generate(idx, AGREE, temperature=0)[0, idx.shape[1]:]
        got = qmodel.generate(idx, AGREE, temperature=0)[0, idx.shape[1]:]
    assert torch.equal(ref, got), (
        f"The int8 Chronicler wrote {TOK.decode(got)!r}; the fp32 one wrote {TOK.decode(ref)!r}. "
        f"They should agree for at least {AGREE} tokens."
    )


def test_int8_storage_is_a_quarter_of_fp32():
    qmodel, _ = room.quantize_model_linears(MODEL)
    fp32_bytes = sum(m.weight.numel() * 4 for n, m in MODEL.named_modules() if isinstance(m, nn.Linear) and n != "lm_head")
    int8_bytes = sum(b.numel() * b.element_size() for m in qmodel.modules() if isinstance(m, room.QuantizedLinear)
                     for b in m.buffers() if b.dtype == torch.int8)
    assert int8_bytes * 4 == fp32_bytes, f"{int8_bytes} int8 bytes vs {fp32_bytes} fp32 bytes: expected exactly 1/4."


# ------------------------------------------------------------------ the prophecy
PROPHECY_KEYS = {
    "int8_weights_per_fp32_weight",
    "larger_error_on_the_quiet_rows",
    "code_of_each_rows_absmax",
    "tensor_tied_to_lm_head",
    "dequant_on_the_fly_saves",
}


def test_the_prophecy_is_complete():
    assert set(room.QUANT_PROPHECY) == PROPHECY_KEYS, "Do not add or remove prophecy lines; answer them."


def _prediction(key):
    value = room.QUANT_PROPHECY.get(key)
    if value is None:
        raise NotImplementedError(f"QUANT_PROPHECY[{key!r}] is still None. Predict first, then run.")
    return value


def test_prophecy_int8_weights_per_fp32_weight():
    measured = torch.tensor(0.0, dtype=torch.float32).element_size() // torch.tensor(0, dtype=torch.int8).element_size()
    assert _prediction("int8_weights_per_fp32_weight") == measured, f"An fp32 weight is {measured} bytes; an int8 code is 1."


def test_prophecy_larger_error_on_the_quiet_rows():
    w = _spiky_matrix()
    quiet = [r for r in range(8) if r != 3]
    q_c, s_c = room.quantize_int8(w, True)
    q_t, s_t = room.quantize_int8(w, False)
    err_c = float((w - room.dequantize(q_c, s_c)).abs()[quiet].max())
    err_t = float((w - room.dequantize(q_t, s_t)).abs()[quiet].max())
    truth = "per-tensor" if err_t > err_c else ("per-channel" if err_c > err_t else "same")
    assert _prediction("larger_error_on_the_quiet_rows") == truth, (
        f"Measured: per-tensor {err_t:.5f} vs per-channel {err_c:.5f} on the quiet rows -> {truth!r}."
    )


def test_prophecy_code_of_each_rows_absmax():
    q, _ = room.quantize_int8(_spiky_matrix(), True)
    measured = int(q.abs().amax(dim=1)[0])
    assert _prediction("code_of_each_rows_absmax") == measured, f"Every row's absmax maps to |code| = {measured}."


def test_prophecy_tensor_tied_to_lm_head():
    name = _prediction("tensor_tied_to_lm_head")
    tied = getattr(MODEL, str(name), None)
    assert tied is not None and getattr(tied, "weight", None) is MODEL.lm_head.weight, (
        f"model.{name} is not the tensor tied to lm_head. Look at how GPT.__init__ ties the head."
    )


def test_prophecy_dequant_on_the_fly_saves():
    assert _prediction("dequant_on_the_fly_saves") == "memory", (
        "Dequantizing on the fly runs the same float32 matmul on a float32 copy of W: "
        "the weight is stored smaller, the arithmetic is unchanged. Compute savings need an int8 kernel."
    )
