"""Part 2 — the Llama-2 model.

The integration test (`test_llama_forward_matches_reference`) is the one that matters:
it loads real pretrained weights, so a subtly wrong RoPE or normalization produces
garbage rather than merely wrong-shaped tensors.
"""
import numpy as np
import pytest
import torch

from src.llama import LlamaConfig, RMSNorm, apply_rotary_emb
from src.llama.llama import Attention, FeedForward, LlamaLayer
from src.pytest_utils.decorators import max_score

from .conftest import TEST_DATA

# atol=1e-5 is NOT portable: a correct implementation differs from the reference by
# ~3e-5 on a few logits purely from float32 accumulation order across architectures.
ATOL, RTOL = 1e-4, 1e-3

SENT_IDS = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])


def _config(**kw):
    defaults = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=512,
                    multiple_of=32, max_seq_len=32, dropout=0.0)
    defaults.update(kw)
    return LlamaConfig(**defaults)


@max_score(3)
def test_rmsnorm_rescales_without_recentering():
    """RMSNorm divides by root-mean-square and does NOT subtract the mean.
    A standard LayerNorm would zero the mean; RMSNorm must not."""
    norm = RMSNorm(8, eps=1e-6)
    x = torch.arange(8, dtype=torch.float32).repeat(2, 1) + 1.0
    out = norm._norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), \
        "output should have unit root-mean-square"
    assert out.mean(-1).abs().min() > 1e-3, \
        "RMSNorm must not re-center: the output mean should not be zero"


@max_score(1)
def test_rmsnorm_has_no_bias():
    """Llama-2's norm has a learnable gain only. A bias term would not exist in
    the pretrained checkpoint and would silently load as zeros."""
    norm = RMSNorm(8)
    names = dict(norm.named_parameters())
    assert "weight" in names
    assert "bias" not in names, "RMSNorm has no bias parameter"


@max_score(4)
def test_apply_rotary_emb_matches_reference():
    torch.manual_seed(0)
    query = 2 * torch.ones([1, 2, 2, 4])
    key = 3 * torch.ones([1, 2, 2, 4])
    got_q, got_k = apply_rotary_emb(query, key, 4, 20)
    ref_q, ref_k = torch.load(TEST_DATA / "rotary_embedding_actual.data", weights_only=False)
    assert torch.allclose(ref_q, got_q, atol=ATOL, rtol=RTOL)
    assert torch.allclose(ref_k, got_k, atol=ATOL, rtol=RTOL)


@max_score(1)
def test_apply_rotary_emb_leaves_position_zero_unchanged():
    """Position 0 has rotation angle 0, so RoPE must be the identity there."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    got_q, got_k = apply_rotary_emb(q, k, 8, 16)
    assert torch.allclose(got_q[:, 0], q[:, 0], atol=1e-5)
    assert torch.allclose(got_k[:, 0], k[:, 0], atol=1e-5)


@max_score(3)
def test_attention_is_causal():
    """Changing a later token must not change an earlier token's output."""
    torch.manual_seed(0)
    cfg = _config()
    attn = Attention(cfg).eval()
    x = torch.randn(1, 8, cfg.dim)
    with torch.no_grad():
        base = attn(x)
        perturbed = x.clone()
        perturbed[:, -1] += 10.0
        after = attn(perturbed)
    assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-5), \
        "earlier positions changed when a later token changed — mask is not causal"
    assert not torch.allclose(base[:, -1], after[:, -1], atol=1e-5)


@max_score(2)
def test_llama_layer_is_residual():
    """With zeroed output projections the block must be the identity."""
    torch.manual_seed(0)
    cfg = _config()
    layer = LlamaLayer(0, cfg).eval()
    with torch.no_grad():
        layer.attention.compute_output.weight.zero_()
        layer.feed_forward.w2.weight.zero_()
        x = torch.randn(1, 6, cfg.dim)
        assert torch.allclose(layer(x), x, atol=1e-5), \
            "block must add residual connections around attention and the FFN"


@max_score(7)
def test_llama_forward_matches_reference(pretrained_llama, sanity_reference):
    """Integration test against real pretrained weights."""
    with torch.no_grad():
        logits, hidden = pretrained_llama(SENT_IDS)
    for name, got, want in [("logits", logits, sanity_reference["logits"]),
                            ("hidden_states", hidden, sanity_reference["hidden_states"])]:
        diff = (got - want).abs()
        n_bad = (diff > ATOL + RTOL * want.abs()).sum().item()
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            f"{name} mismatch: max abs diff {diff.max().item():.3e}, "
            f"{n_bad}/{diff.numel()} elements outside tolerance. "
            f"~1e+00 suggests a wrong normalization or attention; "
            f"~1e-4 is numerical noise."
        )


@max_score(3)
def test_generation_is_coherent(pretrained_llama):
    """Greedy continuation of an in-distribution prompt should be real English.
    A wrong RoPE still produces the right shapes but yields word salad."""
    from src.llama import Tokenizer

    enc = Tokenizer()
    prompt = "Once upon a time, there was a little girl named Lily who"
    ids = torch.tensor(enc.encode(prompt, bos=True, eos=False))[None, ...]
    out = pretrained_llama.generate(ids, 40, temperature=0.0)
    text = enc.decode(out[0].tolist())
    assert text.startswith(prompt.strip()[:20]), "continuation should extend the prompt"
    continuation = text[len(prompt):]
    words = [w for w in continuation.split() if w.isalpha()]
    assert len(words) >= 10, f"expected fluent continuation, got: {continuation!r}"
