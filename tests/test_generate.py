"""Part 4 — sampling and perplexity."""
import math

import pytest
import torch

from src.generate import generate, softmax_with_temperature
from src.llama import Tokenizer
from src.paths import PRETRAINED_LLAMA
from src.pytest_utils.decorators import max_score


@max_score(4)
def test_softmax_with_temperature():
    logits = torch.FloatTensor([[1.0, 2.0, 3.0]])

    probs = softmax_with_temperature(logits, 1.0)
    assert torch.allclose(probs, logits.softmax(-1), atol=1e-6)
    assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-6)

    # low temperature concentrates mass on the argmax
    cold = softmax_with_temperature(logits, 0.01)
    assert cold[0, 2] > 0.99

    # high temperature flattens toward uniform
    hot = softmax_with_temperature(logits, 100.0)
    assert (hot - 1 / 3).abs().max() < 0.05

    # must not divide by zero
    zero = softmax_with_temperature(logits, 0.0)
    assert torch.isfinite(zero).all()
    assert zero[0, 2] > 0.99


@max_score(2)
def test_softmax_with_temperature_is_monotone_in_entropy():
    torch.manual_seed(0)
    logits = torch.randn(1, 50)

    def entropy(t):
        p = softmax_with_temperature(logits, t)
        return -(p * p.clamp_min(1e-12).log()).sum().item()

    entropies = [entropy(t) for t in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(entropies, entropies[1:])), \
        "raising temperature must increase entropy"


@pytest.fixture(scope="module")
def model():
    if not PRETRAINED_LLAMA.exists():
        pytest.skip(f"{PRETRAINED_LLAMA} not present — run `bash setup.sh`")
    from src.llama import load_pretrained

    m = load_pretrained(str(PRETRAINED_LLAMA))
    m.eval()
    return m


@max_score(2)
def test_generate_returns_continuations_and_finite_perplexity(model):
    tokenizer = Tokenizer()
    prefixes = ["Once upon a time there was a little girl who",
                "The dog ran across the field and"]
    gens, ppl = generate(model, "cpu", tokenizer, prefixes,
                         max_new_tokens=16, temperature=0.0)
    assert len(gens) == len(prefixes)
    assert all(isinstance(g, str) and g.strip() for g in gens)
    assert math.isfinite(ppl) and ppl > 1.0, f"implausible perplexity {ppl}"


@max_score(2)
def test_greedy_generation_is_deterministic(model):
    tokenizer = Tokenizer()
    prefixes = ["Once upon a time there was a little girl who"]
    a, _ = generate(model, "cpu", tokenizer, prefixes, max_new_tokens=12, temperature=0.0)
    b, _ = generate(model, "cpu", tokenizer, prefixes, max_new_tokens=12, temperature=0.0)
    assert a == b, "temperature=0.0 must select the argmax, so it cannot vary"


@max_score(2)
def test_top_p_restricts_the_sampling_pool(model):
    """With top_p very small, sampling collapses onto the single most likely
    token, so it must agree with greedy decoding."""
    tokenizer = Tokenizer()
    prefixes = ["Once upon a time there was a little girl who"]
    greedy, _ = generate(model, "cpu", tokenizer, prefixes,
                         max_new_tokens=10, temperature=0.0)
    torch.manual_seed(0)
    nucleus, _ = generate(model, "cpu", tokenizer, prefixes,
                          max_new_tokens=10, temperature=1.0, top_p=1e-6)
    assert greedy == nucleus, \
        "top_p -> 0 should keep only the argmax token in the nucleus"
