"""Part 5 — adapting the pretrained model.

The padding-invariance tests are the important ones. Batches are right-padded to
their longest sequence, so reading hidden_states[:, -1] lands on a PAD token for
every sequence that isn't longest.
"""
import torch
from types import SimpleNamespace

from src.adapt import LlamaEmbeddingClassifier, LlamaZeroShotClassifier
from src.llama import Tokenizer
from src.paths import PRETRAINED_LLAMA
from src.pytest_utils.decorators import max_score

import pytest

LABELS = ["awful", "bad", "average", "good", "excellent"]


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()


def _cfg(tokenizer, option="finetune", **kw):
    base = dict(hidden_dropout_prob=0.0, pretrained_model_path=str(PRETRAINED_LLAMA),
                num_labels=len(LABELS), data_dir=".", option=option,
                pad_id=tokenizer.pad_id)
    base.update(kw)
    return SimpleNamespace(**base)


def _padded_batch(tokenizer, texts):
    enc = [tokenizer.encode(t, bos=True, eos=False) for t in texts]
    width = max(len(e) for e in enc)
    return torch.LongTensor([e + [tokenizer.pad_id] * (width - len(e)) for e in enc])


@pytest.fixture(scope="module")
def _skip_without_checkpoint():
    if not PRETRAINED_LLAMA.exists():
        pytest.skip(f"{PRETRAINED_LLAMA} not present — run `bash setup.sh`")


@max_score(2)
def test_embedding_classifier_returns_log_probabilities(tokenizer, _skip_without_checkpoint):
    model = LlamaEmbeddingClassifier(_cfg(tokenizer)).eval()
    ids = _padded_batch(tokenizer, ["a short one", "a rather longer sentence here"])
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (2, len(LABELS))
    assert torch.allclose(out.exp().sum(-1), torch.ones(2), atol=1e-4), \
        "forward must return log-probabilities (log-softmax), not raw logits"
    assert (out <= 0).all()


@max_score(3)
def test_embedding_classifier_is_padding_invariant(tokenizer, _skip_without_checkpoint):
    """The same sentence must score identically whether or not it is batched
    with a longer one. Reading the final position instead of the last real
    token breaks this."""
    torch.manual_seed(0)
    model = LlamaEmbeddingClassifier(_cfg(tokenizer)).eval()
    short = "the movie was fine"
    alone = _padded_batch(tokenizer, [short])
    with_longer = _padded_batch(
        tokenizer, [short, "a considerably longer sentence that forces extra padding onto the first"]
    )
    with torch.no_grad():
        a = model(alone)[0]
        b = model(with_longer)[0]
    assert torch.allclose(a, b, atol=1e-4), (
        "scores changed when the batch gained a longer sequence — the classifier "
        "is reading a PAD position instead of the last real token"
    )


@max_score(2)
def test_zero_shot_classifier_is_padding_invariant(tokenizer, _skip_without_checkpoint):
    cfg = _cfg(tokenizer, option="prompt")
    cfg.label_names = LABELS
    model = LlamaZeroShotClassifier(cfg, tokenizer, LABELS).eval()
    short = "great film. Is this movie awful, bad, average, good, or excellent? This movie is"
    alone = _padded_batch(tokenizer, [short])
    with_longer = _padded_batch(tokenizer, [short, short + " and then some extra words to pad"])
    with torch.no_grad():
        a = model(alone)[0]
        b = model(with_longer)[0]
    assert torch.allclose(a, b, atol=1e-4), (
        "zero-shot scores changed with batch composition — scoring at the padded "
        "final position rather than the end of the prompt"
    )


@max_score(1)
def test_zero_shot_rejects_multi_token_labels(tokenizer, _skip_without_checkpoint):
    """Summing a label's token log-probs at one position is only valid for
    single-token labels. Fail loudly rather than mis-scoring silently."""
    bad_labels = ["extraordinarily wonderful", "bad"]
    cfg = _cfg(tokenizer, option="prompt", num_labels=2)
    cfg.label_names = bad_labels
    with pytest.raises(AssertionError, match="single token"):
        LlamaZeroShotClassifier(cfg, tokenizer, bad_labels)


@max_score(2)
def test_finetune_option_unfreezes_backbone(tokenizer, _skip_without_checkpoint):
    frozen = LlamaEmbeddingClassifier(_cfg(tokenizer, option="pretrain"))
    unfrozen = LlamaEmbeddingClassifier(_cfg(tokenizer, option="finetune"))
    assert not any(p.requires_grad for p in frozen.llama.parameters()), \
        "option='pretrain' should freeze the backbone (linear probe)"
    assert all(p.requires_grad for p in unfrozen.llama.parameters()), \
        "option='finetune' should train the backbone"
    # the head is trainable either way
    assert all(p.requires_grad for p in frozen.classifier_head.parameters())
