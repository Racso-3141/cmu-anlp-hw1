"""Bonus track — 4-digit addition.

Optional for students, so these carry no autograder points; they exist so staff
know the bonus track still works when the model or optimizer changes.
"""
import random

import torch
from torch.utils.data import DataLoader

from bonus.addition_data_generation import generate_addition_data
from bonus.addition_lib import CharDataset
from bonus.addition_run import _answer_only_loss, evaluate_loss, train_one_epoch
from src.llama import Llama, LlamaConfig
from src.optim import AdamW
from src.pytest_utils.decorators import max_score

CHARS = ['+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']


@max_score(0)
def test_generate_addition_data_is_well_formed():
    random.seed(0)
    data = generate_addition_data(200)
    assert len(data) == 200
    pairs = set()
    for s in data:
        lhs, c = s.split("=")
        a, b = (int(x) for x in lhs.split("+"))
        assert a + b == int(c)
        assert 1000 <= a <= 9999 and 1000 <= b <= 9999
        pairs.add(tuple(sorted((a, b))))
    assert len(pairs) == 200, "(a,b) and (b,a) must count as the same problem"


@max_score(0)
def test_answer_only_loss_ignores_the_question():
    """Targets at and before '=' must be masked out, so changing the model's
    predictions over the question cannot change the loss."""
    words = ["1000+2000=3000"]
    ds = CharDataset(words, CHARS, max(len(w) for w in words))
    x, y = ds[0]
    y = y.unsqueeze(0)

    torch.manual_seed(0)
    logits = torch.randn(1, y.size(1), len(CHARS) + 1)
    base = _answer_only_loss(logits, y)

    eq_pos = (y == 12).float().argmax(dim=1).item()
    perturbed = logits.clone()
    perturbed[:, : eq_pos + 1] = torch.randn_like(perturbed[:, : eq_pos + 1]) * 10
    assert torch.allclose(base, _answer_only_loss(perturbed, y), atol=1e-6), \
        "loss changed when only question-position logits moved — masking is wrong"

    after = logits.clone()
    after[:, eq_pos + 1:] = torch.randn_like(after[:, eq_pos + 1:]) * 10
    assert not torch.allclose(base, _answer_only_loss(after, y), atol=1e-6), \
        "loss ignored the answer positions too — everything is masked"


@max_score(0)
def test_addition_training_reduces_loss():
    random.seed(0)
    torch.manual_seed(0)
    words = generate_addition_data(256)
    ds = CharDataset(words, CHARS, max(len(w) for w in words))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = Llama(LlamaConfig(vocab_size=13, dim=64, n_layers=2, n_heads=4,
                              n_kv_heads=4, max_seq_len=20, multiple_of=32,
                              dropout=0.0))
    opt = AdamW(model.parameters(), lr=1e-3)

    before = evaluate_loss(model, loader, "cpu")
    for _ in range(3):
        train_one_epoch(model, loader, opt, "cpu")
    after = evaluate_loss(model, loader, "cpu")
    assert after < before, f"loss did not drop: {before:.3f} -> {after:.3f}"
