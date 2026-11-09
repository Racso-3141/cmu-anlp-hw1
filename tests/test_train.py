"""Part 3 — the training loop.

The sanity check at the bottom is the important one: it carries no points but
gates Q3.4-3.6.
"""
import math
from contextlib import nullcontext

import torch
import wandb

from src.llama import Llama, LlamaConfig
from src.optim import AdamW
from src.pytest_utils.decorators import max_score
from src.train import (
    compute_language_modeling_loss,
    cosine_lr_schedule,
    evaluate,
    random_batch_sampler,
    sequential_batch_sampler,
    train,
)

wandb.init(mode="disabled")


def _model(n_vocab=8, dim=16, n_heads=4, n_kv_heads=2, max_seq_len=16, n_layers=2):
    return Llama(LlamaConfig(vocab_size=n_vocab, dim=dim, n_heads=n_heads,
                             n_kv_heads=n_kv_heads, max_seq_len=max_seq_len,
                             n_layers=n_layers, dropout=0.0, multiple_of=8))


@max_score(2)
def test_random_batch_sampler():
    tokens = torch.arange(100)
    sampler = random_batch_sampler(tokens, "cpu", batch_size=5, seq_len=7)
    batch = next(sampler)
    assert batch.shape == (5, 7)
    # every row must be a contiguous run of the source tokens
    for row in batch:
        assert torch.equal(row, torch.arange(row[0].item(), row[0].item() + 7))
    # and successive draws should differ (sampled at random, not sequential)
    assert not torch.equal(batch, next(sampler))


@max_score(2)
def test_sequential_batch_sampler():
    tokens = torch.arange(100)
    batches = list(sequential_batch_sampler(tokens, "cpu", batch_size=2, seq_len=5))
    assert all(b.shape == (2, 5) for b in batches)
    # a sequential pass covers the tokens in order, dropping any short final batch
    flat = torch.cat([b.reshape(-1) for b in batches])
    assert torch.equal(flat, torch.arange(len(flat)))
    assert len(flat) <= len(tokens)


@max_score(3)
def test_cosine_lr_schedule():
    warmup, total, min_lr, max_lr = 10, 100, 1e-4, 5e-4
    get_lr = cosine_lr_schedule(warmup, total, min_lr, max_lr)
    assert math.isclose(get_lr(0), 0.0, abs_tol=1e-9), "warmup starts at 0"
    assert math.isclose(get_lr(warmup), max_lr, rel_tol=1e-6), "peaks at end of warmup"
    assert math.isclose(get_lr(total), min_lr, rel_tol=1e-6), "decays to min_lr"
    assert math.isclose(get_lr(total + 50), min_lr, rel_tol=1e-6), "stays at min_lr"
    mid = get_lr((warmup + total) // 2)
    assert math.isclose(mid, (min_lr + max_lr) / 2, rel_tol=5e-2), "half-cosine midpoint"
    # monotonically decreasing through the annealing phase
    lrs = [get_lr(t) for t in range(warmup, total + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))


@max_score(3)
def test_compute_language_modeling_loss():
    input_ids = torch.LongTensor([[0, 1, 2], [2, 1, 0]])
    logits = torch.FloatTensor([[[1.0, -2.0, 3.0] for _ in range(3)] for _ in range(2)])
    assert torch.allclose(
        compute_language_modeling_loss(input_ids, logits),
        torch.FloatTensor([3.1328]),
        rtol=1e-3,
    )


@max_score(2)
def test_language_modeling_loss_is_shifted():
    """The label for position t is the token at t+1. A loss that compares logits
    to the *same* position would be near zero here, which is the classic bug."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 5, (2, 6))
    # logits that perfectly predict the CURRENT token (not the next one)
    logits = torch.full((2, 6, 5), -10.0)
    for b in range(2):
        for t in range(6):
            logits[b, t, input_ids[b, t]] = 10.0
    loss = compute_language_modeling_loss(input_ids, logits)
    assert loss.item() > 1.0, (
        "loss is near zero for a model that copies the current token — "
        "labels are probably not shifted by one position"
    )


@max_score(2)
def test_train_fits_a_learnable_sequence():
    """Fit a trivially learnable dataset; loss must drop."""
    torch.manual_seed(0)
    device = "cpu"
    model = _model(n_vocab=8, max_seq_len=16).to(device)
    tokens = torch.arange(8)

    train_sampler = random_batch_sampler(tokens, device, 4, 4)
    val_sampler = sequential_batch_sampler(tokens, device, 2, 4)
    optimizer = AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.95))
    lr_schedule = cosine_lr_schedule(10, 300, 1e-4, 5e-3)

    model.train()
    train(model, train_sampler, optimizer, lr_schedule, nullcontext(), 300, 2)
    model.eval()
    results = evaluate(model, val_sampler, nullcontext())
    assert results["val-loss"] <= 1.3, f"failed to fit: val-loss={results['val-loss']:.3f}"


@max_score(0)
def test_train_sanity_check():
    """An *unlearnable* sequence: tokens drawn uniformly from {0..7}. No model can
    do better than log(8) ~= 2.08 without peeking at the token it is predicting.

    This test carries no points, but failing it invalidates Q3.4-3.6 — those
    results would be meaningless. (Q3.1 asks you to explain why.)
    """
    torch.manual_seed(0)
    device = "cpu"
    model = _model(n_vocab=8, max_seq_len=4).to(device)
    tokens = torch.randint(0, 8, (4096,))

    train_sampler = random_batch_sampler(tokens, device, 8, 4)
    val_sampler = sequential_batch_sampler(tokens, device, 8, 4)
    optimizer = AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.95))
    lr_schedule = cosine_lr_schedule(10, 300, 1e-4, 5e-4)

    model.train()
    train(model, train_sampler, optimizer, lr_schedule, nullcontext(), 300, 2)
    model.eval()
    results = evaluate(model, val_sampler, nullcontext())
    assert results["val-loss"] >= math.log(8) - 0.1, (
        f"val-loss={results['val-loss']:.3f} is below log(8)={math.log(8):.3f} on an "
        "unlearnable sequence — the model is seeing the token it is predicting"
    )
