"""Part 2 — AdamW with decoupled weight decay and gradient clipping."""
import numpy as np
import torch

from src.optim import AdamW
from src.pytest_utils.decorators import max_score

from .conftest import TEST_DATA

SEED = 0


def _train_linear(opt_class, **opt_kwargs):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(model.parameters(), **opt_kwargs)
    for _ in range(1000):
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


@max_score(8)
def test_adamw_matches_reference():
    ref = torch.tensor(np.load(TEST_DATA / "optimizer_test.npy"))
    got = _train_linear(AdamW, lr=1e-3, weight_decay=1e-4,
                        correct_bias=True, max_grad_norm=0.9)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-3), \
        f"max abs diff {(ref - got).abs().max().item():.3e}"


@max_score(2)
def test_bias_correction_changes_the_first_step():
    """With correct_bias=True the first step is scaled by
    sqrt(1-b2)/(1-b1); with it off the step is just lr."""
    def one_step(correct_bias):
        torch.manual_seed(SEED)
        p = torch.nn.Parameter(torch.zeros(4))
        opt = AdamW([p], lr=0.1, betas=(0.9, 0.999), weight_decay=0.0,
                    correct_bias=correct_bias)
        p.grad = torch.ones(4)
        opt.step()
        return p.detach().clone()

    corrected, uncorrected = one_step(True), one_step(False)
    assert not torch.allclose(corrected, uncorrected), \
        "correct_bias must change the update"
    # first step with bias correction moves ~lr regardless of beta values
    assert torch.allclose(corrected, torch.full((4,), -0.1), atol=1e-4)


@max_score(2)
def test_weight_decay_is_decoupled():
    """Decay must be applied directly to the parameter, not folded into the
    gradient — and it must be scaled by the learning rate."""
    def final_param(weight_decay):
        torch.manual_seed(SEED)
        p = torch.nn.Parameter(torch.ones(4))
        opt = AdamW([p], lr=0.1, weight_decay=weight_decay, correct_bias=True)
        for _ in range(5):
            p.grad = torch.zeros(4)  # no gradient signal at all
            opt.step()
        return p.detach().clone()

    assert torch.allclose(final_param(0.0), torch.ones(4), atol=1e-6), \
        "with zero gradient and no decay the parameter must not move"
    decayed = final_param(0.1)
    assert (decayed < 1.0).all(), "weight decay must shrink parameters on its own"


@max_score(2)
def test_gradient_clipping_bounds_the_update():
    """A huge gradient must be clipped to max_grad_norm before the update."""
    def step_with(max_grad_norm):
        torch.manual_seed(SEED)
        p = torch.nn.Parameter(torch.zeros(4))
        opt = AdamW([p], lr=0.1, weight_decay=0.0, correct_bias=True,
                    max_grad_norm=max_grad_norm)
        p.grad = torch.full((4,), 1e6)
        opt.step()
        return p.grad.norm().item()

    assert step_with(0.9) <= 0.9 + 1e-4, "gradients were not clipped"
    assert step_with(None) > 1e5, "clipping should be off when max_grad_norm is None"
