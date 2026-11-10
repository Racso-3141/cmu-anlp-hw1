from typing import Callable, Iterable, Tuple

import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        # todo: AdamW update with efficient bias correction and decoupled weight decay
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            correct_bias = group["correct_bias"]
            max_grad_norm = group["max_grad_norm"]

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(group["params"], max_grad_norm)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)      # m
                    state["exp_avg_sq"] = torch.zeros_like(p.data)   # v

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # step 3
                m = beta1 * m + (1 - beta1) * grad
                state["exp_avg"] = m

                # step 4
                v = beta2 * v + (1 - beta2) * grad * grad
                state["exp_avg_sq"] = v

                # step 5
                if correct_bias:
                    alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                else:
                    alpha_t = lr

                # step 6
                p.data = p.data - alpha_t * m / (v.sqrt() + eps)

                # step 7
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data