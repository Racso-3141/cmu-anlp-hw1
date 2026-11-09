from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Callable
from rich import print

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from .llama import Llama, LlamaConfig, Tokenizer
from .optim import AdamW
from .paths import TINYSTORIES_TOKENS, require
from .utils import (
    count_params,
    determine_device,
    enable_tf32,
    estimate_model_disk_size,
)


def random_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """An infinite generator that samples batches of sequences from the tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        An infinite generator that samples batches of sequences from the
        tokens. Each batch has shape (B x S). Every sequence in the batch is
        a contiguous subsequence of x, sampled uniformly at random. The
        output tensor should be on the right device.

    Pseudocode:
        # loop forever, since this is an infinite generator

        # for each sequence in the batch, pick a random starting offset
        # into tokens (hint: valid offsets must leave room for a full
        # seq_len-length span, so the offset can't start too close to
        # the end)

        # slice out the contiguous span of length seq_len starting at
        # each offset, and stack them into a single (batch_size, seq_len)
        # tensor

        # move the batch to the right device, and yield it
    """

    # todo: yield (B x S) batches of contiguous spans sampled uniformly at random
    raise NotImplementedError


def sequential_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """A generator that yields batches of tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        A generator that yields a batch of tokens at a time. Each batch has
        shape (B x S). Every sequence in the batch is a contiguous subsequence
        of x in sequential order. The output tensor should be on the right
        device.

    Note: If the last batch is incomplete, which could happen when the number
        of tokens is not divisible by (batch_size * seq_len), you could drop
        the last batch.

    Pseudocode:
        # figure out how many tokens one batch consumes (batch_size * seq_len)

        # walk through tokens from the start, taking one chunk of that size
        # at a time

        # reshape each chunk from a flat 1d slice into (batch_size, seq_len),
        # move it to the right device, and yield it

        # stop once there isn't enough left for a full chunk
    """

    # todo: yield (B x S) batches covering the tokens in order
    raise NotImplementedError


def cosine_lr_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
    max_lr: float,
) -> Callable[[int], float]:
    def get_lr(t: int) -> float:
        """Outputs the learning rate at step t under the cosine schedule.

        Args:
            t: the current step number

        Returns:
            lr: learning rate at step t
        """
        # todo: linear warmup, then half-cosine decay to min_lr (see Q3.3)
        raise NotImplementedError

    return get_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def compute_language_modeling_loss(
    input_ids: torch.LongTensor, logits: torch.FloatTensor
) -> torch.FloatTensor:
    """Outputs the language modeling loss given input_ids and logits

    Args:
        input_ids: the input token ids
        logits: the next token logits produced by the language model

    Returns:
        loss: the mean cross entropy loss for next token prediction

    Hint: Think about what are the groundtruth labels for next token prediction.

    Pseudocode:
        # the label for predicting the next token at position i is just
        # the token that actually appears at position i+1, so shift
        # input_ids by one position to get the labels

        # a logit at position i is a prediction for the token at
        # position i+1, so the last position's logit has no matching
        # label (there's nothing after it) — drop it to line logits and
        # labels back up

        # flatten the batch and sequence dimensions together so you have
        # one (logits, label) pair per prediction

        # compute and return the mean cross-entropy loss over all
        # predictions
    """

    # todo: cross-entropy of next-token prediction (mind the shift)
    raise NotImplementedError


def train(
    model: Llama,
    batch_sampler: Iterator[torch.LongTensor],
    optimizer: torch.optim.Optimizer,
    lr_schedule: Callable[[int], float],
    autocast: torch.autocast | nullcontext = nullcontext(),
    num_training_steps: int = 0,
    grad_accumulation_steps: int = 1,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 1000,
    start_step: int = 0,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """A training loop for the language model

    Args:
        model: the decoder LM
        batch_sampler: a generator that produces batches of token ids
        optimizer: an optimizer for gradient update
        lr_schedule: a callable that produces the learning at a step number
        autocast: a context manager that handles tensor casting (you do not need
          to care about this for your implementation)
        num_training_steps: number of steps to train for
        grad_accumulation_steps: number of "micro" training steps before each
          gradient update
        scaler: a GradScaler for fp16 mixed-precision training; pass None to
          disable loss scaling (e.g. for bf16 or CPU training)
    """
    # stores training losses for the 20 latest steps
    losses = deque(maxlen=20 * grad_accumulation_steps)

    for step in (pbar := trange(start_step, num_training_steps, initial=start_step,
                                total=num_training_steps)):
        t0 = time.time()
        lr = lr_schedule(step)
        set_lr(optimizer, lr)
        for _ in range(grad_accumulation_steps):
            input_ids = next(batch_sampler)

            with autocast:
                # targets=input_ids asks for logits at every position; with
                # targets=None the model returns only the final position (an
                # inference-time optimization).
                logits, _ = model(input_ids, targets=input_ids)
            loss = compute_language_modeling_loss(input_ids, logits)
            scaled_loss = loss / grad_accumulation_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            loss_f = loss.item()
            losses.append(loss_f)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        FLOPs_per_step = (
            model.flops_per_token
            * input_ids.shape[0]
            * input_ids.shape[1]
            * grad_accumulation_steps
        )
        loss_mean = np.mean(losses).item()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        pbar.set_postfix(
            {
                "train loss": f"{loss_mean:.2f}",
                "TFLOPS": f"{FLOPs_per_step / dt / 1e12:.1f}",
            }
        )
        wandb.log({"train-loss": loss_mean, "learning-rate": lr}, step=step)

        # Periodic checkpoint. A run at the Q3.6 budget takes hours, and losing
        # all of it to a dropped connection, a reclaimed spot instance or a
        # cluster allocation expiring is a miserable way to spend an afternoon.
        # Written to a temp file and renamed so an interrupted save cannot
        # corrupt the previous checkpoint.
        if checkpoint_dir and checkpoint_every and (step + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            tmp = path + ".tmp"
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "step": step + 1}, tmp)
            os.replace(tmp, path)


@torch.inference_mode()
def evaluate(
    model: Llama,
    batch_sampler: Iterator[torch.LongTensor],
    autocast: torch.autocast | nullcontext = nullcontext(),
) -> dict[str, float]:
    losses = []

    for input_ids in tqdm(batch_sampler, desc="evaluating.."):
        with autocast:
            logits, _ = model(input_ids, targets=input_ids)
        loss = compute_language_modeling_loss(input_ids, logits)
        losses.append(loss.item())

    # mean of the losses is the average negative log likelihood
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)

    eval_results = {
        "val-loss": mean_loss,
        "val-perplexity": perplexity,
    }
    wandb.log(eval_results)
    return eval_results


def main():
    enable_tf32()

    # create an output directory and dump the configuration file
    assert len(sys.argv) > 1, "provide a configuration file"
    config = OmegaConf.load(sys.argv[1])
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    print("#" * 40, OmegaConf.to_yaml(config).strip(), "#" * 40, sep="\n")
    wandb.init(project="anlp-f2026-hw1", config=OmegaConf.to_container(config))

    # initialize tokenizer and model
    tokenizer = Tokenizer()
    device = determine_device() if config.device == "auto" else config.device
    model_config = LlamaConfig(vocab_size=tokenizer.n_words, **config.model_config)
    model = Llama(model_config).to(device)
    print(f"model parameters = {count_params(model) / 1e6:.0f}M")

    model_disk_size_MB = estimate_model_disk_size(model) * 1e-6
    if model_disk_size_MB > 98:
        print(
            f"[red]WARNING: your model is {model_disk_size_MB:.1f}MB. "
            "The largest model size allowed by GradeScope is 100MB, "
            "and you may have trouble with submitting the assignment. "
            "Please update your config so your model is at most 100 MB.[/red]"
        )
    else:
        print(
            f"Your model is {model_disk_size_MB:.1f}MB. This should be within "
            "the 100MB limit of Gradescope."
        )

    # prepare data and data generator
    assert config.seq_len <= config.model_config.max_seq_len
    tokens = np.load(require(TINYSTORIES_TOKENS))

    train_tokens = torch.from_numpy(tokens["train"].astype(int))
    val_tokens = torch.from_numpy(tokens["val"].astype(int))

    train_sampler = random_batch_sampler(
        train_tokens, device, config.batch_size, config.seq_len
    )
    val_sampler = sequential_batch_sampler(
        val_tokens, device, config.batch_size, config.seq_len
    )
    print(f"train dataset tokens = {len(train_tokens) / 1e6:.0f}M")
    FLOPs = (
        model.flops_per_token
        * config.num_training_steps
        * config.grad_accumulation_steps
        * config.batch_size
        * config.seq_len
    )
    print(f"train FLOPs = {FLOPs:.2e}")
    if FLOPs > 1e17:
        print(
            f"[red]WARNING: your train FLOPs is {FLOPs:.2e}. "
            "This is more than the max compute that we allow (1e+17). "
            "Please reduce your model size or train steps.[/red]"
        )

    # Resume from a periodic checkpoint if one is present, so an interrupted run
    # can be restarted with the same command rather than started over.
    start_step = 0
    resume_path = os.path.join(config.output_dir, "checkpoint.pt")

    # prepare optimizer and lr schedule
    # Your AdamW from Part 2, not torch's.
    optimizer = AdamW(
        model.parameters(),
        lr=0.0,  # will set this dynamically in the training loop
        betas=(0.9, 0.95),
    )
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"[green]resuming from {resume_path} at step {start_step}[/green]")

    lr_schedule = cosine_lr_schedule(
        config.num_warmup_steps, config.num_training_steps, config.min_lr, config.max_lr
    )
    # `precision: auto` (the default) decides by GPU generation: native bf16 on
    # compute capability 8.0+ (A100, A10G, L4), fp16 with dynamic loss scaling
    # on older GPUs (T4, V100), fp32 on CPU / Apple Silicon. Do not decide with
    # torch.cuda.is_bf16_supported(): on a T4 it returns True because bf16 is
    # *emulated* there, which measured slower than fp32 (892 vs 571 ms/step;
    # fp16 is 299). Set bf16 / fp16 / fp32 explicitly to override.
    precision = str(config.get("precision", "auto")).lower()
    if precision not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError(f"unknown precision {precision!r}; expected auto, bf16, fp16 or fp32")
    if device != "cuda":
        precision = "fp32"
    elif precision == "auto":
        precision = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    elif precision == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        print("[yellow]bf16 is emulated on this GPU and slower than fp16; "
              "consider precision: auto or fp16[/yellow]")
    autocast = (
        torch.autocast(device, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
        if precision in ("bf16", "fp16")
        else nullcontext()
    )
    if device == "cuda":
        print(f"mixed precision: {precision} on {torch.cuda.get_device_name()}")
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    if os.path.exists(resume_path) and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    # training
    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        autocast,
        config.num_training_steps,
        config.grad_accumulation_steps,
        checkpoint_dir=config.output_dir,
        checkpoint_every=int(config.get("checkpoint_every", 1000)),
        start_step=start_step,
        scaler=scaler,
    )

    # save the trained model
    model_path = os.path.join(config.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"model saved to {model_path}")

    # evaluation
    model.eval()
    eval_results = evaluate(model, val_sampler, autocast)
    print("evaluation results:", json.dumps(eval_results))
    with open(os.path.join(config.output_dir, "eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    print("done!")


if __name__ == "__main__":
    main()
