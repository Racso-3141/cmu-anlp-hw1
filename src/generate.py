"""Part 4 — sampling from your model and measuring perplexity.

Batched generation with left-padding is not available here for a real reason:
left-padding corrupts every real token, because with causal attention a real
token at position i attends to all pads before it. (Right-padding is harmless by
the same argument — pads only ever sit *after* the real tokens.) Rather than add
mask plumbing through the whole model, generation runs one prefix at a time,
which is fine for the handful of prefixes Q4.2 asks for.

Example running at temp 0.1 with 32 new tokens generated after prefix
    python -m src.generate \
        configs/your_config.yaml \
        --prefixes data/prefixes.jsonl \
        --max-new-tokens 32 \
        --temperature 0.1 \
        --out outputs/your_run/gens_t0.1.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .llama import Llama, LlamaConfig, Tokenizer
from .paths import DATA_DIR
from .train import compute_language_modeling_loss
from .utils import determine_device, enable_tf32


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of logits (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """
    # todo: divide the logits by the temperature, then softmax
    raise NotImplementedError


@torch.inference_mode()
def generate(
    model: Llama,
    device: str,
    tokenizer: Tokenizer,
    prefixes: list[str],
    max_new_tokens: int = 32,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> tuple[list[str], float]:
    """Generate a continuation for each prefix and report perplexity.

    Args:
        model: the language model
        device: device to put tensors on
        tokenizer: the SentencePiece tokenizer
        prefixes: strings to condition on
        max_new_tokens: tokens to generate per prefix
        temperature: 0.0 selects greedily; higher values sample more freely
        top_p: nucleus threshold

    Returns:
        (continuations, perplexity of the full generated sequences)
    """
    model.eval()
    generations: list[str] = []
    total_loss, total_tokens = 0.0, 0

    for prefix in tqdm(prefixes, desc="generating"):
        ids = tokenizer.encode(prefix, bos=True, eos=False)
        assert len(ids) + max_new_tokens <= model.params.max_seq_len, (
            f"prefix ({len(ids)} tokens) plus {max_new_tokens} new tokens exceeds "
            f"the model's context window ({model.params.max_seq_len})"
        )
        input_ids = torch.LongTensor([ids]).to(device)

        out = model.generate(input_ids, max_new_tokens,
                             temperature=temperature, top_p=top_p)

        # perplexity of the generated sequence under the model that produced it
        logits, _ = model(out, targets=out)
        loss = compute_language_modeling_loss(out, logits)
        n_tok = out.shape[1] - 1  # next-token predictions available
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

        generations.append(tokenizer.decode(out[0, len(ids):].tolist()))

    ppl = math.exp(total_loss / total_tokens)
    return generations, ppl


def main():
    enable_tf32()
    ap = argparse.ArgumentParser(description="Generate continuations from a trained model.")
    ap.add_argument("config", help="the config the model was trained with")
    ap.add_argument("--checkpoint", default=None,
                    help="model.pt to load (defaults to <output_dir>/model.pt)")
    ap.add_argument("--prefixes", default=str(DATA_DIR / "prefixes.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default=None, help="where to write generations (jsonl)")
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    device = determine_device() if config.device == "auto" else config.device
    tokenizer = Tokenizer()

    model = Llama(LlamaConfig(vocab_size=tokenizer.n_words, **config.model_config)).to(device)
    ckpt = args.checkpoint or os.path.join(config.output_dir, "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))

    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f if line.strip()]

    generations, ppl = generate(model, device, tokenizer, prefixes,
                                args.max_new_tokens, args.temperature, args.top_p)
    print(f"perplexity: {ppl:.3f}")
    for prefix, gen in zip(prefixes, generations):
        print(f"\n--- {prefix!r}\n{gen}")

    out_path = args.out or os.path.join(config.output_dir, "generations.jsonl")
    with open(out_path, "w") as f:
        for prefix, gen in zip(prefixes, generations):
            f.write(json.dumps({"prefix": prefix, "generation": gen,
                                "temperature": args.temperature}) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
