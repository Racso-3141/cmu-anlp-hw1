"""Generate from two models side by side, at several temperatures (Question 4.2).

    python -m src.compare_generations --config configs/llama-base.yaml \
        --checkpoint outputs/llama-base/model.pt

Q4.2 is a writing question about what you observe, so this does the mechanical
part: it loads your Part 3 model and the provided stories42M, generates from the
same prefixes at the same temperatures, reports perplexity for each, and writes
everything to a jsonl you can paste from.

Omit --checkpoint to see stories42M alone.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from .generate import generate
from .llama import Llama, LlamaConfig, Tokenizer, load_pretrained
from .paths import DATA_DIR, PRETRAINED_LLAMA, require
from .utils import determine_device, enable_tf32


def load_student_model(config_path, checkpoint, tokenizer, device):
    cfg = OmegaConf.load(config_path)
    model = Llama(LlamaConfig(vocab_size=tokenizer.n_words, **cfg.model_config))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    return model.to(device).eval()


def main():
    enable_tf32()
    ap = argparse.ArgumentParser(description="Side-by-side generations for Q4.2.")
    ap.add_argument("--config", default=None, help="config your Part 3 model was trained with")
    ap.add_argument("--checkpoint", default=None, help="your Part 3 model.pt")
    ap.add_argument("--prefixes", default=str(DATA_DIR / "prefixes.jsonl"))
    ap.add_argument("--temperatures", default="0.0,0.7,1.5",
                    help="comma-separated, e.g. 0.0,0.7,1.5")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="outputs/generations-comparison.jsonl")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = determine_device() if args.device == "auto" else args.device
    tok = Tokenizer()
    temps = [float(t) for t in args.temperatures.split(",")]

    models = {"stories42M": load_pretrained(str(require(PRETRAINED_LLAMA))).to(device).eval()}
    if args.checkpoint:
        if not args.config:
            ap.error("--checkpoint needs --config so the architecture is known")
        models["your model"] = load_student_model(args.config, args.checkpoint, tok, device)
    else:
        print("(no --checkpoint given, showing stories42M only)\n")

    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f if line.strip()]

    rows = []
    for temp in temps:
        print("=" * 72)
        print(f"temperature {temp}")
        print("=" * 72)
        for name, model in models.items():
            torch.manual_seed(1337)          # same seed, so the comparison is fair
            gens, ppl = generate(model, device, tok, prefixes,
                                 max_new_tokens=args.max_new_tokens, temperature=temp)
            print(f"\n--- {name}   (perplexity {ppl:.2f})")
            for prefix, gen in zip(prefixes, gens):
                print(f"  {prefix}")
                print(f"    -> {gen.strip()}")
                rows.append({"model": name, "temperature": temp,
                             "prefix": prefix, "generation": gen, "perplexity": ppl})
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out} ({len(rows)} generations)")


if __name__ == "__main__":
    main()
