"""Part 4 — perplexity of documents under your model.

Every comparison in Part 4 is between your Part 3 model and `stories42M`,
both of them Llamas built from your own code.

    python -m src.evaluate_perplexity --checkpoint data/stories42M.pt \
        --documents data/ppl_examples.jsonl
    python -m src.evaluate_perplexity --config configs/llama-base.yaml \
        --checkpoint outputs/llama-base/model.pt --documents ...
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .llama import Llama, LlamaConfig, Tokenizer, load_pretrained
from .paths import DATA_DIR
from .train import compute_language_modeling_loss
from .utils import determine_device, enable_tf32


@torch.inference_mode()
def compute_perplexity(
    model: Llama,
    device: str,
    tokenizer: Tokenizer,
    documents: list[str],
) -> tuple[float, float]:
    """Perplexity and bits-per-byte of a set of documents.

    Documents are scored one at a time and truncated to the model's context
    window. Perplexity is exp of the mean per-token loss.

    Bits-per-byte normalizes by the UTF-8 length of the text instead of by the
    token count, which makes it comparable across tokenizers — perplexity is
    not.
    """
    model.eval()
    total_nats, total_tokens, total_bytes = 0.0, 0, 0

    for doc in tqdm(documents, desc="scoring"):
        ids = tokenizer.encode(doc, bos=True, eos=False)[: model.params.max_seq_len]
        if len(ids) < 2:
            continue
        input_ids = torch.LongTensor([ids]).to(device)
        logits, _ = model(input_ids, targets=input_ids)
        loss = compute_language_modeling_loss(input_ids, logits)
        n_pred = len(ids) - 1
        total_nats += loss.item() * n_pred
        total_tokens += n_pred
        total_bytes += len(doc.encode("utf-8"))

    mean_loss = total_nats / total_tokens
    perplexity = math.exp(mean_loss)
    bits_per_byte = total_nats / (total_bytes * math.log(2))
    return perplexity, bits_per_byte


def load_model(args, tokenizer, device):
    """Load either a stories42M-style checkpoint or one of your own training runs.

    They are stored differently: `stories42M.pt` carries its own `model_args`,
    while `src/train.py` writes a bare state dict whose shape is described by the
    config you trained with.
    """
    if args.config is None:
        return load_pretrained(args.checkpoint).to(device)
    config = OmegaConf.load(args.config)
    model = Llama(LlamaConfig(vocab_size=tokenizer.n_words, **config.model_config))
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    return model.to(device)


def main():
    enable_tf32()
    ap = argparse.ArgumentParser(description="Perplexity of documents under a Llama.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="required for your own models; omit for stories42M.pt")
    ap.add_argument("--documents", default=str(DATA_DIR / "ppl_examples.jsonl"),
                    help="jsonl with a 'text' field per line")
    ap.add_argument("--strip-word", default=None,
                    help="delete every occurrence of this word first (Q4.3.A)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = determine_device() if args.device == "auto" else args.device
    tokenizer = Tokenizer()
    model = load_model(args, tokenizer, device)

    with open(args.documents) as f:
        docs = [json.loads(line) for line in f if line.strip()]

    for doc in docs:
        text = doc["text"]
        if args.strip_word:
            # keep spacing sane after deletion so we measure the word's absence,
            # not a sudden pile-up of double spaces
            text = " ".join(w for w in text.split() if w.lower() != args.strip_word.lower())
        ppl, bpb = compute_perplexity(model, device, tokenizer, [text])
        label = doc.get("name", "document")
        print(f"{label:24s} perplexity {ppl:10.2f}   bits/byte {bpb:6.3f}")


if __name__ == "__main__":
    main()
