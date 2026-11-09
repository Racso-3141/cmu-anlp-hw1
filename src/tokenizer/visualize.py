"""Inspect the tokens a BPE tokenizer learned (Question 1.3).

    python -m src.tokenizer.visualize
    python -m src.tokenizer.visualize --longest 20

Previously this ran at import time and resolved its data file relative to the
current working directory, so importing it printed the whole vocabulary and
running it from anywhere but the repository root failed.
"""
from __future__ import annotations

import argparse
import unicodedata

from ..paths import DATA_DIR, require
from .bpe import UnicodeBPETokenizer


def visualize_bytes(b: bytes) -> str:
    return "".join(
        ch
        for ch in b.decode("utf-8", errors="replace")
        if unicodedata.category(ch)[0] != "C"
    )


def main():
    ap = argparse.ArgumentParser(description="Show the tokens a BPE tokenizer learned.")
    ap.add_argument("--tokenizer", default=str(DATA_DIR / "english-tokenizer.json"))
    ap.add_argument("--longest", type=int, default=0,
                    help="also list the N longest tokens (useful for Q1.3.A)")
    args = ap.parse_args()

    tokenizer = UnicodeBPETokenizer.from_config(str(require(
        __import__("pathlib").Path(args.tokenizer))))

    learned = tokenizer.vocab[256:]   # everything past the raw byte alphabet
    for b in learned:
        print(f"bytes: {b} <-> token: <{visualize_bytes(b)}>")

    if args.longest:
        print(f"\n--- {args.longest} longest learned tokens ---")
        for b in sorted(learned, key=len, reverse=True)[: args.longest]:
            print(f"{len(b):>3} bytes  <{visualize_bytes(b)}>")


if __name__ == "__main__":
    main()
