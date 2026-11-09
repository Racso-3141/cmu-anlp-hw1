"""Print the measurements Question 1.4 asks about.

    python -m src.tokenizer.inspect_tokenizer

Q1.4 is a writing question. You should not have to write a script to answer it,
so this prints everything you need: how the Llama tokenizer splits numbers, and
how many tokens it spends per character across three kinds of text.

Add your own text with --text "..." if you want to check something else.
"""
from __future__ import annotations

import argparse

from ..llama import Tokenizer

SAMPLES = {
    "English prose": (
        "The quick brown fox jumps over the lazy dog near the river bank at "
        "dawn every single morning without fail."
    ),
    "Python code": (
        "def f(x, y=3):\n    return [i**2 for i in range(x) if i % y == 0]\n"
    ),
    "Hindi (Devanagari)": (
        "यह एक परीक्षण वाक्य है जो हिंदी भाषा में लिखा गया है और इसमें कुछ सामान्य शब्द हैं।"
    ),
}


def show_digits(tok):
    print("=" * 68)
    print("Q1.4.A  How does the tokenizer split numbers?")
    print("=" * 68)
    print(f"{'number':>8}  {'tokens':>6}  pieces")
    for n in ["1", "7", "12", "19", "42", "100", "999", "1000", "1005",
              "1234", "1235", "9999"]:
        ids = tok.encode(n, bos=False, eos=False)
        pieces = [tok.sp_model.id_to_piece(i) for i in ids]
        print(f"{n:>8}  {len(ids):>6}  {pieces}")
    print("\nLook at numbers with the SAME number of digits. Do they always cost")
    print("the same number of tokens? Would that be true of every tokenizer?\n")


def show_domains(tok, extra=None):
    print("=" * 68)
    print("Q1.4.B  How many tokens per character, by kind of text?")
    print("=" * 68)
    samples = dict(SAMPLES)
    if extra:
        samples["your text"] = extra
    print(f"{'text':>22}  {'chars':>6} {'tokens':>7} {'tok/char':>9} {'bytes/tok':>10}")
    for name, text in samples.items():
        ids = tok.encode(text, bos=False, eos=False)
        print(f"{name:>22}  {len(text):>6} {len(ids):>7} "
              f"{len(ids)/len(text):>9.3f} {len(text.encode())/len(ids):>10.2f}")
    print("\nWhich is most expensive per character, and why might that be?")
    print("Remember that API pricing and context windows are counted in tokens.\n")


def main():
    ap = argparse.ArgumentParser(description="Measurements for Question 1.4.")
    ap.add_argument("--text", default=None, help="also measure your own text")
    args = ap.parse_args()

    tok = Tokenizer()
    show_digits(tok)
    show_domains(tok, args.text)


if __name__ == "__main__":
    main()
