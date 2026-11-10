"""Break down the training FLOPs of a config, term by term.

Usage: python scripts/flops.py configs/llama-base.yaml [budget]
       budget defaults to 1e15 (Q3.5); use 1e17 for Q3.6.

FLOPs = flops_per_token * num_training_steps * grad_accumulation_steps
        * batch_size * seq_len
flops_per_token = 6 * N + 12 * n_layers * dim * max_seq_len
N = 32000 * dim
    + n_layers * (2 * dim * dim + 2 * dim * n_kv_heads * (dim / n_heads)
                  + 3 * dim * hidden_dim + 2 * dim)
    + dim
hidden_dim = multiple_of * ceil(floor(8 * dim / 3) / multiple_of)
"""

import math
import sys

import yaml

VOCAB = 32000


def pct(part, total):
    return f"{100 * part / total:6.2f}%"


def show(title, rows, total, total_name):
    """rows: list of (term, value, meaning)."""
    print(title)
    for term, value, meaning in rows:
        print(f"  {term:<54}{value:>14,}  {pct(value, total)}")
        print(f"      {meaning}")
    print(f"  {total_name:<54}{total:>14,}\n")


def main(path, budget):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    m = cfg["model_config"]
    dim = m["dim"]
    n_layers = m["n_layers"]
    n_heads = m["n_heads"]
    n_kv_heads = m.get("n_kv_heads") or n_heads
    max_seq_len = m["max_seq_len"]
    multiple_of = m["multiple_of"]
    batch_size = cfg["batch_size"]
    seq_len = cfg["seq_len"]
    num_training_steps = cfg["num_training_steps"]
    grad_accumulation_steps = cfg.get("grad_accumulation_steps", 1)

    hidden_dim = multiple_of * math.ceil((8 * dim // 3) / multiple_of)
    head_dim = dim // n_heads

    print(f"config: {path}")
    print(f"  dim={dim}  n_layers={n_layers}  n_heads={n_heads}  "
          f"n_kv_heads={n_kv_heads}  max_seq_len={max_seq_len}  multiple_of={multiple_of}")
    print(f"  batch_size={batch_size}  seq_len={seq_len}  "
          f"grad_accumulation_steps={grad_accumulation_steps}  "
          f"num_training_steps={num_training_steps}")
    print(f"  hidden_dim = multiple_of * ceil(floor(8 * dim / 3) / multiple_of) = {hidden_dim}")
    print(f"  head_dim   = dim / n_heads = {head_dim}\n")

    # --- N (number of parameters) ---
    n_rows = [
        ("32000 * dim", VOCAB * dim,
         "Token embedding (32000 x dim), tied with output layer, counted once."),
        ("n_layers * 2 * dim * dim", n_layers * 2 * dim * dim,
         "Attention query + output projections (dim -> dim each)."),
        ("n_layers * 2 * dim * n_kv_heads * (dim/n_heads)",
         n_layers * 2 * dim * n_kv_heads * head_dim,
         "Attention key + value projections (dim -> n_kv_heads * head_dim, GQA)."),
        ("n_layers * 3 * dim * hidden_dim", n_layers * 3 * dim * hidden_dim,
         "SwiGLU feed-forward: w1, w3 (dim -> hidden_dim) and w2 (hidden_dim -> dim)."),
        ("n_layers * 2 * dim", n_layers * 2 * dim,
         "RMSNorm scales, two per layer (before attention, before FFN)."),
        ("dim", dim,
         "Final RMSNorm scale before the output layer."),
    ]
    N = sum(v for _, v, _ in n_rows)
    show("--- N (number of parameters) ---", n_rows, N, "N")

    # --- flops_per_token ---
    fpt_rows = [
        (f"6 * [{term}]", 6 * v, "Forward 2x + backward 4x of: " + meaning)
        for term, v, meaning in n_rows
    ]
    fpt_rows.append((
        "12 * n_layers * dim * max_seq_len", 12 * n_layers * dim * max_seq_len,
        "Parameter-free attention matmuls (Q * K^T, softmax * V), fwd + bwd.",
    ))
    flops_per_token = sum(v for _, v, _ in fpt_rows)
    show("--- flops_per_token = 6 * N + 12 * n_layers * dim * max_seq_len ---",
         fpt_rows, flops_per_token, "flops_per_token")

    # --- FLOPs ---
    tokens = num_training_steps * grad_accumulation_steps * batch_size * seq_len
    flops = flops_per_token * tokens
    max_steps = int(budget // (flops_per_token * grad_accumulation_steps * batch_size * seq_len))

    print("--- FLOPs = flops_per_token * num_training_steps * "
          "grad_accumulation_steps * batch_size * seq_len ---")
    print(f"  tokens = num_training_steps * grad_accumulation_steps * batch_size * seq_len"
          f" = {tokens:,}")
    print(f"      Total tokens processed in training ({tokens / N:.2f} tokens per parameter).")
    print(f"  FLOPs  = {flops:.3e}  ({pct(flops, budget).strip()} of {budget:.0e} budget)"
          f"  {'OK' if flops <= budget else 'OVER BUDGET'}")
    print(f"  max num_training_steps within budget = {max_steps}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/llama-base.yaml"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1e15
    main(cfg_path, budget)
