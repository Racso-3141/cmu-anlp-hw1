# 11-711 Advanced NLP (Fall 2026) — Homework 1

Build a Llama-2 language model end to end: tokenize text, implement the
architecture, pretrain it under a compute budget, evaluate what it learned, and
adapt it to a downstream task.

**Released** Sep 1 · **Due** Sep 24, 11:59 PM · **Individual** · 10% of your grade


**The assignment itself is `spec/homework1.pdf`.** This README covers setup and
mechanics; the questions are in the PDF. Please read this entire README before starting the HW, as it has valuable information about setup and running commands (on your machine or on Google Colab).

Write your report from
`spec/report-template.tex`.

## Setup

You will need a [Weights & Biases](https://wandb.ai) account for Part 3
(`wandb login`).

### On your own machine 

```bash
bash setup.sh              # conda env, dependencies, data downloads
conda activate anlp-hw1
pytest tests/ -q           # most tests fail initially — that is the point
```

### On Google Colab

1. Unzip the starter and upload the folder to your Google Drive, so your code and
   outputs persist across sessions even when your runtime disconnects.
2. Upload `setup.ipynb` to Google Colab so that you can run the notebook cells.
3. **Runtime -> Change runtime type -> select a runtime** (if you're running
   training, we suggest a GPU, e.g. T4 or better). 
4. Run `setup.ipynb` — in Section 0, remember to edit `REPO_DIR` to wherever you put the starter folder. The remaining cells install dependencies, fetch the data, and log in to W&B.
5. For future scripts that you run, you can add cells to this notebook and use python commands as described below.

To edit code in Colab, open the `.py` files from Colab's file browser and edit them directly.

> `train.py` picks the precision for your GPU automatically (`precision: auto` in the configs):
> bf16 on Ampere or newer GPUs, fp16 with loss scaling on a T4.

## The five parts

| Part | What you build | Points |
|---|---|---:|
| 1. Tokenization | BPE tokenizer: `merge`, `encode`, `decode` | 35 |
| 2. The model | RMSNorm, RoPE, grouped-query attention, the transformer block | 37 |
| 3. Pretraining | AdamW, batch samplers, cosine LR schedule, LM loss — then a FLOP-budgeted search and a final model | 68 |
| 4. Inference | Temperature and top-p sampling, perplexity, comparison against the provided `stories42M` | 35 |
| 5. Adaptation | A classification head on the provided `stories42M` checkpoint; prompting vs. finetuning — then the same on **your own** Part 3 model, and compare | 25 |

Roughly half of the 200 points are **written analysis** submitted as
`report.pdf`. Per the syllabus you may use AI to help write code, but **not** to
write any part of your report. The report includes a required "Use of AI"
question — answer it honestly.

## Layout

```
configs/          training configs (llama-tiny for smoke tests, llama-base to start from)
data/             datasets, tokenizer model, pretrained checkpoint (fetched by setup.sh)
src/
  tokenizer/      Part 1 — BPE
  llama/          Part 2 — model, RoPE, config, SentencePiece tokenizer
  optim/          Part 3 — AdamW
  train.py        Part 3 — training loop and FLOP accounting
  generate.py     Part 4 — sampling
  compare_generations.py    Part 4 — your model vs stories42M, side by side
  evaluate_perplexity.py    Part 4 — perplexity
  adapt.py        Part 5 — classification head and zero-shot prompting
tests/            visible tests — the autograder runs these plus hidden ones
bonus/            optional: 4-digit addition, LoRA
```

## Running things

```bash
# Part 3: smoke test, then your own configs
python -m src.train configs/llama-tiny.yaml
python -m src.train configs/llama-base.yaml

# Part 1 (Q1.4): digit splits and tokens-per-character across domains
python -m src.tokenizer.inspect_tokenizer

# Part 4
python -m src.generate <your-config.yaml> --checkpoint outputs/<your-run>/model.pt
python -m src.compare_generations --config <your-config.yaml> --checkpoint outputs/<your-run>/model.pt
python -m src.evaluate_perplexity --checkpoint outputs/<your-run>/model.pt --config <your-config.yaml>
# Q4.3.A: the same, with --strip-word the

# Part 5 — Q5.2's four runs (two options x two datasets; add --use-gpu on a
# GPU machine). The dataset flags default to SST-5; pass the cfimdb files
# explicitly for the CFIMDB runs. Every run writes its dev and test prediction
# files into outputs/preds/ automatically, and zip-submission.sh ships
# everything in that folder — those files are how we verify your reported
# numbers and score the blind CFIMDB test split.
python -m src.adapt --option prompt   --pretrained-model-path data/stories42M.pt
python -m src.adapt --option finetune --pretrained-model-path data/stories42M.pt
python -m src.adapt --option prompt   --pretrained-model-path data/stories42M.pt \
    --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt \
    --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json
python -m src.adapt --option finetune --pretrained-model-path data/stories42M.pt \
    --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt \
    --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json

# Part 5.3 — the same finetune against the model YOU pretrained in Part 3
# (SST-5 only). Its prediction files get an -own suffix so they do not
# overwrite the Q5.2 run's.
python -m src.adapt --option finetune --pretrained-model-path outputs/<your-run>/model.pt --backbone-config <your-config.yaml>
```

### Running the parts on Colab

Same commands, run from a notebook cell (after `setup.ipynb`
has mounted Drive and `%cd`'d into place). Prefix each with `!`, or drop the
`!` and use `%run` if you want the process's variables left behind in the
notebook's Python session for inspection:

```
# if you have not already moved to the REPO_DIR
REPO_DIR = "/content/drive/MyDrive/<path-to-hw1>"  
%cd {REPO_DIR}

# run commands as normal in notebook cell
!python -m src.train configs/llama-tiny.yaml
!python -m src.generate <your-config.yaml>
``` 

**Useful Notes** 
- Training checkpoints save every 1000 steps by default (`checkpoint_every` in the
config). If a run is interrupted — dropped server connection, Colab disconnects — just
re-run the same command and it picks up where it stopped rather than starting
over.
- When running Python scripts in Colab, we suggest running in a notebook even if you have access to a terminal with Google Colab Pro. This is because Colab does not recognize notebooks with only a terminal command running as active, which leads to more frequent disconnects.

## Where things get written

Everything a run produces goes under `outputs/`, which is gitignored:

| Path | What |
|---|---|
| `outputs/<config-name>/model.pt` | your trained model (Q3.6 submits this) |
| `outputs/<config-name>/checkpoint.pt` | periodic checkpoint; training resumes from it automatically |
| `outputs/<config-name>/eval.json` | final validation loss and perplexity |
| `outputs/adapt/` | Part 5 classifier checkpoints |
| `outputs/preds/` | Part 5 prediction files — shipped in your code submission; staff score the test split against real held-out labels |

## Compute budget

Part 3 is budgeted in FLOPs, not wall-clock, so the assignment is fair across
hardware. `train.py` prints the projected cost before training and warns if you
exceed the final 1e17 budget; for Q3.5, check the printed projection against 1e15 yourself.

| | Budget | Roughly |
|---|---|---|
| Q3.5, each search config | ≤ 1e15 FLOPs | ~4 min on a Google Colab T4 instance |
| Q3.6, your final model | ≤ 1e17 FLOPs | ~6 h on a Google Colab T4 instance |

Total cost is `flops_per_token × num_training_steps × grad_accumulation_steps ×
batch_size × seq_len`, where `flops_per_token = 6N + 12·L·d·ctx` (`ctx` = `max_seq_len`). Every
hyperparameter moves it — a bigger model buys fewer steps.

## Submission

Due **September 24, 11:59 PM**. Two Gradescope submissions, nothing submitted twice:

1. **Assignment1: Code** — run `MODEL_DIR=outputs/<your-run> bash scripts/zip-submission.sh`
   and upload the `submission.zip` it writes. It contains `src/`, your
   `tests/test_tokenizer.py`, your Part 5 prediction files from `outputs/preds/`,
   and your Q3.6 `model.pt` + `config.yaml` when `MODEL_DIR` is set. Gradescope
   rejects zips over 100MB. You can resubmit as often as you like; only the
   latest submission counts.

   A complete Part 5 leaves ten prediction files in `outputs/preds/`, all
   written automatically by your runs:
   `{sst,cfimdb}-{dev,test}-{prompt,finetune}-output.txt` from Q5.2's four
   runs, plus `sst-{dev,test}-finetune-own-output.txt` from Q5.3. They ride
   along in the zip; we use them to verify the numbers in your report and to
   score the blind CFIMDB test split. Don't rename or move them.
2. **Assignment1: Report** — upload `report.pdf`, built from
   `spec/report-template.tex`. Gradescope asks you to mark which pages answer
   each question; the question numbers match the template. The entries marked
   *code only* need no pages.

The autograder runs the tests in `tests/`, plus hidden tests you cannot see;
one of them measures your final model's perplexity on a held-out split. Your
score appears a few minutes after each upload.

Late submissions cost 5% of the homework grade per day (see the syllabus late
policy) and close on **October 14, 11:59 PM**.

## Grading notes

`test_train_sanity_check` trains on an *unlearnable* sequence — uniform random
tokens — where no model can beat a loss of `log(8) ≈ 2.08` without seeing the
token it is predicting. It is worth zero points, but failing it means results for Q3.4–3.6 would be
meaningless. Fix it first — Q3.1 asks you to explain why.

## Acknowledgements

This code is based on llama2.c by Andrej Karpathy (MIT, see `LICENSE`). Parts of
the code are also from the [`transformers`](https://github.com/huggingface/transformers)
library (Apache License 2.0), and it contains adaptations from
[nanoGPT](https://github.com/karpathy/nanoGPT) and [PyTorch](https://pytorch.org/)
(see `copyright/`).
# cmu-anlp-hw1
