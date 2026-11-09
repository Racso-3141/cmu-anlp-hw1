#!/usr/bin/env bash
# Build submission.zip for the Gradescope CODE submission.
# Your report.pdf goes to the separate written submission — not in this zip.
#
#   bash scripts/zip-submission.sh                      # code only
#   MODEL_DIR=outputs/my-run bash scripts/zip-submission.sh   # code + final model
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf submission submission.zip
mkdir submission

for required in src requirements.txt setup.py tests/test_tokenizer.py; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: $required is missing from the repository." >&2
    exit 1
  fi
done

cp -r src requirements.txt setup.py submission/
# Q1.1's counterexamples live in your tests/test_tokenizer.py --
# included so graders can run it.
mkdir -p submission/tests
cp tests/test_tokenizer.py submission/tests/
# Part 5 prediction files. Your copy of cfimdb-test.txt carries placeholder
# labels; staff score the submitted test predictions against the real
# held-out labels.
if compgen -G "outputs/preds/*.txt" > /dev/null; then
  mkdir -p submission/predictions
  cp outputs/preds/*.txt submission/predictions/
  echo "included $(ls submission/predictions | wc -l | tr -d ' ') Part 5 prediction files"
else
  echo "NOTE: no outputs/preds/*.txt found -- Part 5 predictions not included."
  echo "      If you have done Part 5, re-run src.adapt before zipping."
fi

# Q3.6 asks you to submit your final model. Skipped if MODEL_DIR is unset, so
# the script still works for the earlier parts.
if [[ -n "${MODEL_DIR:-}" ]]; then
  missing=0
  for f in model.pt config.yaml; do
    if [[ -f "$MODEL_DIR/$f" ]]; then
      cp "$MODEL_DIR/$f" submission/
    else
      echo "WARNING: $MODEL_DIR/$f not found" >&2
      missing=1
    fi
  done
  if [[ $missing -eq 1 ]]; then
    echo "WARNING: your model is incomplete, so Q3.6's hidden test cannot score it." >&2
  fi
else
  echo "NOTE: MODEL_DIR not set, so no model is included."
  echo "      For Q3.6, re-run as:  MODEL_DIR=outputs/<your-run> bash scripts/zip-submission.sh"
fi

( cd submission && zip -qr ../submission.zip . -x '*/__pycache__/*' '*.pyc' '*/.DS_Store' )

size_bytes=$(wc -c < submission.zip)
size_mb=$(( size_bytes / 1000000 ))
echo "wrote submission.zip (${size_mb}MB)"
if (( size_bytes > 100000000 )); then
  echo "ERROR: Gradescope rejects uploads over 100MB. Shrink your model." >&2
  exit 1
fi
