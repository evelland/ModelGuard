# Larger baseline: BANKING77 (real customer-support requests)

The first 27-example experiment was only a pipeline smoke test. This experiment replaces the toy target task with **BANKING77**, a public academic dataset of 10,003 training and 3,080 test queries covering **77 banking-support intents** (Casanueva et al., 2020; dataset licensed CC BY 4.0).

**No model results have been generated for this extension.** Source datasets are downloaded only on your machine, and local model-generated results are not committed to Git.

## 0. Update branch and dependencies

From repository root, with the virtual environment activated:

    git checkout prototype/local-qwen25-lora
    git pull
    python -m unittest discover -s prototype/tests -v

No extra Python libraries are required for dataset preparation.

## 1. Download the original public data

    mkdir -p artifacts/datasets/banking77
    curl -fL https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv -o artifacts/datasets/banking77/train.csv
    curl -fL https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv -o artifacts/datasets/banking77/test.csv

The script uses the ORIGINAL train/test split, removes exact normalized text overlap, and samples categories evenly with a recorded RNG seed and source SHA256.

## 2. Prepare about 100x more training data

    python prototype/prepare_banking77.py --train-size 2700 --eval-size 400 --protected-size 300 --seed 42

This writes under artifacts/experiments/banking77 (ignored by Git):
- train.jsonl — 2700 real training requests, 77 possible target labels
- target_eval.jsonl — 400 real held-out test requests
- protected_eval.jsonl — 300 independently generated arithmetic/formatting/instruction/JSON cases
- combined_eval.jsonl — 700 prompts for paired base/candidate inference
- manifest.json — source checksums, label distribution, seed, sizes

Warning: the original task was 3 broad synthetic ticket types; BANKING77 is a NEW, harder 77-intent task, so scores are not comparable across the two target tasks. Protected metrics are also not comparable unless you re-evaluate the old model on the same expanded suite.

## 3. Train a fresh LoRA adapter

Use a new output directory. Do not overwrite the original 27-example adapter.

    python prototype/train.py --device mps --train-data artifacts/experiments/banking77/train.jsonl --output artifacts/banking77-2700-lora --max-length 256 --epochs 1

Start with 1 epoch; for a controlled sweep repeat runs from the original Qwen checkpoint with 2 or 3 epochs using distinct output directories. Training and inference can be substantially slower than for 27 examples.

## 4. Evaluate both on IDENTICAL prompts

    python prototype/evaluate_pair.py --device mps --adapter artifacts/banking77-2700-lora --cases artifacts/experiments/banking77/combined_eval.jsonl --output artifacts/experiments/banking77/predictions.jsonl --max-new-tokens 32

This runs 700 prompts for BOTH the base and candidate models. Runtime may be appreciable. For a shorter initial checkpoint, generate smaller eval sets using --eval-size 100 --protected-size 100 and an alternate --output directory; do not use repeated test peeking to tune hyperparameters.

## 5. Build a report

    python prototype/report.py --input artifacts/experiments/banking77/predictions.jsonl --html artifacts/experiments/banking77/report.html --json artifacts/experiments/banking77/report.json --max-protected-drop 0.02
    open artifacts/experiments/banking77/report.html

2 percentage points is a suggested more conservative **example** threshold, not an empirically validated release policy. The report counts new regressions only where the base answer was correct. Inspect category-level baselines—categories with poor base performance cannot establish forgetting.

## Research validity

- Increase dataset diversity, not just number of near-duplicates. BANKING77 contains human-authored banking intents and official held-out splits.
- Do NOT infer forgetting from aggregate protected accuracy if the base model has poor initial scores.
- The generated arithmetic/formatting/JSON suite is synthetic: it improves *coverage*, but is not a replacement for established benchmarks.
- Record dataset hashes, LoRA config, model revision, learning rates, seed, and hardware timings. Repeat at least three seeds for final claims.
- Consider a second experiment using the EXACT SAME protected evaluation suite for 27/2700 example adapters and identical LoRA settings, and compare tradeoff vs training steps.
- This is still behavioral change testing, not SAE feature analysis or SAE-guided preservation.
