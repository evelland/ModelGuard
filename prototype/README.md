# ModelGuard — Local Qwen Prototype

This directory contains a reproducible local baseline for Qwen2.5-0.5B-Instruct using unquantized LoRA. Reporting/gating unit tests passed locally. Model weights were NOT downloaded or run in this environment; all real training and evaluation remain to be executed on your computer.

## Setup (from repository root)

Use Python 3.10–3.12. For NVIDIA CUDA select a matching PyTorch wheel; on Apple Silicon, use arm64 Python with PyTorch MPS.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r prototype/requirements.txt
    python -m unittest discover -s prototype/tests -v

## Train and evaluate

    python prototype/train.py --max-steps 2
    python prototype/train.py --epochs 3
    python prototype/evaluate_pair.py --adapter artifacts/qwen-ticket-lora
    python prototype/report.py --input artifacts/predictions.jsonl --html artifacts/report.html --json artifacts/report.json

Open artifacts/report.html. To block a release when policies fail, add --ci to report.py (exit code 2 = blocked). The default policy tolerates 10 percentage points of protected accuracy drop and requires nonnegative target-task gain; thresholds are demonstration defaults, not validated criteria.

## Preview without model weights

    python prototype/report.py --input prototype/data/demo_predictions.jsonl --html artifacts/demo_report.html --json artifacts/demo_report.json

The sample preview is deliberately synthetic and prominently labeled illustrative.

## What it measures

- Training: 27 tiny *illustrative* support-ticket routing examples, with assistant-response masking, LoRA on q_proj/v_proj, and local adapter storage.
- Evaluation: 9 held-out routing prompts and 12 protected arithmetic / JSON / formatting prompts, using deterministic greedy base-vs-adapter inference.
- Report: per-category accuracy, before-and-after failures, and configurable deployment gate.
- Security and reproducibility: generated weights and outputs stay in the gitignored artifacts/ directory.

This is an engineering smoke test, NOT evidence of catastrophic forgetting, new-regression discovery, or SAE-guided preservation. The base model may already be good at routing. For the senior design research, increase dataset size, verify strong base capability, measure variability over multiple seeds, introduce replay and other matched baselines, and evaluate SAE-derived features separately.

## Local device support

CUDA, MPS and CPU are automatically detected or can be selected with --device. This starter uses float32 with no bitsandbytes dependency. 16 GB+ RAM/unified memory is preferable; exact resource needs are machine-dependent. Do not claim your local model has trained until the entire sequence completes and artifacts/report.json contains evaluated results.
