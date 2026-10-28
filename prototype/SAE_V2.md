# SAE v2 — BANKING77 TopK diagnostic

This is an experimental follow-up to PR #16, not evidence of causal interpretability or forgetting prevention. It does not alter Qwen weights.

## Prerequisites

Use the same local `artifacts/banking77-2700-lora`, `artifacts/experiments/banking77/train.jsonl`, and `artifacts/experiments/banking77/report.json` from the previous run. Install `prototype/requirements.txt`.

## Run

From the repository root:

```bash
python -m unittest discover -s prototype/tests -v
python prototype/preflight_sae_v2.py
python prototype/sae_v2.py collect --device mps --layer 12 \
  --bank-prompts 2700 --protected-prompts 4000 --positions-per-prompt 3
for k in 16 32 64 128; do
  python prototype/sae_v2.py fit --device mps --expansion 4 \
    --k "$k" --epochs 40 --patience 8 \
    --out "artifacts/experiments/banking77/sae_v2/topk${k}"
done
```

`collect` runs Qwen once per prompt, caching base calibration and validation activations plus paired base/tuned held-out activations. `fit` reuses these cached tensors for TopK, reconstruction validation, exploratory AUC against raw and PCA drift, and bootstrap uncertainty. Training/validation are separated by prompt; the protected test prompts are excluded from SAE training.

Choose SAE hyperparameters from reconstruction on the validation set, *not* held-out regression AUC. Report sparsity, held-out base and tuned reconstruction, and all baseline AUCs; compare on the same prompts. Run at least three SAE seeds and inspect per-category results where there are adequate examples.

**Limitations:** protected cases share a small number of templates; exact-text exclusion is not a template-family holdout. Prompt-level bootstrap does not account for template clustering. PCA is fitted only on base calibration features. SAEs are observational; demonstrating preservation requires a separate intervention experiment. Synthetic unit tests do not establish real BANKING77 results. Check for distribution shift when tuned reconstruction is poor.

CI runs Python syntax checks and offline unit tests on pull requests, but does not download Qwen or run BANKING77. The preflight checks local inputs and eligible regression/retention labels before the expensive model run.\n\nNew outputs are under gitignored `artifacts/experiments/banking77/sae_v2/`; do not commit model weights or datasets.
