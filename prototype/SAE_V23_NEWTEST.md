# SAE v2.3 — Frozen new-template transfer test

This is **not** training. It compares the already trained base Qwen and LoRA on the newly generated 400-case benchmark; detectors are frozen using **old** evaluation labels before looking at new predictions.

From the repository root, with `.venv` active:

```bash
git checkout feature/sae-v2-banking77 && git pull
python -m unittest discover -s prototype/tests -v

# Step 1: Freeze benchmark SHA, checkpoint SHAs, PCA rank, score signs using OLD dev labels.
python prototype/sae_v23_newtest.py freeze

# Step 2: Run BASE and LoRA inference on the frozen protected benchmark.
python prototype/evaluate_pair.py --device mps \
  --adapter artifacts/banking77-2700-lora \
  --cases artifacts/experiments/banking77/sae_v23/new_template_protected.jsonl \
  --output artifacts/experiments/banking77/sae_v23/new_template_predictions.jsonl

# Step 3: Cache matched final assistant-prefix hidden states on exactly those prompts.
python prototype/sae_v23_newtest.py collect --device mps

# Step 4: Score behavior and the frozen detector directions.
python prototype/sae_v23_newtest.py score
```

Outputs:
- `frozen_newtest_plan.json`: benchmark/checkpoint hashes, fixed PCA rank, sign chosen from **old dev** labels
- `new_template_predictions.jsonl`: actual model responses (do not modify or reorder)
- `new_template_activations.pt`: paired hidden states aligned by benchmark case ID
- `new_template_results.json`: accuracy, gains, regressions, and conditional within-category AUC (on base-correct cases only)

**Important methodological limits:** This benchmark uses new task constructions (two-step arithmetic, marker extraction, phrase uppercasing, JSON status/amount), not entirely unseen task categories. It is not a formally blinded test because the generator and evaluation protocol are public in the PR. Base-model competence must be reported separately; AUC is undefined if a category has only regressions or only retained cases. Only the prespecified base/pooled/delta k128 seed42 SAEs and rank32 PCA plus raw drift are compared; there is no selection based on new test accuracy. Calibration and model selection on the old benchmark were exploratory.

Do **not** use `prototype/report.py` here: it requires a combined target+protected dataset, while this benchmark contains protected cases only. The standalone `sae_v23_newtest.py` reports the protected-only results.

If a step fails, stop and address the error without changing the frozen benchmark or hyperparameters. When results are available, share only `new_template_results.json`; raw model responses/activation vectors need not be posted.
