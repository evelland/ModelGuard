# SAE v2.3: Intrinsic dimension, stability, and new-template evaluation

This experimental extension follows the v2.2 controls: a fixed mean delta explains 75% of raw delta energy, while 32-component PCA explains 98.879% of the remaining variance. Our hypothesis is **not** that the SAE wins automatically; we must test whether sparse features convey behaviorally useful information beyond a cheap linear baseline.

## A. Exact PCA rank sweep — no model inference

```bash
git checkout feature/sae-v2-banking77
git pull
source .venv/bin/activate
python -m unittest discover -s prototype/tests -v
python prototype/sae_v23_pca.py
```

This uses the cached paired vectors and deterministic double-precision covariance eigenvectors. It reports centered validation delta R² and PCA training-variance cumulative explained for ranks 1, 2, 4, 8, 16, 32, 64, 128. Family metrics are by the saved `banking` versus synthetic protected calibration **types**, not by new template-family holdouts.

## B. Repeat SAE fits — no model inference

For an initial feasible experiment:

```bash
python prototype/sae_v23_seeds.py \
  --device mps --schemes base pooled delta --k 32 128 --seeds 42 43 44
```

This trains 18 models (3 schemes × 2 sparsity settings × 3 seeds), using the SAME paired cache. Results are written to `artifacts/experiments/banking77/sae_v23/seeds/`. All results are calibration-validation reconstruction only; it intentionally does **not** repeatedly inspect the protected benchmark for hyperparameter selection.

For the full 36-run matrix, use `--k 16 32 64 128`. Note that the seed 42 copies are new initializations using identical settings, not a separately controlled model fine-tuning seed.

## C. Freeze a new-template protected benchmark — then evaluate once

```bash
python prototype/sae_v23_holdout.py --per-family 100
```

The output path is `artifacts/experiments/banking77/sae_v23/new_template_protected.jsonl`. Four task types remain, but prompt syntax and constructions are deliberately different:
- Two-step arithmetic rather than simple addition/subtraction
- Extract a string between markers rather than echoing a string
- Uppercase an entire spaced phrase rather than a one-word token
- Two-field JSON with status/amount rather than count/ok

**Do not evaluate or tune on it before freezing the model, checkpoint and score direction.** Prompt-template difference makes this a harder transfer test but also changes difficulty distribution. Compare base-model competence and only count cases base model gets right as potential forgetting; report unconditional and conditional category denominators, and evaluate both behavior and the proposed detector. The `sae_v22` collector does not automatically collect these NEW test prompts; a paired holdout activation-collection/evaluation step still needs implementing before we can compute diagnostic AUCs. Do not confuse producing the JSONL with running the Qwen model.

The benchmark generator is deterministic and refuses to overwrite an existing file. It is **not** automatically independent of all prior research because the experiment team has already seen the benchmark generator in this PR.

## Decision criteria

1. PCA intrinsic dimension: identify rank needed for 90%, 95%, 99%, 99.9% centered train variance and compare validation performance.
2. Across SAE seeds, report median, range, variability in centered delta R², and whether a gain over rank-matched PCA is stable.
3. Prefer a simple PCA baseline if SAE gives no clear benefit in behavior-relevant predictions or interpretable feature interventions.
4. Do not claim SAE features are semantic/causal or reduce forgetting before intervention tests.

All scripts use local gitignored artifacts and no cloud compute. GitHub CI verifies offline code only.
