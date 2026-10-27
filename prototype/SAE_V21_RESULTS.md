# BANKING77 SAE v2.1 — Results (27 October 2026)

## Source and status

Four model-backed local `analysis.json` files supplied by the team for SAE v2.1, generated with `prototype/sae_v21_analysis.py`. These outputs contain summary statistics only, not the activation cache, raw per-example scores or SAE checkpoints. **Exploratory findings**, not independently reproduced or a pristine held-out evaluation.

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`; adapter: `artifacts/banking77-2700-lora`
- Qwen hidden-state index 12, SAE dictionary width 3,584; TopK settings 16, 32, 64, 128
- All experiments use the **same** dev/test IDs and `seed=42`: 89 development, 133 test; original set had 222 base-correct protected cases
- Test categories: arithmetic (64: 29 regressions), case transformation (22: 13), literal formatting (45: 33), strict JSON (2: both regressions). JSON has no within-category ROC AUC.

## Reconstructing the base-to-fine-tuned activation *difference*

`delta_r2_vs_zero` is 1 – ||Δ – reconstructed Δ||² / ||Δ||²; it is **not** ordinary hidden-state reconstruction R².

| TopK | Delta R² vs zero | Mean delta cosine | Median relative delta error | Mean reconstructed delta L2 |
|---:|---:|---:|---:|---:|
| 16 | 0.02077 | 0.28109 | 0.98852 | 5.93995 |
| 32 | 0.08297 | 0.34111 | 0.95999 | 5.95947 |
| 64 | 0.12735 | 0.40665 | 0.92658 | 6.93128 |
| 128 | **0.20027** | **0.53451** | **0.90141** | **9.46197** |

True delta mean L2 is **11.38980** for every run. Base SAE activation reconstruction was high in v2, but **fine-tuning-change reconstruction remains poor**, even for TopK 128; this distinction matters for ModelGuard. The reconstructed delta underestimates average magnitude, but its per-example error needs further inspection.

## Regression detection (direction chosen on development data)

The learned direction was `smaller` for SAE drift, raw drift and PCA. Below are fixed-orientation, *within-category pair-weighted* test ROC AUCs.

| Method | Within-category ROC AUC | Pooled ROC AUC | Paired bootstrap CI for SAE minus raw conditional AUC |
|---|---:|---:|---|
| TopK 16 | **0.64856** | 0.46869 | [−0.05236, +0.17212] |
| TopK 32 | 0.59620 | 0.68762 | [−0.07788, +0.08966] |
| TopK 64 | 0.59097 | 0.65909 | [−0.12369, +0.13743] |
| TopK 128 | 0.59359 | 0.64170 | [−0.10602, +0.10340] |
| Raw drift | 0.59097 | 0.64402 | — |
| PCA (32 components) | 0.53927 | 0.66048 | — |

**Interpretation:** TopK 16 is numerically best at 0.64856, an improvement of +0.05759 over raw, but its exploratory bootstrap CI includes zero. The different pooled AUC rankings underscore task-category confounding. No TopK run demonstrates a statistically established improvement over raw drift. Do not orient metrics using the test set.

Per-category fixed-orientation test AUC:

| Category | TopK 16 | TopK 32 | TopK 64 | TopK 128 | Raw |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 0.65911 | 0.59507 | 0.58621 | 0.61379 | 0.61281 |
| Case transformation | 0.69231 | 0.73504 | 0.55556 | 0.53846 | 0.64103 |
| Literal formatting | 0.60859 | 0.55808 | 0.61364 | 0.55808 | 0.52020 |

## What we can and cannot claim

- Demonstrated: Code works on local model-produced activations and sparse dictionaries are able to approximate some activation change, increasing with TopK.
- Not demonstrated: Robust, generalizable regression detection beating raw activations; semantic meaning of a feature; causal contribution; forgetting prevention.
- The dev/test sets use **previously investigated examples**: they are not an independent confirmatory test. Only one SAE training seed is represented. Synthetic prompt families overlap between calibration and evaluation. No template-cluster bootstrap. SAE calibration samples various token positions, whereas test uses only final assistant-prefix tokens.

## Proposed SAE v2.2 controlled design

**Scientific question:** Does training an SAE to encode both sides of a fine-tuning change — rather than base activations only — improve *delta reconstruction* and meaningful regression prediction?

1. Collect *paired base and tuned activations on the same NEW calibration prompts and matched token positions*; **never** use the existing protected test examples to train this dictionary. Keep BANKING77 training prompts distinct from evaluation and split by prompt/template family.
2. Compare three matched-width, matched-TopK dictionaries: **base-only** (current), **base+tuned pooled** training, and **delta-trained** TopK on `h_tuned – h_base`. Delta-trained coefficients need not be aligned with base-only features and are not automatically semantic.
3. Train only on calibration prompts; select sparsity and reconstruction configuration on a separate, template-disjoint validation set, using `delta_r2_vs_zero` and cosine plus feature stability / dead-feature coverage. Add linear/PCA delta reconstruction as a non-SAE control.
4. Assess the same fixed, separately labeled development-trained drift-direction rules on a **new, untouched protected test family**. Prespecify category-weighted AUC and interval method. Use multiple SAE random seeds.
5. Only once consistent predictive associations emerge, test intervention/patching or SAE-guided preservation with matched random-feature and raw-activation controls.

**Next engineering dependency:** The existing v2 cache saves calibration `train_h` and `valid_h` from the base model only. v2.2 needs **new paired fine-tuned calibration activations**, not merely another analysis of the four existing `analysis.json` files. Retain PR #17 as a draft until its scope and evaluation are reviewed.
