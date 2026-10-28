# SAE v2: BANKING77 four-run findings (2026-10-26)

## Provenance and scope
These are **user-provided local `summary.json` results** for the existing Qwen2.5-0.5B-Instruct / BANKING77 LoRA experiment on layer 12. Original files were shared in the senior-design conversation on 2026-10-26. No checkpoint, raw per-example drift values, or activations were uploaded, so numerical comparisons below are limited to the summary outputs. The trained model and SAE have not been reproduced in CI.

All runs share 4,864 calibration prompts, 1,216 validation prompts, 2,700 BANKING77 prompts and 3,380 distinct protected prompts, sampling three token positions per calibration prompt, and 3,584 SAE features. Evaluation uses the same **222 base-correct protected examples** (129 regressed / 93 retained).

## Results

| k | Val R² (reported) | Held-out base R² | Held-out tuned R² | SAE regression AUC | Raw drift AUC | PCA drift AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | .97246 | .99244 | .71323 | .52405 | .36559 | .32291 |
| 32 | .97719 | .99822 | .72972 | .31366 | .36559 | .32291 |
| 64 | .97800 | .99721 | .74202 | .33925 | .36559 | .32291 |
| 128 | .98085 | .99799 | .76366 | .36676 | .36559 | .32291 |

**Critical interpretation:** The scalar score is *magnitude of SAE activation drift*, and the AUC convention assumes **larger drift = more likely regression**. AUC below 0.5 may be an inverse association, not absence of discriminative signal. Score orientation must be learned on a separate labeled development split before test-set evaluation. Comparing orientations chosen on the test set would be invalid.

### Category confounding

The apparent pooled TopK-16 lead (**.524 vs .366 raw**) does **not** hold within capability:

| Category | n | regressed | k16 | k32 | k64 | k128 | raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arithmetic | 106 | 48 | .3003 | .3700 | .3707 | .3416 | .3962 |
| Case transformation | 37 | 22 | .3879 | .3606 | .4424 | .5273 | .4545 |
| Literal formatting | 75 | 55 | .3918 | .4809 | .4800 | .5600 | .5055 |

Within-category ROC AUC weighted by numbers of regressed–retained pairs within each reported category (approximate, because input AUCs are rounded):

| k / baseline | Within-category pair-weighted AUC |
|---|---:|
| 16 | .331 |
| 32 | .398 |
| 64 | .405 |
| 128 | .413 |
| Raw | .429 |

The three categories account for 218 examples and 125 regressions. The other four examples are all regressions, so there is no within-category ROC AUC for the remaining category. TopK-16 has worse within-category discrimination than raw drift in all three categories; the pooled ranking is confounded by between-category comparisons. The paired bootstrap CI reported for pooled k16-minus-raw (0.0557, 0.2657) does **not** correct this confounding or template dependence.

### Representation / distribution checks

- Validation reconstruction gets better as k increases; k128 reaches R²-like .98085.
- Held-out fine-tuned reconstruction remains substantially below base (e.g., k128: .76366 vs .99799).
- Held-out base active dictionary fraction is only **2.4%** at k16, compared with **45.76%** on SAE validation activations. At k128 these fractions are **46.93%** and **100%**, respectively. This suggests prompt-position or domain distribution differences and should be examined directly.
- The current `r2_vs_base_train_mean` uses a fixed calibration mean baseline, **not** conventional held-out, per-split mean R². Do not compare it directly to v1's R² until metrics/normalization are aligned.
- Current eval captures **final assistant-prefix token only**; calibration uses mixed token positions. This mismatch could favor reconstruction of a low-variance prefix state.
- Protected training and evaluation use closely related synthetic prompt templates despite exact-prompt de-duplication. A separate template-family holdout is necessary for broader claims.

## Next controlled experiment

1. Retain the layer-12 activation cache and current checkpoints; no need to repeat LoRA fine-tuning.
2. Report within-category pair-weighted AUC alongside pooled AUC and retain per-example drift scores for post-hoc checks.
3. Add `delta` reconstruction fidelity: compare `SAE(h_tuned) - SAE(h_base)` **decoded** difference against true normalized `h_tuned - h_base`; also compute held-out-centered R².
4. Evaluate at matched token positions (final assistant-prefix separately from sampled user-query token positions), with distinct domain- and template-family holdouts.
5. Use a separate *labeled* development set to determine which direction of drift predicts regression, fit any normalization/threshold, and freeze it before test evaluation.
6. Repeat at least three SAE training seeds and compare signed raw/PCA/SAE baselines, including category-conditioned results and uncertainty.
7. Advance to causal feature ablation/patching only after a stable feature-level association or shift-detection result appears. A positive reconstruction metric alone does not prove useful or causal SAE features.

**Decision:** Keep PR #17 draft; do not claim successful regression discovery or start a feature-preservation result claim yet.
