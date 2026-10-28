# SAE v2.2 — Paired Qwen LoRA experiment results (27 October 2026)

## Provenance

Six local model-backed `summary.json` files supplied by the team from `prototype/sae_v22.py fit` on the existing Qwen2.5-0.5B-Instruct BANKING77 LoRA at hidden-state index 12. Team uploads correspond to `base`, `pooled`, and `delta` schemes at `k=32` and `k=128`. Only summary files were uploaded; checkpoints and paired activation tensors remain local.

All runs use the same 4,830 training and 1,207 calibration-validation prompts, one final assistant-prefix token per prompt, and a 3,584-feature dictionary. `pooled` has twice as many *vectors* as `base` or `delta` because every prompt yields base and tuned vectors. Exact normalized evaluation prompts were excluded from calibration; protected prompt *templates* overlap, so this is not template-family held-out. SAE training seed is 42 for every run.

## Delta reconstruction

R² reference is **zero activation change** in original hidden-state units; the metric does **not** check whether the SAE beats a constant mean-delta predictor.

| Scheme | k | Validation delta R² vs 0 | Validation delta mean cosine | Validation median relative error | Previously inspected protected-set delta R² |
|---|---:|---:|---:|---:|---:|
| Base-only | 32 | 0.21926 | 0.47713 | 0.89948 | 0.15188 |
| Base-only | 128 | 0.30645 | 0.65680 | 0.83742 | 0.27126 |
| Pooled base+tuned | 32 | 0.99342 | 0.99668 | 0.07673 | 0.99299 |
| Pooled base+tuned | 128 | 0.99550 | 0.99774 | 0.06435 | 0.99553 |
| Delta-trained | 32 | 0.99613 | 0.99800 | 0.06058 | 0.99542 |
| Delta-trained | 128 | **0.99746** | **0.99873** | **0.04813** | **0.99749** |

Mean true validation delta L2 = **5.04147** for all six runs. This is a dramatic difference from the base-only trained SAE and a strong *zero-reference* reconstruction result. It is **not yet evidence** that pooled/delta representations encode prompt-dependent information: if base→LoRA produces a nearly fixed offset across these prompts, a **mean-delta predictor** may score equally well.

**Critical next control:** Fit the mean activation difference on training pairs. On held-out validation pairs compute its R² against zero, then report `R²_vs_train_mean_delta = 1 - SSE(model)/SSE(mean_train_delta)`. Also report `PCA(delta)` projection performance at 32 and 128 components, plus category-specific and centered-delta errors. Check whether most of the variance captured comes from a near-constant global shift. Do not assume sparse SAE representation was needed until a simple constant or linear predictor is ruled out.

## Regression detection (exploratory)

AUC is **test split within-category pair-weighted** after selecting score direction on dev examples. Test n=133, dev n=89; these cases were inspected in v2.1, so the split is not pristine. Raw activation conditional AUC = **0.59097** for all six.

| Scheme | k | Within-category regression AUC | Pooled AUC |
|---|---:|---:|---:|
| Base-only | 32 | 0.55890 | 0.63729 |
| Base-only | 128 | **0.64398** | 0.50186 |
| Pooled | 32 | 0.62827 | 0.62245 |
| Pooled | 128 | 0.62893 | 0.66512 |
| Delta-trained | 32 | 0.56872 | 0.37894 |
| Delta-trained | 128 | 0.57003 | 0.37941 |
| Raw baseline | — | 0.59097 | 0.64402 |

**Key observation:** The delta-trained SAE achieves the best zero-reference delta reconstruction but the worst AUCs among the three approaches. The strongest category-conditioned ranker is base-only k128, even though it reconstructs only ~31% of the activation change by the current metric.

Only **one SAE seed** and an already-used protected benchmark are represented. No per-example scores, bootstrap AUC differences, confidence bounds, causal tests, or untouched families were supplied for v2.2. Do not claim superiority of any method over raw activations from these results. In addition, delta-trained SAE coefficients represent *changes directly*, whereas the other methods compute a *difference of model encodings*, so their norm scores are not semantically equivalent.

## Decision

1. **Preserve all six checkpoints** and the paired activation cache; no Qwen rerun is necessary for the next analysis.
2. Next experiment is a mean-delta/centered-variance and PCA control on exactly the same held-out calibration activation vectors, including family breakdown. Implement as a **post-hoc script**, avoiding retraining of model or SAEs.
3. Treat robust regression prediction and causal feature attribution as separate research objectives. For any generalization claim, freeze settings and collect a template-family-disjoint new protected benchmark; repeat SAE seeds.
4. Keep PR #17 as a **draft**, but note that code-path offline CI and scientific readiness are distinct.

### Uploaded source filenames

`summary(7).json` = base32; `summary(3).json` = base128; `summary(20261027-164758).json` = pooled32; `summary(5).json` = pooled128; `summary(4).json` = delta32; `summary(6).json` = delta128.
