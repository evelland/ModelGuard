# SAE v2.3 — New-template protected benchmark outcome (28 October 2026)

## Provenance

One team-provided `new_template_results.json`, produced by the frozen v2.3 workflow. No raw `new_template_predictions.jsonl` or activation cache was supplied for inspection, so we **cannot explain why** the base model failed whole categories or verify generation/scorer effects. Report file benchmark SHA256: `d382ea382e68eb6fdfe75dd540e44e61f180f66d77484e3f5230658c792aedad`. Prediction file SHA256: `2cc7936bde4c9a06ff1c02d13a5b3af0b91c4867a5a6396821093369601c188b`.

## Behavior

| Capability | n | Base correct | LoRA correct | New regressions (base yes, LoRA no) | Retained (both yes) | Gains (base no, LoRA yes) |
|---|---:|---:|---:|---:|---:|---:|
| Arithmetic | 100 | 0 | 0 | 0 | 0 | 0 |
| Case transformation | 100 | 59 | 4 | **55** | **4** | 0 |
| Literal formatting | 100 | 0 | 0 | 0 | 0 | 0 |
| Strict JSON formatting | 100 | 0 | 27 | 0 | 0 | 27 |
| Total | 400 | 59 | 31 | 55 | 4 | 27 |

- Base accuracy = **59/400 = 14.75%**; LoRA accuracy = **31/400 = 7.75%** (net −7 percentage points).
- New regressions = **55/400 = 13.75%**, but more relevant **55/59 = 93.22% of base-correct cases**. Only four retained base-correct examples remain.
- Gains = **27/400 = 6.75%**, all in JSON formatting.
- Both wrong = 314/400. These categories were not base-competent on this task construction; a zero regression count on them is **not** proof that they were preserved.
- The benchmark changed task difficulty and format, as well as surface wording. This materially complicates extrapolating from earlier synthetic probes.

## Frozen detectors on new templates

All method directions were chosen **before** new-test scoring on OLD dev labels (`smaller` drift predicts regression). The only evaluated base-correct category is case transformation, **55 regressions and 4 retained cases**, giving 220 within-category positive–negative pairs. In this special case pooled and within-category AUC coincide.

| Frozen score | ROC AUC |
|---|---:|
| Raw activation delta norm | 0.27273 |
| PCA centered delta norm (rank 32) | 0.48182 |
| Base-only SAE TopK 128 | 0.40909 |
| Pooled SAE TopK 128 | 0.49545 |
| Delta-trained SAE TopK 128 | **0.51364** |

None demonstrates reliable discrimination or an SAE advantage. The four negative (retained) examples are far too few for stable ranking: the effective information is much smaller than the 400-test count suggests. Do not reverse the frozen score orientation based on this result or optimize feature/sparsity settings on this now-seen benchmark.

## Immediate diagnostic priority

1. **Audit raw responses** from `artifacts/experiments/banking77/sae_v23/new_template_predictions.jsonl` without regenerating, modifying cases or changing validators. Print representative base/fine answers for each category: base wrong in arithmetic/literal/JSON, case-transform regressions, and JSON gains. Check whether wrong outputs are task failure, excessive explanation, JSON fences/structure, or truncation. The summary JSON alone does not identify which cause applies.
2. Audit the ability of base model to follow each *new* prompt family. A transfer suite on which base scores 0/100 in three categories is poorly suited for conditional forgetting detection. **Do not retune the frozen score or selection based on these 400 cases.**
3. For a future independently generated benchmark, preregister a protocol emphasizing base competence across several distinct categories (e.g., base-only development screening on **separate** candidate prompts, task-difficulty stratification, and fixed metrics), then freeze it before fine-tuned inference and detector evaluation. Avoid choosing easy examples using current test outcomes. Add multiple independent prompt-template families per capability and sufficient retained/regressed diversity; report per-category base correct, regressions and gains.
4. Treat SAE regression prediction and feature interpretability as separate goals. A strong SAE-vs-PCA reconstruction result alone is not a guardrail; prioritize a simple behavior-level regression gate as the useful product baseline, with SAE diagnostics as exploratory instrumentation.

## Decision

**Keep PR #17 draft and preserve all frozen artifacts.** The new-template transfer experiment is complete as an *exploratory failure of generalization*, not a positive result for SAE-based regression prediction. Next experiment should audit why base competence was zero in most categories, then construct a prespecified, representative evaluation protocol. No feature-preserving fine-tuning claims yet.
