# First Experiment — Specification v0.1 (Unapproved)

This document is a **decision scaffold**, not a claim that the team has chosen a model, SAE, or dataset.

## Objective
Establish a **reproducible baseline forgetting setting**: fine-tuning increases target-task performance but decreases at least one previously reliable capability. Then test whether a selected feature-preservation loss improves retention for a comparable target-task gain.

## To be selected
| Field | Choice / notes |
|---|---|
| Base model + exact revision | TBD; initial preference: accessible weights, modest memory, reproducible inference |
| SAE + layer(s) + revision | TBD; prefer compatible feature dictionary; check post-fine-tune alignment |
| Target fine-tuning task + metrics | TBD; evaluate held-out task examples |
| Protected capability + metrics | TBD; verify base score is sufficiently high and reliable |
| Reference examples for preservation | TBD; strictly distinct from final test examples |
| Training approach | TBD (e.g., LoRA or full fine-tuning); keep controls comparable |
| Compute / storage budget | TBD |
| Seeds, sample sizes and precision | TBD before comparison |
| Primary outcome | Protected-capability retention conditioned on target-task learning |
| Reporting | Paired scores, uncertainty / variability, Pareto plot, run manifests |

## Experiment phases
1. **Base evaluation:** frozen initial checkpoint on target and protected test sets.
2. **Unrestricted baseline:** ordinary task fine-tuning; compare new-task gain vs. retained capabilities.
3. **Representation diagnosis:** compare base/fine-tuned activations on the SAME inputs with a consistent SAE encoder; inspect high-activation examples and non-target controls.
4. **Intervention:** fine-tune with feature-specific preservation penalty and swept strength parameter.
5. **Controls and baselines:** ordinary fine-tuning; raw-activation preservation (matched cost where feasible); random SAE feature preservation; replay or distillation/EWC when appropriate; same data and comparable training budget.
6. **Evaluation:** held-out protected and new-task test sets, multiple seeds if feasible, tradeoff curves and ablations.

## Candidate loss (for investigation)
```text
L = L_task + lambda * L_preservation

L_preservation =
  mean_{x in P} || w ⊙ (SAE(h_theta(x)) - SAE(h_base(x))) ||^2
```

P = separate preservation-reference prompts, w = selected/weighted latent features, SAE encoder frozen. This is only one candidate objective; representation matching on selected prompts could affect learning or encourage shortcut solutions.

## Avoid confounds
- No use of final evaluation labels to select SAE features.
- Same tokenizer, prompt formatting, inference settings and scoring rules for paired model comparisons.
- Don't compare LoRA to full fine-tuning as though they were equal-budget controls.
- Account for run-to-run variance, SAE reconstruction limits, changing feature geometry, and risks from interpreting correlation causally.
- If no reproducible forgetting occurs, do not interpret an apparently 'protected' model as evidence of forgetting mitigation.

## Minimal results manifest fields
```json
{
  "run_id": "TBD",
  "git_sha": "TBD",
  "base_model_revision": "TBD",
  "fine_tuned_checkpoint": "TBD",
  "sae_revision": "TBD",
  "datasets_and_splits": {},
  "seed": null,
  "training_config": {},
  "metrics": {},
  "artifacts": {},
  "notes": ""
}
```

## Go/no-go for pursuing SAE preservation
Proceed only once one teammate independently reproduces M3, the SAE is technically compatible, feature selection methodology is documented, and training with the SAE in the loop fits compute.
