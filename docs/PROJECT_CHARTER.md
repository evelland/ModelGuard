# ModelGuard — Project Charter (Draft for Team Approval)

## Problem statement
Fine-tuning language models for new tasks can improve specialized performance while degrading previously learned behaviors. Standard regression tests are limited to behaviors explicitly included in the suite. We will explore internal representation signals as a complement to—not a replacement for—behavioral evaluation.

## Working goal
Build a reproducible fine-tuning regression analysis system and **experimentally test** whether SAE-guided selective representation preservation provides a better stability–plasticity tradeoff.

## Research questions
1. Can SAE feature changes help anticipate, identify, or interpret benchmark-measurable regressions?
2. Does explicitly constraining selected SAE features during training reduce forgetting at comparable new-task performance?

## Deliverable tiers
### Required MVP (engineering)
- One selected, legally usable, tractable open model and clearly versioned dataset/task.
- Scriptable base-model evaluation and at least one fine-tuning pipeline.
- Base vs. fine-tuned performance comparisons on target and held-out protected tasks.
- Reproducible run manifest and generated regression report (initially static HTML/Markdown/CSV/plots acceptable).
- Setup instructions + small end-to-end demonstration.

### Research objective (main scientific contribution)
- Consistent SAE activation extraction at one or more carefully selected layers.
- Defensible selection and validation of candidate protected-capability features.
- Feature-preservation loss implemented in training.
- Controlled comparison with ordinary training, a matched representation-preservation control, random-feature control, and suitable established forgetting baselines.
- Multiple seeds / settings where feasible; stability–plasticity comparison and limitations.

### Stretch only
- Automated discovery of regressions not included in an original evaluation suite.
- Polished interactive UI, automated natural-language reports.
- Cross-model compatibility, multiple fine-tuning techniques/domains, deployment integration.

## What success looks like
**Engineering:** a teammate can reproduce the base-vs-fine-tuned experiment and report from documented configs.

**Scientific:** either (a) evidence that selective SAE feature preservation improves retention at comparable target-task learning, or (b) a well-controlled negative/mixed result explaining when it does not help. The latter is valid research, not a product success claim.

**Presentation:** a demonstrable model-change example and interpretable quantitative result; report causal limitations plainly.

## Out of scope for first iteration
- Production-grade hosted model monitoring or SaaS; proving regulatory compliance; safety guarantees; support for proprietary inaccessible weights; universal explanation of model behavior.

## Critical feasibility gate
Before substantial UI development or feature-preservation optimization:
- baseline fine-tuning works,
- target-task gain and protected-capability regression are measurable,
- evaluation is stable enough to compare methods,
- compute fits the team's budget.

If forgetting is weak or absent, revise task/method while keeping a transparent experimental record. Do not claim successful prevention without observable baseline forgetting.

## Decisions requiring team agreement
- Team members and leads; adviser/course rubric; meeting cadence.
- Available GPU time, budget, and storage rules.
- Base model, compatible SAE or SAE-training plan, datasets, prompts and metrics.
- Protected behavior, target task and split policy.
- Exact baseline methods, controls, seeds, confidence criteria.
- Deliverable dates and presentation format.
- MVP interface (command-line reports first; UI only if justified).

**Approval:** Pending team review. Any changes after approval should be recorded in ../DECISIONS.md.
