# ModelGuard — Senior Design 2026–27

ModelGuard is a University of Pennsylvania senior design research-and-engineering project investigating how to **detect and reduce behavioral regressions when language models are fine-tuned**.

> A model may improve on a new task while losing previously reliable capabilities. We want to measure these changes, study whether sparse autoencoder (SAE) features help explain them, and test whether selectively preserving such features can mitigate forgetting.

**Project status:** scope and experimental protocol are provisional. No positive SAE-preservation result is assumed.

## Two research questions
1. **Diagnosis:** Can changes in internal SAE features reveal or help explain behavioral regressions beyond predefined evaluation results?
2. **Intervention (primary hypothesis):** Can SAE-guided preservation during fine-tuning retain selected prior capabilities without substantially sacrificing new-task learning?

Internal feature changes are **not** proof of behavior or causality. Behavioral benchmarks and controlled interventions are required.

## What we'll build
- **MVP:** reproducible base-versus-fine-tuned evaluation pipeline and a machine-readable + human-readable regression report.
- **Core research:** SAE feature analysis and a selective preservation objective tested against matched controls and strong forgetting-mitigation baselines.
- **Stretch:** interactive dashboard, discovery of unanticipated regressions, additional model families.

See [Project Charter](docs/PROJECT_CHARTER.md), [Roadmap](docs/ROADMAP.md), [Experiment Spec](docs/EXPERIMENT_SPEC.md).

## How we work
- [Working Agreement](docs/WORKFLOW.md) — async updates, ownership, pull requests, definition of done.
- [Deliverables Tracker](docs/DELIVERABLES.md) — external deadlines and internal checkpoints (dates not yet supplied).
- [Decisions](DECISIONS.md) — confirmed choices and open questions.
- **GitHub Issues:** discrete tasks and evidence of completion. **GitHub Projects:** proposed board/roadmap (to be created in GitHub UI).
- **Shared calendar / Drive:** meeting invitations and collaborative presentations / course submissions (not set up from this repo).

## Suggested implementation layout (to populate as code is written)
```
configs/                  # versioned run configuration
src/training/             # baseline fine-tuning and preservation
src/evaluation/           # benchmarks and result comparisons
src/interpretability/     # SAE feature extraction and analysis
src/reporting/            # report generation
tests/                    # deterministic tests
experiments/              # lightweight run manifests + result indexes
docs/                     # decisions, literature notes, protocols
```
Do not commit model checkpoints, private datasets, access tokens, or large activation dumps. Keep artifacts in suitable storage and record immutable references + configuration for reproducibility.

## First steps
1. Review and approve the [Project Charter](docs/PROJECT_CHARTER.md).
2. Enter course requirements and dates in [Deliverables Tracker](docs/DELIVERABLES.md) and a shared calendar.
3. Set workstream owners and issue priorities.
4. Select one feasible base model, target task, protected capability, compute budget, and evaluation protocol.
5. Reproduce a measurable regression **before** prioritizing preservation experiments or polishing the UI.
