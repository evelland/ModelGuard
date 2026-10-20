# ModelGuard — Senior Design 2026–27

ModelGuard is a research-and-engineering project investigating how to detect and reduce behavioral regressions when language models are fine-tuned.

> A model may improve on a new task while losing previously reliable capabilities. We want to measure these changes, study whether sparse autoencoder (SAE) features help explain them, and test whether selectively preserving such features can mitigate forgetting.

## Research questions
1. Discovery: Can changes in internal SAE features reveal or explain behavioral regressions better than predefined evaluation results?
2. Intervention: Can SAE-guided preservation during fine-tuning retain selected prior capabilities without sacrificing new learning?

## Deliverable
- **MVP:** reproducible base-versus-fine-tuned evaluation pipeline and a machine + human-readable regression report.
- **Research validation:** SAE feature analysis and a selective preservation objective tested against matched controls and strong forgetting-mitigation baselines.
- **Product:** interactive dashboard, discovery of unanticipated regressions, expansion to additional model families.

## Documentation
- [Working Agreement](docs/WORKFLOW.md) — async updates, ownership, pull requests, definition of done.
- [Deliverables Tracker](docs/DELIVERABLES.md) — external deadlines and internal checkpoints (dates not yet supplied).
- [Decisions](DECISIONS.md) — confirmed choices and open questions.
- **GitHub Issues:** discrete tasks and evidence of completion. **GitHub Projects:** proposed board/roadmap (to be created in GitHub UI).
- **Shared calendar / Drive:** meeting invitations and collaborative presentations / course submissions (not set up from this repo).

## Pending implementation layout
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
