# ModelGuard — Milestone Roadmap

Dates are intentionally **TBD** until the official course calendar and team schedule are confirmed. These are technical gates, not invented academic submission deadlines.

| Gate | Outcome | Exit criteria |
|---|---|---|
| M0 — Scope and coordination | Approved charter, GitHub board, owners, compute and course dates | Charter reviewed; task ownership established; deliverables calendar filled |
| M1 — Experiment contract | Initial model/task/protected behaviors and metrics chosen | Protocol, splits, resource estimates, reproducibility checklist signed off |
| M2 — Baseline pipeline | Train + evaluate base and adapted checkpoints | One scripted reproducible run; metrics and configs saved |
| M3 — Demonstrate forgetting | One controlled measurable tradeoff | Target improvement and protected-task degradation validated, with variance checks |
| M4 — SAE diff | Extract aligned features and compare activations | Feature-change analysis linked to examples and behavioral evaluations |
| M5 — Preservation study | Run SAE-regularized training + controls | Matched comparisons including random-feature and raw-activation controls |
| M6 — Integration | ModelGuard report and consolidated evidence | Teammate can run the demo; plots and limitations are documented |
| M7 — Final course deliverables | Final presentation/report/demo | Rubric checklist and rehearsal complete |

## Suggested scheduling rule
For each official presentation or submission: **T−7 days** draft; **T−3 days** group review; **T−1 day** final version and rehearsal. Put actual dates in the calendar only after the course deadlines are confirmed.

## Workstream dependencies
- Evaluation design precedes trustworthy training comparisons.
- Baseline reproducibility precedes the claim that forgetting occurred.
- SAE selection depends on model/layer choice; feature validation precedes causality claims.
- Preservation experiments depend on an SAE analysis path and demonstrated forgetting.
- Reporting code can start early with mock/example fixtures clearly labeled synthetic, but polishing waits for real results.

## Decision gates
**Gate A, after M2/M3:** Is regression measurable, or must the task / model change?

**Gate B, after M4:** Are SAE features stable, informative, and computationally usable enough to justify regularization?

**Gate C, after M5:** Is there evidence of SAE-specific improvement beyond generic representation regularization? If not, report the negative or mixed result and emphasize diagnostic capabilities.

## Course requirements checklist
- [ ] Add official proposal/presentation dates
- [ ] Add adviser meeting dates
- [ ] Add midterm review/progress report requirements
- [ ] Add final report, poster, presentation, demonstration dates
- [ ] Identify designated person for submitting each artifact
