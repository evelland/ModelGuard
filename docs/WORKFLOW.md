# Collaboration and Async Working Agreement

## Single sources of truth
- GitHub Issues: who owns each task, what done means, blockers, links to outputs.
- Pull requests and repo: all code and technical documentation changes.
- DECISIONS.md: meaningful scope/method choices and their rationale.
- Google Drive (team-selected folder): slides, course-facing prose, meeting notes.
- Shared calendar (team-selected): meetings, presentations, faculty deadlines; hard deadlines are also copied into docs/DELIVERABLES.md.
- Chat is coordination only, not a durable record.

## GitHub issue workflow
**Backlog → Ready → In progress → Review → Done.** Add a GitHub Project board with these statuses after team setup.

Each issue needs:
- A specific outcome and deliverable (not merely 'investigate').
- One directly responsible owner, reviewer if relevant, and due date/relative milestone.
- Scope and acceptance criteria.
- Dependencies, files/data needed, and evidence of completion.

No owner or date is assigned until agreed; don't fabricate commitments.

## Async status updates
Each member posts twice per week to their active issue(s):
- **Done:** link to merged PR, note, plots, run, or result.
- **Next:** next testable output.
- **Blocked by:** an explicit decision, person, compute resource, or dataset (or 'none').

Weekly sync (proposed 30–45 min):
1. Quick demonstration of what actually works.
2. Resolve decisions and blockers.
3. Review experiments / reproducibility.
4. Confirm owners and nearest deliverable.

Record 3–6-line notes: decisions, owners, action items, due dates. Put durable decisions in DECISIONS.md.

## Branch and review policy
- Branches: `feature/<issue>-description`, `experiment/<issue>-description`, `docs/<issue>-description`.
- Link issue in PR; summarize method, validation, limitations.
- At least one teammate reviews substantive work before merge when possible.
- Avoid merging work whose outputs cannot be reproduced or interpreted.

## Research standards
A reported run includes:
- Exact base/checkpoint/model/SAE references, code commit SHA.
- Dataset version and train/validation/test splits.
- Training and evaluation config, seed(s), hyperparameters.
- Metric definitions, reference prompt provenance, compute notes.
- Raw results location and a short interpretation.
- Negative results and failures are documented, not silently dropped.

Keep held-out evaluations separate from SAE feature selection and preservation-reference examples. Define thresholds before looking at the final comparison.

## Definition of done
An issue is done when acceptance criteria are met; the associated outputs are linked; a reviewer can reproduce or inspect the work; relevant docs are updated; and blockers or follow-up work are captured as separate issues.
