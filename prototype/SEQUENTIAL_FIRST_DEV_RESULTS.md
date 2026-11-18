# Sequential A→B learning: first development run

**Source:** team-uploaded `dev_comparison.json` from the fresh disjoint-item run (seed 43 as configured in `prototype/sequential_run.py`; the JSON does not itself encode the seed, checkpoint hashes, or training metadata). This is **development-only**, not the untouched test set. Model inference was executed locally by the team, not in GitHub CI.

## Results

| Task | Correct after A | Correct after B | Net change |
|---|---:|---:|---:|
| A (previously learned mapping) | 27/48 = 56.25% | 28/48 = 58.33% | +1 / 48 (+2.08 pp) |
| B (new mapping) | 12/48 = 25.00% | 36/48 = 75.00% | +24 / 48 (+50 pp) |

Of 27 Task A examples correct after stage A, **4 became wrong after stage B**, while **23 remained correct** (retention 23/27 = 85.19%). At the same time, **5 A examples changed from wrong to right**, causing a small net improvement. These transitions show *individual regressions*, not a net degradation of overall A accuracy.

All four A pass-to-fail cases are items `alder` and `cypress` (2 each), both expected `K2`, both changing to `V9`, all using the same dev question template: `In mode A, what is the code for item ...? Sample DEV-... Answer only the code.` It is premature to infer a general forgetting mechanism; this could reflect item- or label-specific interference, template sensitivity, or a mixture. The uploaded summary does not include the five gained examples, logits, or per-prompt predictions needed to distinguish them.

## Decision

**Do not proceed to SAE feature-preservation claims yet.** The B acquisition is encouraging (25% → 75%), but A was only 56% accurate before B, far short of a reliable remembered capability. We need a cleaner starting checkpoint.

### Next controlled study on DEV

1. **Diagnose A underlearning.** Evaluate after-A accuracy by item, label and prompt template, including on training examples versus held-out dev prompts, to distinguish lookup memorization from prompt-format transfer failure. Inspect raw outputs.
2. **Strengthen stage A acquisition** with a small, documented dev-only training schedule sweep (e.g., 4, 8, 12 epochs using the same mapping and model but separate output folders). A useful working gate is ≥90% dev accuracy. Do not use final TEST to tune.
3. **Rerun B from the selected A checkpoint**, using the *same trainable adapter*, with B checkpoints at multiple update budgets to measure B learning versus A retention, not just the endpoints. The B criterion is a meaningful improvement over pre-B performance.
4. **Check the shared-item condition** only after the disjoint baseline is understood; its task modes are behaviorally compatible but representation overlap may be stronger.
5. **Replicate across seeds**, then add replay and raw-activation regularization before the SAE-specific intervention. Always match B accuracy when comparing retention.

## Advisor-oriented interpretation

The simplified benchmark gives precisely specified learning and retention objectives and creates inspectable pass→fail examples. The initial run demonstrates B learning and some A-level transitions, but does **not** yet demonstrate the controlled forgetting frontier necessary for a convincing feature-preservation comparison. This is a productive feasibility result and a prompt to improve acquisition, not a positive SAE result.
