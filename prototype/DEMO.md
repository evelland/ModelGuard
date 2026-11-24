# End-of-semester demo (4 minutes)

Shows the whole product loop on the controlled A → B benchmark: fine-tune, compare against the previous model, gate the release, and show that a preservation method changes the decision.

## Setup (before presenting)

Run once from the repository root. It needs the seed-43 sweep outputs in `artifacts/experiments/sequential/replay_disjoint_seed43` (see `REPLAY_BASELINE.md`).

~~~bash
R=artifacts/experiments/sequential/replay_disjoint_seed43
python3 prototype/sequential_report.py --after-a $R/probe_after_A.jsonl --after-b $R/probe_after_B_r00.jsonl \
  --html artifacts/demo/plain.html --json artifacts/demo/plain.json --gate mcnemar
python3 prototype/sequential_report.py --after-a $R/probe_after_A.jsonl --after-b $R/probe_after_B_r10.jsonl \
  --html artifacts/demo/replay10.html --json artifacts/demo/replay10.json --gate mcnemar
open artifacts/demo/plain.html artifacts/demo/replay10.html
~~~

## Script

1. **Setup (30s).** "We taught the model mapping A, then kept training the same adapter on a new mapping B. ModelGuard treats Task A as the behavior we must keep and Task B as the new task."
2. **Plain fine-tune (90s).** Open `plain.html`. The deploy gate reads **BLOCK**: Task B went to 100%, while Task A lost 11 of 36 answers. The paired McNemar test gives 11 regressions vs 0 gains, p ≈ 0.0005. Scroll to a failure, e.g. *cedar*: correct (T4) after Task A, wrong (R7) after Task B.
3. **The gate as CI (20s).** Run the same command with `--ci` in a terminal: exit code 2, which fails a pipeline.
4. **With 10% replay (60s).** Open `replay10.html`. Same training budget; the gate reads **PASS**, with 0 Task A regressions and Task B still 100%.
5. **Close (20s).** "Replay needs the old training data. Next semester we test methods that only need a few reference prompts: KL anchoring (3 failures left), raw-activation and SAE-feature preservation."

## Q&A notes

- **Why do 5% replay (2 failures) and KL (3 failures) pass?** With 36 probes, 2–3 regressions against 0 gains is not statistically significant (p = 0.25 and 0.125). A team can switch to the stricter `--gate threshold --max-protected-drop 0` if any regression should block.
- **"Base" in the report** is the model after Task A, the release currently in production. The candidate is the model after Task B.
- One seed shown. Seed 44 replicates the plain-vs-replay result (13 → 3).
