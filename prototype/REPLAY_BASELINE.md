# Replay baseline for the sequential A → B study

Control #2 from `SEQUENTIAL_FAMILIAR_FORGETTING_RESULTS.md`: does mixing previously seen Task A examples into stage-B training reduce forgetting of A, at a **matched stage-B budget**?

## What changed

- `prototype/sequential_replay.py`: `build_replay_mix(b_rows, a_rows, fraction, seed)`.
  - Replaces `round(fraction × |B|)` stage-B rows with Task A **training** rows, so the stage-B example count (and therefore the optimizer update count) is identical across fractions.
  - Both the kept B rows and the replayed A rows are stratified evenly across items, so no item is starved or over-represented.
  - Replay rows must be Task A prompts; overlap with B prompts is rejected. Replay never draws from dev/test/probe prompts or the reserved `reference_A.jsonl`.
  - `fraction = 0` returns the original B rows unchanged, so the 0% run is the plain A → B baseline.
- `prototype/sequential_train.py`: new stage-B flags `--replay-data <train_A.jsonl>` and `--replay-frac <f>`. Run metadata records the replay count, seed and a SHA-256 of the mixed training set.
- `prototype/sequential_replay_sweep.py`: end-to-end runner. Reproduces the first clean forgetting setting (disjoint, seed 43, A 8 epochs, B 2 epochs, fresh-nonce familiar-template probes), then continues the **same saved `adapter_A`** once per replay fraction and writes `comparison_rNN.json`, `replay_sweep_summary.{json,md}` and `replay_tradeoff.png`. Existing outputs in `--root` are reused, so a previously trained A8 adapter can be copied in instead of retraining.
- `prototype/tests/test_sequential_replay.py`: offline tests (matched budget, A-training-only replay, stratification, determinism, input validation, sweep plan, summary). Added to the sequential CI workflow.

## Also on this branch: paired McNemar release gate

Peer review flagged that a fixed 2-point accuracy threshold sits inside sampling noise for a 300-prompt suite. `prototype/report.py` now always reports an exact one-sided McNemar test on the protected suite (regressions = base-pass → candidate-fail, gains = base-fail → candidate-pass), and `--gate mcnemar [--alpha 0.05]` blocks only when regressions significantly exceed gains. The default `--gate threshold` keeps the old behavior. On the midterm BANKING77 run (129 regressions vs 11 gains) the test gives p < 1e-20, so the BLOCK holds under either gate.

~~~bash
python prototype/report.py --input artifacts/predictions.jsonl --gate mcnemar --ci
~~~

## Run locally (Mac, MPS)

~~~bash
python3 -m venv .venv && .venv/bin/pip install -r prototype/requirements.txt matplotlib
python3 -m unittest prototype.tests.test_sequential_replay -v
.venv/bin/python prototype/sequential_replay_sweep.py --dry-run
.venv/bin/python prototype/sequential_replay_sweep.py            # fractions 0, 5, 10, 20%
.venv/bin/python prototype/sequential_replay_sweep.py --fractions 0 0.1 --seed 44 \
  --root artifacts/experiments/sequential/replay_disjoint_seed44   # replicate
~~~

## Reading the results

Compare A retention **at matched B accuracy**. A retention gain bought by lower B learning is not a win. Single seed results are exploratory; replicate on seeds 44/45 and the shared-item condition before claiming an effect.

## Results

See `REPLAY_BASELINE_RESULTS.md` once the sweep has been run.
