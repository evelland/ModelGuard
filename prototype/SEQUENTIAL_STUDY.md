# Controlled Sequential Learning: Task A → Task B

This is a new focused experiment responding to advisor feedback. Unlike BANKING77, the task explicitly distinguishes acquiring a new lookup mapping from retaining an earlier one.

## Design

Task A and Task B each map (MODE, ITEM) to a code K2/R7/T4/V9. Stage 1 learns A; stage 2 continues training **the same LoRA adapter** on B only. We evaluate both tasks at both checkpoints. The disjoint condition uses different items; the shared condition reuses items with different mapping rules but an explicit mode label. Mapping codes, output format, and general prompt structure are held constant across stages.

Data generation is seeded and refuses overwrites. Training/dev/test have distinct prompt templates; each split reuses the same item identities because the task is remembering a learned arbitrary mapping, not generalization to unseen items. A separate Task A reference set is reserved for later replay and regularization.

## Commands on Mac

From the repository root, install the dependencies from prototype/requirements.txt in a virtual environment and run:

~~~bash
git fetch origin
git checkout experiment/sequential-mappings
git pull

python -m unittest discover -s prototype/tests -v
python prototype/sequential_data.py --condition disjoint --seed 42

python prototype/sequential_train.py --stage A --device mps \
  --train-data artifacts/experiments/sequential/disjoint/train_A.jsonl \
  --epochs 4 --learning-rate 0.0002 \
  --out artifacts/experiments/sequential/disjoint/adapter_A

python prototype/sequential_train.py --stage B --device mps \
  --train-data artifacts/experiments/sequential/disjoint/train_B.jsonl \
  --prior-adapter artifacts/experiments/sequential/disjoint/adapter_A \
  --epochs 2 --learning-rate 0.0002 \
  --out artifacts/experiments/sequential/disjoint/adapter_AB

python prototype/evaluate_pair.py --device mps \
  --adapter artifacts/experiments/sequential/disjoint/adapter_A \
  --cases artifacts/experiments/sequential/disjoint/eval_combined_dev.jsonl \
  --output artifacts/experiments/sequential/disjoint/dev_after_A.jsonl

python prototype/evaluate_pair.py --device mps \
  --adapter artifacts/experiments/sequential/disjoint/adapter_AB \
  --cases artifacts/experiments/sequential/disjoint/eval_combined_dev.jsonl \
  --output artifacts/experiments/sequential/disjoint/dev_after_B.jsonl

python prototype/sequential_score.py \
  --after-a artifacts/experiments/sequential/disjoint/dev_after_A.jsonl \
  --after-b artifacts/experiments/sequential/disjoint/dev_after_B.jsonl \
  --out artifacts/experiments/sequential/disjoint/dev_comparison.json
~~~

Tune training epochs using the DEV results. Only once the configuration is frozen, rerun the paired evaluation commands with eval_combined_test.jsonl and unique test output files, then score those. Never overwrite a frozen test or tune using test labels.

To run the shared condition, separately generate using --condition shared, and replace the disjoint path segment in every command.

## Scientific checks

- First make sure stage A achieves useful held-out accuracy; otherwise "forgetting" is uninterpretable.
- Make sure Task B improves from stage-A to stage-AB checkpoint. A drop in Task A at no B learning is not a useful tradeoff.
- Primary forgetting fraction = (A examples correct after A but incorrect after AB) / (A examples correct after A), with both aggregate A accuracies also reported.
- Repeat with mapping/training seeds 42, 43, 44 in separately named folders.
- If no forgetting occurs, vary B stage length, learning rate, and shared-item condition on DEV, not the final TEST.
- Next controls: less B training, small replay mixture, raw-activation matching, SAE selected-feature matching, and random-feature matching. Compare A retention at equal Task B accuracy, not merely equal epochs.
- Optional circuit intervention: patch selected layer activations from stage A into stage AB on the same A prompt and measure token logit/behavior recovery. This would test causality more directly than latent-norm drift.
- The explicit task contract makes the behaviors logically compatible, not necessarily mechanistically independent.
- No Qwen model-backed sequential results have been run yet; the prior SAE experiments remain separately in draft PR #17.

## One-command fresh rerun (recommended)

If earlier training or evaluation output is missing, use the dedicated runner
instead of piecing together partial checkpoints. It loads pretrained
Qwen2.5-0.5B-Instruct afresh, initializes a new LoRA adapter for Task A,
then continues that same adapter on B. A different mapping seed (43) and
separate output folder avoid mixing runs. It checks that both development
evaluations are complete before scoring.

From the repository root:

~~~bash
git fetch origin
git checkout experiment/sequential-mappings
git pull
source .venv/bin/activate
python -m unittest prototype.tests.test_sequential_run -v
python prototype/sequential_run.py --dry-run
python prototype/sequential_run.py
~~~

Report location:

~~~text
artifacts/experiments/sequential/disjoint_fresh_seed43/dev_comparison.json
~~~

To reveal in Finder:

~~~bash
open -R artifacts/experiments/sequential/disjoint_fresh_seed43/dev_comparison.json
~~~

Do not delete earlier checkpoints or overwrite any existing run folder.
If that folder already exists, use another, e.g.:

~~~bash
python prototype/sequential_run.py --out artifacts/experiments/sequential/disjoint_fresh_seed43_b
~~~

This runs DEV only and does not inspect the reserved TEST split. If Task A
or Task B acquisition is weak, adjust training on another dev run before
freezing settings for the final test. The full Qwen run executes locally;
GitHub Actions tests offline orchestration and data generation only.
