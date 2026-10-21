# SAE pilot — base vs fine-tuned Qwen2.5

**Status:** experimental implementation, not yet run on your Mac. This is a *diagnostic*, not SAE-preservation training or validated mechanistic explanation. SAEs approximate hidden activations with an overcomplete sparse code; the discovered axes are not necessarily human-interpretable.

## Command (from ModelGuard root)

    source .venv/bin/activate
    git pull
    python prototype/sae_probe.py --device mps \
      --adapter artifacts/banking77-2700-lora \
      --report artifacts/experiments/banking77/report.json \
      --calibration 600 --eval-limit 80 --layer 12 \
      --epochs 45

    open artifacts/experiments/banking77/sae_probe/summary.html

For additional coverage, repeat with --eval-limit 0 (all original base-correct protected examples). The initial run chooses up to 40 regressed and 40 retained prompts, balancing groups only for comparison. It will download no new model checkpoint if Qwen is already cached.

## Protocol

1. Generate a distinct calibration set of 600 protected-style prompts, using another random seed, filtering prompt overlap with the selected held-out prompts.
2. Capture base-model hidden activations at layer index 12, for the final token of the assistant prefix.
3. Train an overcomplete ReLU SAE (2x hidden size) using reconstruction MSE + L1 sparsity penalty.
4. Freeze the SAE; encode hidden activations from both the base model and the LoRA adapter on the SAME held-out prompts.
5. Compare feature changes between base-correct / fine-wrong ("regressed") and base-correct / fine-correct ("retained") prompts, and compare to an equally simple raw-activation-distance predictor.
6. Save summary.html, summary.json, and the locally stored SAE weights in artifacts/ (ignored by Git).

## Read the results correctly

- **Held-out base reconstruction R²:** Does the SAE reproduce the original activation vectors? Negative or low R² means feature comparisons are especially untrustworthy.
- **Sparsity:** Mean number of positive features. If most features activate, the SAE is not meaningfully sparse; tune L1 / report failure.
- **AUC:** Descriptive ranking of which examples regress using total SAE feature drift, compared to raw activation drift. In-sample, not independently validated.
- **Top candidate features:** Features whose absolute changes differ most between regressed and retained prompts. Their highest-activation calibration prompts are *illustrations only*, not semantic definitions.
- **No causal attribution:** To substantiate explanations, validate feature labels across contexts, compare stronger feature-extraction baselines, and intervene (ablate or patch) in model activations.

## Important caveats

The calibration sample uses template-generated arithmetic/formatting/JSON prompts and only a single token position. This SAE is a tiny approximation to methods typically trained on enormous corpora. Source vectors are from residual hidden states, not a specialized crosscoder. The base SAE dictionary is held fixed after LoRA training. Regression labels are used for post-hoc ranking, not SAE fitting; use a separate validation set before reporting predictive performance of selected features.

For the midterm, if the command completes, describe this as "our first SAE activation-diff pilot." Put R², sparsity and raw-vs-SAE AUC on one slide and avoid claims of interpreting named concepts or fixing forgetting.
