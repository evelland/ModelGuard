# SAE v2.1 — Diagnosing representation changes

Use the existing v2 activation cache and checkpoint. **Do not retrain Qwen or LoRA.**

## Run
From the ModelGuard root, after switching to `feature/sae-v2-banking77` and activating `.venv`:

```bash
python -m unittest discover -s prototype/tests -v
for k in 16 32 64 128; do
  python prototype/sae_v21_analysis.py \
    --cache artifacts/experiments/banking77/sae_v2/cache.pt \
    --checkpoint "artifacts/experiments/banking77/sae_v2/topk${k}/sae_checkpoint.pt" \
    --out "artifacts/experiments/banking77/sae_v21/topk${k}/analysis.json"
done
```

## Metrics
- `delta_r2_vs_zero`: R² of `SAE_decode(base) - SAE_decode(fine)` against the true change, with a **zero-change** predictor as the baseline. This is *not* absolute activation reconstruction R²; a negative value means worse than predicting zero change.
- Cosine similarity and per-example relative error for the reconstructed difference.
- Dev-selected score direction: Does a larger **or smaller** drift predict regression? Direction is selected on the stratified dev split using *within-category* ROC AUC.
- Fixed-direction test performance: within-category pair-weighted ROC AUC is primary; pooled ROC AUC is secondary. Raw and PCA baselines get their own dev-selected directions.
- Stratified paired bootstrap CI for SAE-minus-raw within-category test AUC (exploratory).

## Methodological boundaries
These are **post-hoc exploratory splits from an already-inspected benchmark**; the resulting `test` is not a genuinely untouched test. The benchmark still shares synthetic prompt families between calibration and evaluation, and evaluation captures one final assistant-prefix token while SAE calibration also uses earlier positions. These diagnostics cannot establish causal or semantic features. The selected TopK configurations have also been compared on previous evaluation results.

For a defensible future test, create an independent protected benchmark with novel template families, freeze the model/checkpoint and scoring direction on development data before inspecting that benchmark, and then evaluate once.

Keep PR #17 draft until CI and model-backed experiments are reviewed. The analysis command loads local cached activations and SAE checkpoints; it does not download models or perform any model inference.
