# SAE v2.2 — Paired calibration and three SAE objectives

**Goal.** Compare whether a base-only SAE, a base-plus-fine-tuned pooled SAE, or a delta-trained SAE better preserves fine-tuning-induced activation changes. All three use exactly the same frozen Qwen LoRA, layer, TopK, dictionary expansion, and matched calibration prompts; no Qwen retraining is involved.

## Run on Mac

From repository root:

```bash
git checkout feature/sae-v2-banking77
git pull
source .venv/bin/activate
python -m unittest discover -s prototype/tests -v
python prototype/preflight_sae_v2.py
python prototype/sae_v22.py collect --device mps --layer 12 \
  --bank-prompts 2700 --protected-prompts 4000 \
  --positions-per-prompt 1
for scheme in base pooled delta; do
  for k in 16 32 64 128; do
    python prototype/sae_v22.py fit --device mps \
      --scheme "$scheme" --k "$k" --expansion 4 \
      --eval-cache artifacts/experiments/banking77/sae_v2/cache.pt
  done
done
```

Results: `artifacts/experiments/banking77/sae_v22/{scheme}_topk{k}_seed42/summary.json`. The paired activation cache is saved to `sae_v22/paired_cache.pt`. Start with `k=32` and `k=128` if running the full sweep is impractical.

## What is compared

- **Base:** SAE reconstructs base activation vectors alone, and estimated fine-tuning delta is the difference of the two reconstructed outputs.
- **Pooled:** SAE trains on matched base and fine-tuned activation vectors, reconstructing both before differencing. Pooled collection never uses protected evaluation labels.
- **Delta:** SAE trains directly on the paired `h_fine - h_base` difference, in its own normalized coordinate system; it estimates delta directly. Its features are *not* interchangeable with base/pooled features.

Primary exploratory metric is held-out **calibration validation** delta R² *against a zero-delta predictor*, measured in the same **raw hidden-state units** for all three methods. Also report cosine and relative error. A high R² does not establish causal features or successful forgetting prevention.

**Important design decisions:** Calibration uses `positions-per-prompt 1` by default for a consistent assistant-prefix token, rather than mixing token positions as v2 did; every base/fine pair uses identical token IDs and indices. Normalized exact prompt hashes are removed from the evaluated report and from the opposite calibration split. The BANKING77 training subset may overlap the model's original fine-tuning inputs, and generated protected prompts share template families with the old evaluated benchmark. Therefore *the evaluation is still exploratory*. Do not claim a pristine template-disjoint holdout.

## Research checks before concluding

The optional `--eval-cache` scores the previously examined protected examples and reports a dev-oriented within-category ROC AUC against raw activations. This is an exploratory diagnostic only. The delta-trained latent score is a norm of its direct delta encoding; its scale and semantics differ from the pooled SAE's difference-of-encodings.

For causal interpretation and credible regression generalization, later experiments need a newly built template-family holdout, multiple model and SAE seeds, and matched baselines (PCA, raw activations, replay). Check that improved delta fidelity is not just memorizing specific LoRA effects from calibration.

GitHub CI runs only offline tests and source syntax validation. The code cannot execute full Qwen inference in CI.


## Crucial post-hoc control after v2.2 results

The extraordinarily high zero-reference delta R² may be explained by a nearly constant model-wide shift. Evaluate each saved checkpoint against the train-set mean-delta and a calibration-trained PCA projection:

```bash
for scheme in base pooled delta; do
  for k in 32 128; do
    folder="artifacts/experiments/banking77/sae_v22/${scheme}_topk${k}_seed42"
    python prototype/sae_v22_controls.py \
      --checkpoint "$folder/sae_checkpoint.pt" \
      --out "$folder/controls.json" --rank 32
  done
done
```

The control reports both **R² vs zero** and **R² vs the train-set mean delta** (which is zero for the mean predictor). If the mean predictor already explains nearly all change energy, even a 0.997 zero-reference R² does not establish prompt-specific information. PCA is trained using **only paired training deltas**. These diagnostics do not claim causal feature relevance.
