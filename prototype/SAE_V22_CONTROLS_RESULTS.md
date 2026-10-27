# SAE v2.2 mean-delta and PCA falsification controls — 27 October 2026

## Provenance

These are the six local `controls.json` outputs supplied by the team after running `prototype/sae_v22_controls.py` on the paired calibration cache (1,207 validation vectors), for a fixed Qwen2.5-0.5B-Instruct BANKING77 LoRA, layer 12 and SAE seed 42. Checkpoints and raw activations were not uploaded here; numerical observations below come from the supplied summary files. All six use the same mean-delta and rank-32 PCA control.

## Key fact: 75% of delta energy is the common train-set mean shift

The simple predictor `Δhat(x) = mean_train(Δ)` achieves **zero-reference delta R² = 0.75007**. Therefore **~24.993%** of validation delta energy remains after removing the estimated mean shift. The reported mean-delta norm is **4.39456**, versus average true delta norm **5.04147**. Common direction is substantial; zero-reference R² alone overstates how much prompt-dependent information a model reconstructs.

## Centered delta reconstruction (R² relative to train-set mean-delta baseline)

| Model | k or PCA rank | Validation R² vs train mean delta | R² vs zero delta |
|---|---:|---:|---:|
| Base-only TopK SAE | 32 | -2.12386 | 0.21926 |
| Base-only TopK SAE | 128 | -1.77498 | 0.30645 |
| Pooled base+fine SAE | 32 | 0.97367 | 0.99342 |
| Pooled base+fine SAE | 128 | 0.98201 | 0.99550 |
| Delta-trained SAE | 32 | 0.98450 | 0.99613 |
| Delta-trained SAE | 128 | **0.98984** | **0.99746** |
| **Training-delta PCA** | **32** | **0.98879** | **0.99720** |
| Train mean-delta (constant) | — | 0.00000 | 0.75007 |

**Interpretation:** Even after removing the common mean shift, the delta-trained SAE at k128 reconstructs most observed changes. However **rank-32 PCA achieves virtually the same fidelity**: R² = .98879 versus delta-trained k128 = .98984, only **+0.00105 R²**. The k32 delta-trained SAE, and both pooled variants, score below PCA. PCA's 32 dense coordinates are not identical in capacity/interpretability to 128 positive features selected from an SAE dictionary of 3,584 features; parameter-count/representation-cost controls are still needed.

The base-only SAE scores **worse than the constant mean-delta predictor** under this more informative metric. Good base activation reconstruction is not change reconstruction.

## Implications and limits

- There is a **large common shift** and a **highly compressible prompt-dependent component** in layer-12 differences for this one LoRA checkpoint.
- Our strongest SAE does **not** establish a meaningful reconstruction advantage beyond PCA, let alone better detection of regressions.
- No provided control tests downstream behavior, feature semantics, or causality. Earlier v2.2 regression AUCs are exploratory on previously inspected protected cases, and show no stable superiority to raw drift.
- Only one checkpoint, model layer, SAE training seed, and synthetic/templated protected-family context are represented. The validation examples are exact-prompt held-out, not necessarily template-family held-out.
- `torch.pca_lowrank` is randomized; a +0.00105 R² difference may not be meaningful without exact-SVD or repeated-seed comparisons.
- A low-dimensional linear delta may arise from the fine-tuning adapter architecture. Avoid making claims about general model changes until multiple LoRA checkpoints, layers, or fully fine-tuned models are evaluated.

## Next priority — determine intrinsic dimension, then move toward behavior

1. **No model rerun needed:** using the existing matched activation cache, calculate *exact* PCA and test ranks 1, 2, 4, 8, 16, 32, 64, 128. Report centered-delta R² and per-family reconstruction (banking vs protected), and how many components explain 90/95/99% of train-set centered delta variance. This tests whether our TopK SAE is simply approximating a small linear subspace.
2. **Shift evaluation priority** from reconstructing `Δh` to explaining behavioral failures. A useful representation should beat straightforward **raw/PCA** baselines at held-out regression prediction and survive prespecified split/score orientation. Direction of drift was inverse for all earlier detectors.
3. Build a **new protected benchmark with distinct template families**, freeze model, SAE/PCA, evaluation protocol and direction selection before scoring. Repeat with at least three independent SAE seeds and several fine-tuning checkpoints; within-category inference should account for template grouping.
4. If stable predictive features emerge, conduct feature ablation/activation patching and matched random-direction controls to evaluate causality, before proposing SAE preservation loss.

**Decision:** Keep draft PR #17 as a research/experimental PR; do not claim the approach prevents forgetting or meaningfully outperforms PCA.
