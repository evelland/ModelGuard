# Sequential Task A → B: first clean forgetting result

**Source:** user-uploaded `familiar_forgetting_comparison.json`, produced using a new-identifier / familiar-training-template probe after continuing the same LoRA adapter from Task A (8 epochs) to Task B (2 epochs). Local run paths in the team workflow indicate the disjoint-item seed-43 dataset. The uploaded comparison does **not** contain checkpoint hashes, training metadata, or a seed field, so these provenance details should be verified against the original artifacts before publication.

| Task | Correct after A8 | Correct after A8→B2 | Net |
|---|---:|---:|---:|
| A: mapping retention | 36/36 (100%) | 31/36 (86.11%) | −5 (−13.89 pp) |
| B: new mapping | 12/36 (33.33%) | 36/36 (100%) | +24 (+66.67 pp) |

**New A pass→fail transitions: 5/36 (13.89%)**; 31 previously correct stayed correct; no A wrong→right gains. All five new errors returned `R7`, replacing expected codes for `cedar` (`T4`→`R7`, 3/3 prompts), `pine` (`T4`→`R7`, 1/3), and `alder` (`K2`→`R7`, 1/3). A complete item-level error distribution is not available from this summary; inspect prediction JSONLs for all post-B responses before attributing a general code bias.

**Interpretation:** Unlike the earlier held-out-template development evaluation (A8 = 27/48), the fresh-nonce, familiar-template probe establishes an acquired ability (A8 = 36/36) that deteriorates after B learning, while B reaches 36/36. This is controlled **behavioral regression in a restricted template setting**, not proof of mechanistic overwriting, broad capability forgetting, or an SAE advantage.

### Follow-up controls (development first)

1. **Verify error mechanism:** inspect full post-B A/B predictions by item, expected label, predicted label, and template, including frequency of `R7`. Consider first-token logit margins if pursuing mechanistic claims.
2. **Run a replay baseline at identical stage-B example/optimizer budget:** mix 10% previously seen A examples into B training, starting from the *same saved A8 adapter*. Compare A retention at achieved B accuracy. Mix only training data, never A/B test/dev or reference prompts.
3. **Strength-sweep** replay (0, 5, 10, 20%) and B duration/LR on DEV; compare a tradeoff frontier, not just two endpoints. Record seeds and data hashes.
4. **Replicate** on other mapping seeds and shared-item mode after establishing the first control. New seed runs require re-establishing A competence.
5. **After controls:** compare raw activation matching, random SAE features, selected SAE preservation, and update-scaling / grafting in the same experiment, with matched B accuracy and multiple seeds.

### Reporting caveats

- These 36 examples share known item identities and familiar training instruction templates; nonces are unseen, and all 36 were correct before B. The earlier novel-template DEV set shows that prompt paraphrase generalization is weaker.
- Regressions cluster in `cedar`, `pine`, `alder`; examples are correlated by item. A naive 36-case independent-sample significance calculation would exaggerate certainty.
- This is one mapping seed / one training run; generalization to other seeds, target datasets and adapters is untested.
- No model-backed SAE preservation, replay, or raw-activation intervention has been run yet.
