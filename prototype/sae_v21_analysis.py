"""SAE v2.1: diagnose how well TopK features preserve fine-tuning changes.

Uses existing SAE v2 activation cache and a frozen TopK checkpoint. No Qwen
weights, training, or network access required. Exploratory analysis only:
the protected benchmark has already been inspected in prior experiments.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from prototype.sae_v2 import auc, make_sae
except ModuleNotFoundError:
    from sae_v2 import auc, make_sae


def stratified_split(categories, labels, dev_fraction=0.4, seed=42):
    """Deterministic development/test split within category AND outcome."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    if len(categories) != len(labels):
        raise ValueError("Categories and outcome lengths differ")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, (cat, label) in enumerate(zip(categories, labels)):
        groups[(str(cat), bool(label))].append(i)
    dev, test = [], []
    for indices in groups.values():
        rng.shuffle(indices)
        n_dev = max(1, round(len(indices) * dev_fraction)) if len(indices) >= 2 else 0
        n_dev = min(n_dev, len(indices) - 1)
        dev.extend(indices[:n_dev])
        test.extend(indices[n_dev:])
    return sorted(dev), sorted(test)


def within_category_auc(scores, labels, categories, indices):
    """Pair-weighted within-category AUC; excludes single-class groups."""
    groups = defaultdict(list)
    for i in indices:
        groups[str(categories[i])].append(i)
    pair_sum, weighted_sum = 0, 0.0
    breakdown = {}
    for category, ids in sorted(groups.items()):
        positives = sum(bool(labels[i]) for i in ids)
        negatives = len(ids) - positives
        if not positives or not negatives:
            breakdown[category] = {"n": len(ids), "positive": positives, "auc": None}
            continue
        area = auc([scores[i] for i in ids], [labels[i] for i in ids])
        pairs = positives * negatives
        weighted_sum += area * pairs
        pair_sum += pairs
        breakdown[category] = {"n": len(ids), "positive": positives,
                               "auc": round(area, 5), "pairs": pairs}
    return (weighted_sum / pair_sum if pair_sum else None), breakdown


def orient_on_dev(scores, labels, categories, dev):
    area, _ = within_category_auc(scores, labels, categories, dev)
    if area is None:
        raise ValueError("Development split has no comparable within-category classes")
    return (1 if area >= 0.5 else -1), area


def fixed_direction_report(scores, labels, categories, dev, test):
    direction, dev_area = orient_on_dev(scores, labels, categories, dev)
    oriented = [direction * float(score) for score in scores]
    test_within, by_category = within_category_auc(oriented, labels, categories, test)
    test_pooled = auc([oriented[i] for i in test], [labels[i] for i in test])
    return {
        "direction_chosen_on_dev": "greater" if direction == 1 else "smaller",
        "dev_auc_before_orientation": round(dev_area, 5),
        "test_within_category_auc": round(test_within, 5) if test_within is not None else None,
        "test_pooled_auc": round(test_pooled, 5) if test_pooled is not None else None,
        "test_by_category": by_category,
    }


def paired_bootstrap_test_difference(sae, raw, labels, categories, indices, seed=42, rounds=500):
    """Paired resampling within category × label, directions ALREADY frozen."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i in indices:
        groups[(str(categories[i]), bool(labels[i]))].append(i)
    estimates = []
    for _ in range(rounds):
        sampled = []
        for group in groups.values():
            sampled.extend(rng.choices(group, k=len(group)))
        a, _ = within_category_auc(sae, labels, categories, sampled)
        b, _ = within_category_auc(raw, labels, categories, sampled)
        if a is not None and b is not None:
            estimates.append(a - b)
    if not estimates:
        return None
    estimates.sort()
    return [round(estimates[int((len(estimates) - 1) * .025)], 5),
            round(estimates[int((len(estimates) - 1) * .975)], 5)]


def delta_metrics(true_delta, reconstructed_delta):
    """R² relative to ZERO delta; do not confuse with activation R²."""
    import torch
    from torch.nn.functional import cosine_similarity

    if true_delta.shape != reconstructed_delta.shape:
        raise ValueError("Mismatched delta shapes")
    error = true_delta - reconstructed_delta
    numer = error.square().sum().item()
    denom = true_delta.square().sum().item()
    cosine = cosine_similarity(true_delta, reconstructed_delta, dim=-1, eps=1e-8)
    true_norm = true_delta.norm(dim=-1)
    recon_norm = reconstructed_delta.norm(dim=-1)
    relative = error.norm(dim=-1) / true_norm.clamp_min(1e-8)
    valid = true_norm > 1e-8
    return {
        "delta_r2_vs_zero": round(1 - numer / max(denom, 1e-12), 5),
        "true_delta_mean_l2": round(true_norm.mean().item(), 5),
        "reconstructed_delta_mean_l2": round(recon_norm.mean().item(), 5),
        "cosine_mean_nonzero_delta": round(cosine[valid].mean().item(), 5) if valid.any() else None,
        "cosine_median_nonzero_delta": round(cosine[valid].median().item(), 5) if valid.any() else None,
        "relative_error_median_nonzero_delta": round(relative[valid].median().item(), 5) if valid.any() else None,
        "relative_error_mean_nonzero_delta": round(relative[valid].mean().item(), 5) if valid.any() else None,
        "zero_delta_examples": int((~valid).sum().item()),
    }


def analyze(cache_path, checkpoint_path, output_path, seed=42, dev_fraction=.4,
            pca_rank=32, bootstrap_rounds=500):
    import torch

    torch.manual_seed(seed)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base = cache["eval_base_h"].float()
    fine = cache["eval_fine_h"].float()
    labels = [bool(v) for v in cache["eval_regressed"]]
    categories = list(cache["eval_categories"])
    ids = list(cache["eval_ids"])
    if base.shape != fine.shape or base.shape[0] != len(labels) or len(labels) != len(categories):
        raise ValueError("Misaligned evaluation data")
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicated evaluation IDs")
    train = cache["train_h"].float()
    mean, scale = checkpoint["mean"].float(), checkpoint["scale"].float()
    net = make_sae(int(checkpoint["d"]), int(checkpoint["width"]), int(checkpoint["k"]))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    xbase = (base - mean) / scale
    xfine = (fine - mean) / scale
    delta = xfine - xbase
    with torch.inference_mode():
        recon_base, zbase = net(xbase)
        recon_fine, zfine = net(xfine)
    reconstructed_delta = recon_fine - recon_base
    metrics = delta_metrics(delta, reconstructed_delta)
    raw_scores = delta.norm(dim=1).tolist()
    sae_scores = (zfine - zbase).norm(dim=1).tolist()
    # Compare to linear PCA using ONLY SAE calibration activations.
    pca_scores = None
    if pca_rank:
        matrix = (train - mean) / scale
        if len(matrix) > 2000:
            matrix = matrix[torch.randperm(len(matrix))[:2000]]
        rank = min(pca_rank, matrix.shape[0], matrix.shape[1])
        if rank < 1:
            raise ValueError("PCA rank must be positive")
        _, _, components = torch.pca_lowrank(matrix, q=rank, center=False, niter=2)
        pca_scores = (delta @ components).norm(dim=1).tolist()
    dev, test = stratified_split(categories, labels, dev_fraction, seed)
    scores = {"sae_latent_drift": sae_scores, "raw_activation_drift": raw_scores}
    if pca_scores is not None:
        scores["pca_drift"] = pca_scores
    reports = {
        name: fixed_direction_report(values, labels, categories, dev, test)
        for name, values in scores.items()
    }
    signs = {key: 1 if value["direction_chosen_on_dev"] == "greater" else -1
             for key, value in reports.items()}
    ci = paired_bootstrap_test_difference(
        [signs["sae_latent_drift"] * x for x in sae_scores],
        [signs["raw_activation_drift"] * x for x in raw_scores],
        labels, categories, test, seed, bootstrap_rounds)
    test_count_by_category = defaultdict(lambda: {"n": 0, "regressed": 0})
    for i in test:
        test_count_by_category[str(categories[i])]["n"] += 1
        test_count_by_category[str(categories[i])]["regressed"] += int(labels[i])
    result = {
        "experiment": "sae_v2.1_exploratory_cached_diagnostics",
        "inputs": {"cache": str(cache_path), "checkpoint": str(checkpoint_path),
                   "sae_k": int(checkpoint["k"]), "layer": cache.get("metadata", {}).get("layer")},
        "split": {"seed": seed, "dev_fraction": dev_fraction,
                  "dev_n": len(dev), "test_n": len(test),
                  "dev_ids": [ids[i] for i in dev], "test_ids": [ids[i] for i in test],
                  "test_by_category": dict(sorted(test_count_by_category.items()))},
        "representation_change": metrics,
        "drift_detection": reports,
        "paired_test_95pct_bootstrap_ci_sae_minus_raw_conditional_auc": ci,
        "limitations": [
            "The benchmark and hyperparameters were examined before this new dev/test split: exploratory, NOT untouched-test validation.",
            "Direction calibration uses development labels; no test labels are used to choose direction.",
            "Within-category AUC removes cross-category pairs but cannot control for template-family confounds.",
            "Category-by-label bootstrap ignores shared-template dependencies and SAE seed variability.",
            "Final assistant-prefix token is used for eval; previous calibration used mixed positions.",
            "A faithful activation delta is not a causal explanation or proof of preventing forgetting.",
        ],
    }
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {dest}")
    print(f"Delta R2 vs zero: {metrics['delta_r2_vs_zero']:.4f}")
    for name, report in reports.items():
        print(f"{name}: dev direction={report['direction_chosen_on_dev']}; "
              f"test within-category AUC={report['test_within_category_auc']}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="artifacts/experiments/banking77/sae_v2/cache.pt")
    p.add_argument("--checkpoint", default="artifacts/experiments/banking77/sae_v2/topk32/sae_checkpoint.pt")
    p.add_argument("--out", default="artifacts/experiments/banking77/sae_v21/topk32/analysis.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=.4)
    p.add_argument("--pca-rank", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=500)
    args = p.parse_args()
    if not 0 < args.dev_fraction < 1 or args.pca_rank < 0 or args.bootstrap_rounds < 1:
        p.error("dev-fraction must be in (0,1), pca-rank >= 0, bootstrap-rounds >= 1")
    analyze(args.cache, args.checkpoint, args.out, args.seed,
            args.dev_fraction, args.pca_rank, args.bootstrap_rounds)


if __name__ == "__main__":
    main()
