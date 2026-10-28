"""Exact PCA rank sweep on cached paired base/fine-tuned calibration changes.

Uses train-only covariance eigendecomposition; no model inference or labels.
Scores on prompt-disjoint calibration validation vectors in raw hidden-state units.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def vector_families(cache, subset, count):
    families = cache.get(f"{subset}_families")
    positions = cache.get(f"{subset}_positions")
    if families is None:
        return ["unknown"] * count
    if positions is not None:
        expanded = [name for name, indices in zip(families, positions) for _ in indices]
    else:
        n = count // len(families) if families and count % len(families) == 0 else None
        if n is None:
            raise ValueError("No unambiguous family-to-vector alignment")
        expanded = [name for name in families for _ in range(n)]
    if len(expanded) != count:
        raise ValueError("Invalid family-to-vector alignment")
    return expanded


def compare_ranks(train, valid, families, ranks=DEFAULT_RANKS):
    """Deterministic symmetric eigensolver, projection learned only from train."""
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1]:
        raise ValueError("Expect matching 2D train/validation delta tensors")
    if not torch.isfinite(train).all() or not torch.isfinite(valid).all():
        raise ValueError("Nonfinite activations")
    if len(families) != len(valid):
        raise ValueError("Family count must match number of validation vectors")
    if min(train.shape) < 2 or len(valid) < 2:
        raise ValueError("Insufficient training or validation data")
    ranks = sorted(set(int(k) for k in ranks))
    max_rank = min(train.shape[0] - 1, train.shape[1])
    if not ranks or any(k <= 0 or k > max_rank for k in ranks):
        raise ValueError(f"Ranks must be within [1, {max_rank}]")

    # Double-precision deterministic covariance eigendecomposition. 896x896 for
    # Qwen2.5-0.5B layer 12, not an SVD of the (4830, 896) activation matrix.
    train = train.to(dtype=torch.float64, device="cpu")
    valid = valid.to(dtype=torch.float64, device="cpu")
    mu = train.mean(dim=0, keepdim=True)
    centered_train = train - mu
    centered_valid = valid - mu
    total = centered_valid.square().sum()
    if total <= 1e-16:
        raise ValueError("Validation delta variance relative to train mean is effectively zero")
    covariance = centered_train.T @ centered_train
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    vectors = eigenvectors.flip(1).contiguous()
    train_energy = eigenvalues.sum().item()
    if train_energy <= 1e-16:
        raise ValueError("Training deltas have negligible centered variance")
    cumulative = (eigenvalues.cumsum(dim=0) / train_energy).tolist()
    k_thresholds = {}
    for target in (.9, .95, .99, .999):
        k_thresholds[str(target)] = next((i + 1 for i, v in enumerate(cumulative) if v >= target), None)
    category_indices = defaultdict(list)
    for i, family in enumerate(families):
        category_indices[str(family)].append(i)
    results = []
    for k in ranks:
        # Orthogonal PCA projection on training principal directions.
        v = vectors[:, :k]
        prediction = mu + (centered_valid @ v) @ v.T
        residual = (valid - prediction).square()
        mse = residual.sum().item()
        row = {
            "rank": k,
            "train_centered_explained_variance": round(cumulative[k - 1], 8),
            "validation_r2_vs_train_mean_delta": round(1 - mse / total.item(), 8),
            "validation_r2_vs_zero_delta": round(
                1 - mse / max(valid.square().sum().item(), 1e-16), 8),
            "by_family": {},
        }
        for family, idx in sorted(category_indices.items()):
            a = valid[idx]
            baseline = mu.expand_as(a)
            denom = (a - baseline).square().sum().item()
            numerator = (a - prediction[idx]).square().sum().item()
            row["by_family"][family] = {
                "n": len(idx),
                "r2_vs_train_mean_delta": round(1 - numerator / denom, 8)
                if denom > 1e-16 else None,
            }
        results.append(row)
    return {
        "dimension": train.shape[1],
        "train_n": train.shape[0],
        "valid_n": valid.shape[0],
        "mean_delta_norm": round(mu.norm().item(), 6),
        "valid_centered_energy_fraction": round(
            total.item() / max(valid.square().sum().item(), 1e-16), 8),
        "ranks": results,
        "rank_to_reach_train_variance_fraction": k_thresholds,
        "limitations": [
            "This is calibration-validation reconstruction, not behavioral regression prediction.",
            "Validation examples are prompt-disjoint but not template-family-disjoint.",
            "PCA rank and SAE active-feature count are not parameter/interpretability-matched.",
            "No test labels or protected-evaluation results are used to fit PCA.",
        ],
    }


def run(cache_path, out, ranks=DEFAULT_RANKS):
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    for subset in ("train", "valid"):
        if data[f"{subset}_base_h"].shape != data[f"{subset}_fine_h"].shape:
            raise ValueError(f"{subset} base/fine tensors do not match")
    train = data["train_fine_h"].float() - data["train_base_h"].float()
    valid = data["valid_fine_h"].float() - data["valid_base_h"].float()
    families = vector_families(data, "valid", len(valid))
    result = compare_ranks(train, valid, families, ranks)
    result["cache"] = str(cache_path)
    result["layer"] = data.get("metadata", {}).get("layer")
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print("PCA rank | validation centered R2 | train explained variance")
    for row in result["ranks"]:
        print(f"{row['rank']:>8} | {row['validation_r2_vs_train_mean_delta']:.6f} "
              f"| {row['train_centered_explained_variance']:.6f}")
    print(f"Saved {dest}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="artifacts/experiments/banking77/sae_v22/paired_cache.pt")
    p.add_argument("--out", default="artifacts/experiments/banking77/sae_v23/pca_rank_sweep.json")
    p.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    args = p.parse_args()
    run(args.cache, args.out, args.ranks)


if __name__ == "__main__":
    main()
