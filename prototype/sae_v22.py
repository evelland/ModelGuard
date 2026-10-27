"""SAE v2.2: matched base/LoRA calibration and three comparable TopK fits.

Usage:
  python prototype/sae_v22.py collect [--device mps]
  python prototype/sae_v22.py fit --scheme pooled --k 32 [--device mps]

Only the collection stage requires Qwen / LoRA inference. All evaluation on the
previously inspected protected set is exploratory, not a confirmatory test.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    from prototype.prepare_banking77 import normalized, protected_cases
    from prototype.sae_v2 import make_sae, read_jsonl, select_token_positions
    from prototype.sae_v21_analysis import delta_metrics, fixed_direction_report, stratified_split
except ModuleNotFoundError:
    from prepare_banking77 import normalized, protected_cases
    from sae_v2 import make_sae, read_jsonl, select_token_positions
    from sae_v21_analysis import delta_metrics, fixed_direction_report, stratified_split


def fingerprint(text):
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def make_calibration_splits(train_rows, eval_rows, bank_count=2700,
                            protected_count=4000, seed=20261027, val_fraction=0.2):
    """Exclude all evaluated prompt TEXT, deduplicate, split by exact prompt.

    Synthetic family overlap is NOT eliminated; use a new benchmark to validate.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    if bank_count < 0 or protected_count < 0:
        raise ValueError("Calibration sizes must be nonnegative")
    rng = random.Random(seed)
    excluded = {fingerprint(r["prompt"]) for r in eval_rows}
    seen = set(excluded)
    groups = defaultdict(list)
    banks = [r["prompt"] for r in train_rows]
    rng.shuffle(banks)
    protected = protected_cases(max(protected_count, 4), seed + 401)
    prot = [(r["prompt"], r["category"]) for r in protected]
    rng.shuffle(prot)
    candidates = [("banking", text) for text in banks[:bank_count]]
    candidates += [(kind, text) for text, kind in prot[:protected_count]]
    for family, text in candidates:
        key = fingerprint(text)
        if key not in seen:
            seen.add(key)
            groups[family].append((text, key, family))
    train, valid = [], []
    for family in sorted(groups):
        rows = groups[family]
        rng.shuffle(rows)
        n_valid = min(len(rows) - 1, max(1, round(len(rows) * val_fraction))) if len(rows) >= 2 else 0
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])
    rng.shuffle(train)
    rng.shuffle(valid)
    if not train or not valid:
        raise ValueError("Insufficient distinct calibration prompts")
    if {r[1] for r in train} & {r[1] for r in valid}:
        raise RuntimeError("Training/validation prompt leakage")
    if {r[1] for r in train + valid} & excluded:
        raise RuntimeError("Calibration/evaluation prompt leakage")
    return train, valid


def resolve_device(choice):
    import torch
    actual = choice if choice != "auto" else (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    if actual == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if actual == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    return actual


def collect(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    eval_rows = json.loads(Path(args.report).read_text())["rows"]
    train, valid = make_calibration_splits(
        read_jsonl(args.train_data), eval_rows, args.bank_prompts,
        args.protected_prompts, args.seed, args.validation_fraction)
    device = resolve_device(args.device)
    print(f"Matched base/LoRA collection: {len(train)} train + {len(valid)} valid prompts; "
          f"positions={args.positions_per_prompt}; device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager")
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    if not 0 <= args.layer <= model.config.num_hidden_layers:
        raise ValueError("Invalid hidden-state index")
    special = set(tokenizer.all_special_ids)

    @torch.inference_mode()
    def paired(rows, tag):
        base_chunks, fine_chunks, hashes, families, token_positions = [], [], [], [], []
        for i, (text, key, family) in enumerate(rows, 1):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True)
            if len(tokens) > args.max_length:
                raise ValueError(f"Prompt exceeds max_length={args.max_length}: {key}")
            positions = select_token_positions(
                tokens, args.positions_per_prompt, rng, special)
            if len(positions) != args.positions_per_prompt:
                raise ValueError("Prompt has fewer usable positions than requested")
            inputs = torch.tensor([tokens], device=device)
            with model.disable_adapter():
                result = model(input_ids=inputs, use_cache=False, output_hidden_states=True)
                base_h = result.hidden_states[args.layer][0, positions].float().cpu()
                del result
            result = model(input_ids=inputs, use_cache=False, output_hidden_states=True)
            fine_h = result.hidden_states[args.layer][0, positions].float().cpu()
            del result
            base_chunks.append(base_h)
            fine_chunks.append(fine_h)
            hashes.append(key)
            families.append(family)
            token_positions.append(positions)
            if i % 250 == 0 or i == len(rows):
                print(f"{tag}: {i}/{len(rows)}", flush=True)
        return (torch.cat(base_chunks), torch.cat(fine_chunks),
                hashes, families, token_positions)

    tr_b, tr_f, tr_keys, tr_families, tr_positions = paired(train, "train")
    va_b, va_f, va_keys, va_families, va_positions = paired(valid, "valid")
    # Single saved cache: all paired activations share the exact same text/token
    # positions and sample order; NEVER concatenate independently sampled vectors.
    output = {
        "train_base_h": tr_b, "train_fine_h": tr_f,
        "valid_base_h": va_b, "valid_fine_h": va_f,
        "train_prompt_hashes": tr_keys, "valid_prompt_hashes": va_keys,
        "train_families": tr_families, "valid_families": va_families,
        "train_positions": tr_positions, "valid_positions": va_positions,
        "metadata": {
            "model": args.model, "adapter": args.adapter, "layer": args.layer,
            "seed": args.seed, "positions_per_prompt": args.positions_per_prompt,
            "train_prompt_count": len(train), "valid_prompt_count": len(valid),
            "report": args.report, "excluded_eval_count": len(eval_rows),
            "split": "exact normalized prompt disjoint, NOT template-family holdout",
        },
    }
    validate_paired_cache(output)
    dest = Path(args.cache)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, dest)
    print(f"Saved {dest} ({tr_b.shape[0]} train and {va_b.shape[0]} validation vector pairs)")


def validate_paired_cache(cache):
    """Validate alignment, shape and prompt-disjoint split, including old caches."""
    import torch
    for prefix in ("train", "valid"):
        a, b = cache[f"{prefix}_base_h"], cache[f"{prefix}_fine_h"]
        if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 2 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Invalid or misaligned {prefix} activations")
    if cache["train_base_h"].shape[1] != cache["valid_base_h"].shape[1]:
        raise ValueError("Training/validation hidden dimensions differ")
    tr, va = set(cache.get("train_prompt_hashes", [])), set(cache.get("valid_prompt_hashes", []))
    if tr and va and tr & va:
        raise ValueError("Overlapping calibration prompt hashes")
    return True


def scheme_vectors(cache, scheme):
    if scheme not in ("base", "pooled", "delta"):
        raise ValueError("Unknown SAE scheme")
    train_b, train_f = cache["train_base_h"].float(), cache["train_fine_h"].float()
    valid_b, valid_f = cache["valid_base_h"].float(), cache["valid_fine_h"].float()
    if scheme == "base":
        return train_b, valid_b
    if scheme == "pooled":
        return __import__("torch").cat([train_b, train_f]), __import__("torch").cat([valid_b, valid_f])
    return train_f - train_b, valid_f - valid_b


def fit(args):
    import torch
    from torch.nn import functional as F

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    validate_paired_cache(cache)
    training, validation = scheme_vectors(cache, args.scheme)
    d = training.shape[1]
    width = int(d * args.expansion)
    if not 1 <= args.k <= width:
        raise ValueError("TopK cannot exceed dictionary width")
    mean = training.mean(dim=0, keepdim=True)
    scale = (training - mean).square().mean().sqrt().clamp_min(1e-6)
    tr, va = ((training - mean) / scale).to(device), ((validation - mean) / scale).to(device)
    model = make_sae(d, width, args.k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state, stale = None, 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tr), device=device)
        for batch in order.split(args.batch_size):
            pred, _ = model(tr[batch])
            loss = F.mse_loss(pred, tr[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.normalize_decoder()
        model.eval()
        with torch.inference_mode():
            val_loss = sum(F.mse_loss(model(group)[0], group, reduction="sum").item()
                           for group in va.split(args.batch_size)) / va.numel()
        if val_loss < best - 1e-5:
            best = val_loss
            stale = 0
            best_state = {name: value.cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def model_outputs(raw):
        with torch.inference_mode():
            x = ((raw - mean) / scale).to(device)
            rec, features = [], []
            for group in x.split(args.batch_size):
                r, z = model(group)
                rec.append((r * scale.to(device) + mean.to(device)).cpu())
                features.append(z.cpu())
            return torch.cat(rec), torch.cat(features)

    vb, vf = cache["valid_base_h"].float(), cache["valid_fine_h"].float()
    if args.scheme == "delta":
        valid_delta = vf - vb
        reconstructed, _ = model_outputs(valid_delta)
    else:
        reconstructed = model_outputs(vf)[0] - model_outputs(vb)[0]
    # For comparability, the R² reference and distances are in original
    # hidden-state units, not per-scheme normalized latent coordinates.
    from prototype.sae_v21_analysis import delta_metrics as _metric
    delta_valid = _metric(vf - vb, reconstructed)

    output = {
        "scheme": args.scheme, "k": args.k, "width": width,
        "seed": args.seed, "best_validation_mse_normalized": best,
        "paired_validation_delta": delta_valid,
        "train_vectors": len(training), "valid_vectors": len(validation),
        "metadata": cache["metadata"],
        "note": "Validation delta is from calibration prompts only; old protected benchmark, if used, remains exploratory.",
    }

    if args.eval_cache:
        old = torch.load(args.eval_cache, map_location="cpu", weights_only=True)
        if old["metadata"]["layer"] != cache["metadata"]["layer"] or old["metadata"]["model"] != cache["metadata"]["model"]:
            raise ValueError("Evaluation cache model/layer does not match")
        eb, ef = old["eval_base_h"].float(), old["eval_fine_h"].float()
        if eb.shape != ef.shape or eb.shape[1] != d:
            raise ValueError("Unmatched protected evaluation shapes")
        if args.scheme == "delta":
            r_delta, z = model_outputs(ef - eb)
            scores = z.norm(dim=1).tolist()
        else:
            re_b, zb = model_outputs(eb)
            re_f, zf = model_outputs(ef)
            r_delta = re_f - re_b
            scores = (zf - zb).norm(dim=1).tolist()
        actual = ef - eb
        labels = list(old["eval_regressed"])
        cats = list(old["eval_categories"])
        ids = list(old["eval_ids"])
        if len(labels) != len(scores) or len(labels) != len(ids):
            raise ValueError("Evaluation scores and IDs misaligned")
        dev, test = stratified_split(cats, labels, .4, 42)
        from prototype.sae_v21_analysis import fixed_direction_report as _report
        output["exploratory_protected"] = {
            "n": len(scores), "evaluation_cache": args.eval_cache,
            "delta": _metric(actual, r_delta),
            "latent_score": _report(scores, labels, cats, dev, test),
            "raw_score": _report(actual.norm(dim=1).tolist(), labels, cats, dev, test),
            "previously_inspected": True,
        }

    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    torch.save({"state_dict": best_state, "mean": mean, "scale": scale,
                "d": d, "width": width, "k": args.k, "scheme": args.scheme}, folder / "sae_checkpoint.pt")
    print(f"{args.scheme}, k={args.k}: validation delta R²={delta_valid['delta_r2_vs_zero']:.4f}; output {folder}")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("collect")
    a.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    a.add_argument("--adapter", default="artifacts/banking77-2700-lora")
    a.add_argument("--train-data", default="artifacts/experiments/banking77/train.jsonl")
    a.add_argument("--report", default="artifacts/experiments/banking77/report.json")
    a.add_argument("--cache", default="artifacts/experiments/banking77/sae_v22/paired_cache.pt")
    a.add_argument("--layer", type=int, default=12)
    a.add_argument("--bank-prompts", type=int, default=2700)
    a.add_argument("--protected-prompts", type=int, default=4000)
    a.add_argument("--positions-per-prompt", type=int, default=1,
                   help="Default 1 matches the protected evaluation assistant-prefix position")
    a.add_argument("--validation-fraction", type=float, default=.2)
    a.add_argument("--max-length", type=int, default=256)
    a.add_argument("--seed", type=int, default=20261027)
    a.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    b = sub.add_parser("fit")
    b.add_argument("--cache", default="artifacts/experiments/banking77/sae_v22/paired_cache.pt")
    b.add_argument("--scheme", choices=("base", "pooled", "delta"), required=True)
    b.add_argument("--out", default=None)
    b.add_argument("--eval-cache", default=None, help="Optional old v2 evaluation cache (exploratory)")
    b.add_argument("--k", type=int, default=32)
    b.add_argument("--expansion", type=int, default=4)
    b.add_argument("--epochs", type=int, default=40)
    b.add_argument("--patience", type=int, default=8)
    b.add_argument("--batch-size", type=int, default=256)
    b.add_argument("--lr", type=float, default=.001)
    b.add_argument("--weight-decay", type=float, default=.0001)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = p.parse_args()
    if args.action == "collect":
        if not (0 < args.validation_fraction < 1) or args.positions_per_prompt < 1 or args.max_length < 1:
            p.error("Invalid collection settings")
        collect(args)
    else:
        if min(args.k, args.expansion, args.epochs, args.patience, args.batch_size) < 1:
            p.error("k, expansion, epochs, patience and batch-size must be positive")
        if args.out is None:
            args.out = f"artifacts/experiments/banking77/sae_v22/{args.scheme}_topk{args.k}_seed{args.seed}"
        fit(args)


if __name__ == "__main__":
    main()
