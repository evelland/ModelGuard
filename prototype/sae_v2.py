"""BANKING77 SAE v2: cache Qwen activations, train TopK SAE, evaluate drift.

Usage from repository root: python prototype/sae_v2.py collect | fit
This is a diagnostic, not a causal intervention or forgetting prevention.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def get_rows(report):
    rows = json.loads(Path(report).read_text())["rows"]
    result = [r for r in rows if r.get("group") == "protected"
              and r.get("base_pass") is True and isinstance(r.get("fine_pass"), bool)]
    if not any(not r["fine_pass"] for r in result) or not any(r["fine_pass"] for r in result):
        raise ValueError("Need base-correct retained AND regressed held-out examples")
    return result


def auc(scores, labels):
    """Mann-Whitney ROC AUC, including 0.5 credit for ties."""
    pos = [float(s) for s, y in zip(scores, labels) if y]
    neg = [float(s) for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg) / (len(pos) * len(neg))


def paired_auc_bootstrap(sa, raw, labels, seed=42, n=1000):
    """Stratified prompt-level exploratory CI for the paired difference."""
    rng = random.Random(seed)
    pos = [i for i, y in enumerate(labels) if y]
    neg = [i for i, y in enumerate(labels) if not y]
    if len(pos) < 2 or len(neg) < 2:
        return None
    diffs = []
    for _ in range(n):
        sample = rng.choices(pos, k=len(pos)) + rng.choices(neg, k=len(neg))
        ys = [labels[i] for i in sample]
        diffs.append(auc([sa[i] for i in sample], ys) - auc([raw[i] for i in sample], ys))
    diffs.sort()
    return [round(diffs[int(.025 * (n - 1))], 4), round(diffs[int(.975 * (n - 1))], 4)]


def select_token_positions(ids, n, rng, all_special_ids):
    """Include final assistant-prefix token and optionally sample earlier prompt tokens."""
    last = len(ids) - 1
    pool = [i for i in range(last) if ids[i] not in all_special_ids]
    extras = rng.sample(pool, min(max(0, n - 1), len(pool)))
    return sorted(extras + [last])


def collect(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    try:
        from prototype.prepare_banking77 import protected_cases
    except ModuleNotFoundError:
        from prepare_banking77 import protected_cases

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")

    eval_rows = get_rows(args.report)
    eval_text = [r["prompt"] for r in eval_rows]
    excluded = set(eval_text)
    banks = [r["prompt"] for r in read_jsonl(args.train_data)]
    rng.shuffle(banks)
    banks = banks[:args.bank_prompts]
    generated = protected_cases(max(args.protected_prompts, 1000), args.seed + 12345)
    prot = [r["prompt"] for r in generated]
    rng.shuffle(prot)
    prot = prot[:args.protected_prompts]
    prompts, types = [], []
    seen = set(excluded)
    for name, group in [("banking", banks), ("protected", prot)]:
        for x in group:
            if x not in seen:
                prompts.append(x)
                types.append(name)
                seen.add(x)
    if len(prompts) < 100:
        raise ValueError("Fewer than 100 distinct SAE calibration prompts")
    pairs = list(zip(prompts, types))
    rng.shuffle(pairs)
    n_val = max(1, int(round(args.validation_fraction * len(pairs))))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager")
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()
    if not (0 <= args.layer <= model.config.num_hidden_layers):
        raise ValueError("Invalid hidden-state index")
    special = set(tokenizer.all_special_ids)

    @torch.inference_mode()
    def encode(pairs_to_capture, is_base, multiple_positions, tag):
        result = []
        context = model.disable_adapter() if is_base else contextlib.nullcontext()
        with context:
            for i, (text, _) in enumerate(pairs_to_capture, start=1):
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=True,
                    add_generation_prompt=True)
                if len(tokens) > args.max_length:
                    raise ValueError(f"Prompt {i} exceeds max_length={args.max_length}")
                ids = torch.tensor([tokens], device=device)
                outputs = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                h = outputs.hidden_states[args.layer][0]
                positions = (select_token_positions(tokens, args.positions_per_prompt, rng, special)
                             if multiple_positions else [len(tokens) - 1])
                result.extend(h[positions].float().cpu().unbind(0))
                del outputs
                if i % 250 == 0 or i == len(pairs_to_capture):
                    print(f"{tag} {i}/{len(pairs_to_capture)}", flush=True)
        return torch.stack(result)

    print(f"Cache train={len(train_pairs)} prompts, valid={len(val_pairs)}, eval={len(eval_rows)}; "
          f"device={device}; layer={args.layer}", flush=True)
    tr = encode(train_pairs, True, True, "calibration")
    va = encode(val_pairs, True, True, "validation")
    ev = list(zip(eval_text, [r["category"] for r in eval_rows]))
    bh = encode(ev, True, False, "base-eval")
    fh = encode(ev, False, False, "fine-eval")
    cache = {
        "train_h": tr, "valid_h": va, "eval_base_h": bh, "eval_fine_h": fh,
        "eval_regressed": [not r["fine_pass"] for r in eval_rows],
        "eval_categories": [r["category"] for r in eval_rows],
        "eval_ids": [r["id"] for r in eval_rows],
        "metadata": {"model": args.model, "adapter": args.adapter, "layer": args.layer,
                     "train_prompt_count": len(train_pairs), "valid_prompt_count": len(val_pairs),
                     "positions_per_prompt": args.positions_per_prompt,
                     "n_bank_prompts": sum(t == "banking" for _, t in pairs),
                     "n_protected_prompts": sum(t == "protected" for _, t in pairs),
                     "seed": args.seed, "report": args.report},
    }
    dest = Path(args.cache)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, dest)
    print(f"Saved {dest} ({tr.shape[0]} train vectors, {va.shape[0]} valid, "
          f"{bh.shape[0]} evaluation pairs)")


def make_sae(d, width, k):
    from torch import nn
    from torch.nn import functional as F

    class TopKSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(d, width)
            self.decoder = nn.Linear(width, d)
            self.k = k
            nn.init.normal_(self.decoder.weight, std=1 / (d ** .5))
            self.normalize_decoder()
            with __import__("torch").no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
                self.encoder.bias.zero_()
                self.decoder.bias.zero_()

        @__import__("torch").no_grad()
        def normalize_decoder(self):
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

        def forward(self, x):
            pre = self.encoder(x)
            values, idx = pre.topk(self.k, dim=-1)
            latents = __import__("torch").zeros_like(pre).scatter(-1, idx, F.relu(values))
            return self.decoder(latents), latents

    return TopKSAE()


def fit(args):
    import torch
    from torch.nn import functional as F

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    c = torch.load(args.cache, map_location="cpu", weights_only=True)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    train_raw, valid_raw = c["train_h"], c["valid_h"]
    mean = train_raw.mean(dim=0, keepdim=True)
    scale = (train_raw - mean).square().mean().sqrt().clamp_min(1e-6)
    tr = ((train_raw - mean) / scale).to(device)
    va = ((valid_raw - mean) / scale).to(device)
    d = tr.shape[1]
    width = int(d * args.expansion)
    if not 1 <= args.k <= width:
        raise ValueError("k must be between 1 and feature dictionary width")
    net = make_sae(d, width, args.k).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad = float("inf"), 0
    best_state = None
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(tr), device=device)
        loss_sum, n_batches = 0.0, 0
        for ids in order.split(args.batch_size):
            x = tr[ids]
            recon, _ = net(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            net.normalize_decoder()
            loss_sum += loss.item()
            n_batches += 1
        net.eval()
        with torch.inference_mode():
            val_loss = sum(F.mse_loss(net(x)[0], x, reduction="sum").item()
                           for x in va.split(args.batch_size)) / va.numel()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1}/{args.epochs} train={loss_sum/n_batches:.4f} "
                  f"valid={val_loss:.4f}", flush=True)
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            best_state = {name: val.detach().cpu().clone() for name, val in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    net.load_state_dict(best_state)
    net.eval()

    @torch.inference_mode()
    def encode(raw):
        x = ((raw - mean) / scale).to(device)
        recs, zs = [], []
        for group in x.split(args.batch_size):
            rec, z = net(group)
            recs.append(rec.cpu())
            zs.append(z.cpu())
        return torch.cat(recs), torch.cat(zs)

    def quality(raw, reconstructed, z):
        x = (raw - mean) / scale
        mse = (reconstructed - x).square().mean().item()
        denom = x.square().mean().item()
        return {"mse": round(mse, 5), "r2_vs_base_train_mean": round(1 - mse / max(denom, 1e-9), 5),
                "mean_l0": round((z > 1e-8).sum(dim=1).float().mean().item(), 2),
                "active_dictionary_fraction": round((z > 1e-8).any(dim=0).float().mean().item(), 4)}

    recon_tr, ztr = encode(train_raw)
    recon_va, zva = encode(valid_raw)
    recon_b, zb = encode(c["eval_base_h"])
    recon_f, zf = encode(c["eval_fine_h"])
    sae_drift = (zf - zb).norm(dim=1).tolist()
    raw_drift = ((c["eval_fine_h"] - c["eval_base_h"]) / scale).norm(dim=1).tolist()
    labels = c["eval_regressed"]
    pca_drift = None
    if args.pca_rank:
        sample = tr.detach().cpu()
        if len(sample) > 2000:
            sample = sample[torch.randperm(len(sample))[:2000]]
        q = min(args.pca_rank, d, len(sample))
        _, _, principal = torch.pca_lowrank(sample, q=q, center=False, niter=2)
        delta_raw = (c["eval_fine_h"] - c["eval_base_h"]) / scale
        pca_drift = (delta_raw @ principal[:, :q]).norm(dim=1).tolist()
    cat_groups = defaultdict(list)
    for i, name in enumerate(c["eval_categories"]):
        cat_groups[name].append(i)
    category_auc = {}
    for name, ix in sorted(cat_groups.items()):
        if len(ix) >= 6 and len(set(labels[i] for i in ix)) == 2:
            category_auc[name] = {
                "count": len(ix), "regressed": sum(labels[i] for i in ix),
                "sae": round(auc([sae_drift[i] for i in ix], [labels[i] for i in ix]), 4),
                "raw": round(auc([raw_drift[i] for i in ix], [labels[i] for i in ix]), 4)}
    score_sae, score_raw = auc(sae_drift, labels), auc(raw_drift, labels)
    score_pca = auc(pca_drift, labels) if pca_drift is not None else None
    result = {
        "model": c["metadata"], "sae": {"type": "TopK", "expansion": args.expansion,
        "width": width, "k": args.k, "seed": args.seed, "best_validation_mse": best,
        "epochs_max": args.epochs},
        "quality": {"train": quality(train_raw, recon_tr, ztr),
                    "validation": quality(valid_raw, recon_va, zva),
                    "heldout_base": quality(c["eval_base_h"], recon_b, zb),
                    "heldout_fine": quality(c["eval_fine_h"], recon_f, zf)},
        "regression_eval": {"n": len(labels), "regressed": sum(labels),
            "retained": len(labels) - sum(labels),
            "sae_auc": score_sae, "raw_auc": score_raw,
            "pca_auc": score_pca, "pca_rank": args.pca_rank,
            "sae_minus_raw_auc": score_sae - score_raw,
            "paired_bootstrap_95_ci_difference": paired_auc_bootstrap(
                sae_drift, raw_drift, labels, args.seed, n=1000),
            "by_category": category_auc},
        "limitations": ["Prompt-level bootstrap ignores within-template dependence.",
                        "No SAE feature is established as causal or semantically interpretable.",
                        "SAE hyperparameters must be selected using reconstruction validation, not test AUC.",
                        "All regression AUC results are observational, on one LoRA checkpoint."],
    }
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "summary.json").open("w") as f:
        json.dump(result, f, indent=2)
    torch.save({"state_dict": best_state, "mean": mean, "scale": scale,
                "d": d, "width": width, "k": args.k}, dest / "sae_checkpoint.pt")
    print(f"Saved {dest}/summary.json and SAE checkpoint")
    print(f"TopK k={args.k}: heldout base R2={result['quality']['heldout_base']['r2_vs_base_train_mean']:.3f}; "
          f"fine R2={result['quality']['heldout_fine']['r2_vs_base_train_mean']:.3f}; "
          f"AUC SAE={score_sae:.3f} vs raw={score_raw:.3f} "
          f"vs PCA={score_pca if score_pca is not None else 'off'}")


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("collect", help="Collect and cache activations ONCE")
    a.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    a.add_argument("--adapter", default="artifacts/banking77-2700-lora")
    a.add_argument("--train-data", default="artifacts/experiments/banking77/train.jsonl")
    a.add_argument("--report", default="artifacts/experiments/banking77/report.json")
    a.add_argument("--cache", default="artifacts/experiments/banking77/sae_v2/cache.pt")
    a.add_argument("--layer", type=int, default=12)
    a.add_argument("--bank-prompts", type=int, default=2700)
    a.add_argument("--protected-prompts", type=int, default=4000)
    a.add_argument("--validation-fraction", type=float, default=0.2)
    a.add_argument("--positions-per-prompt", type=int, default=3)
    a.add_argument("--max-length", type=int, default=256)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    b = sub.add_parser("fit", help="Train one SAE from cached activations")
    b.add_argument("--cache", default="artifacts/experiments/banking77/sae_v2/cache.pt")
    b.add_argument("--out", default="artifacts/experiments/banking77/sae_v2/topk32")
    b.add_argument("--expansion", type=int, default=4)
    b.add_argument("--k", type=int, default=32)
    b.add_argument("--pca-rank", type=int, default=32, help="PCA raw-drift baseline; 0 disables")
    b.add_argument("--epochs", type=int, default=40)
    b.add_argument("--patience", type=int, default=8)
    b.add_argument("--batch-size", type=int, default=256)
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--weight-decay", type=float, default=1e-4)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = p.parse_args()
    if args.mode == "collect":
        if not (0 < args.validation_fraction < 1) or args.positions_per_prompt < 1:
            p.error("validation-fraction must be (0, 1), positions-per-prompt >= 1")
        collect(args)
    else:
        if args.pca_rank < 0 or min(args.k, args.expansion, args.epochs, args.batch_size, args.patience) < 1:
            p.error("k, expansion, epochs, batch-size, patience must be positive")
        fit(args)


if __name__ == "__main__":
    cli()
