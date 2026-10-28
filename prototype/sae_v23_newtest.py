"""Frozen new-template transfer evaluation for ModelGuard SAE v2.3.

Commands:
  freeze  - freeze benchmark checksum, checkpoints and score directions on OLD dev
  collect - paired base/LoRA hidden states for NEW already-inferred benchmark
  score   - evaluate the frozen detectors on the NEW base-correct cases

Never train on or choose score orientation using new-template predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

try:
    from prototype.sae_v2 import make_sae, auc
    from prototype.sae_v21_analysis import stratified_split, within_category_auc
    from prototype.report import score as behavior_score
except ModuleNotFoundError:
    from sae_v2 import make_sae, auc
    from sae_v21_analysis import stratified_split, within_category_auc
    from report import score as behavior_score


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_rows(cases, predictions):
    """Require exactly the frozen cases, with no missing/reordered/extra IDs."""
    if not cases or len(cases) != len(predictions):
        raise ValueError("Predictions must cover exactly the frozen benchmark")
    case_ids = [r["id"] for r in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate benchmark case IDs")
    pred_ids = [r["id"] for r in predictions]
    if len(set(pred_ids)) != len(pred_ids) or case_ids != pred_ids:
        raise ValueError("Prediction IDs must be unique and in benchmark order")
    for ref, got in zip(cases, predictions):
        for key in ("id", "prompt", "category", "group", "expected", "validator", "template_family"):
            if got.get(key) != ref.get(key):
                raise ValueError(f"Prediction/benchmark mismatch: {ref['id']} {key}")
        if got.get("source") != "model_inference":
            raise ValueError("Predictions must come from model inference")
        if not isinstance(got.get("base_response"), str) or not isinstance(got.get("fine_response"), str):
            raise ValueError("Missing generated model responses")


def outcome_summary(cases, predictions):
    validate_rows(cases, predictions)
    buckets = defaultdict(list)
    for row in predictions:
        base = behavior_score(row, row["base_response"])
        fine = behavior_score(row, row["fine_response"])
        buckets[row["category"]].append((base, fine))
    def summarize(values):
        return {
            "n": len(values),
            "base_correct": sum(b for b, _ in values),
            "fine_correct": sum(f for _, f in values),
            "new_regressions": sum(b and not f for b, f in values),
            "retained_base_correct": sum(b and f for b, f in values),
            "new_gains": sum((not b) and f for b, f in values),
            "base_incorrect_both": sum((not b) and (not f) for b, f in values),
        }
    return {"total": summarize(sum(buckets.values(), [])),
            "by_category": {k: summarize(v) for k, v in sorted(buckets.items())}}


def get_checkpoint_scores(base, fine, checkpoint):
    """Same latent-norm definitions as SAE v2.2; NO model training."""
    net = make_sae(int(checkpoint["d"]), int(checkpoint["width"]), int(checkpoint["k"]))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    center, scale = checkpoint["mean"].float(), checkpoint["scale"].float()
    if base.shape[1] != checkpoint["d"] or base.shape != fine.shape:
        raise ValueError("Checkpoint dimensions do not match paired activations")
    with torch.inference_mode():
        if checkpoint["scheme"] == "delta":
            z = net(((fine - base) - center) / scale)[1]
            return z.norm(dim=1).tolist()
        zbase = net((base - center) / scale)[1]
        zfine = net((fine - center) / scale)[1]
        return (zfine - zbase).norm(dim=1).tolist()


def pca_directions(paired, rank=32):
    tr = (paired["train_fine_h"] - paired["train_base_h"]).double()
    mean = tr.mean(dim=0, keepdim=True)
    centered = tr - mean
    vals, vectors = torch.linalg.eigh(centered.T @ centered)
    return mean.float(), vectors[:, -rank:].float()


def detection_scores(base, fine, checkpoints, mean, basis):
    delta = fine - base
    scores = {"raw_delta_norm": delta.norm(dim=1).tolist(),
              "pca_centered_norm": ((delta - mean) @ basis).norm(dim=1).tolist()}
    for name, ck in checkpoints.items():
        scores[name] = get_checkpoint_scores(base, fine, ck)
    return scores


def file_manifest(args):
    result = {"cases": args.cases, "old_eval": args.old_eval,
              "paired_cache": args.paired_cache,
              "checkpoints": {
                  "base_k128": args.base_checkpoint,
                  "pooled_k128": args.pooled_checkpoint,
                  "delta_k128": args.delta_checkpoint,
              }}
    paths = {k: result[k] for k in ("cases", "old_eval", "paired_cache")}
    paths.update(result["checkpoints"])
    result["sha256"] = {key: checksum(path) for key, path in paths.items()}
    return result


def verify_files(manifest, keys=None):
    paths = {k: manifest[k] for k in ("cases", "old_eval", "paired_cache")}
    paths.update(manifest["checkpoints"])
    for key, path in paths.items():
        if keys is not None and key not in keys:
            continue
        if checksum(path) != manifest["sha256"][key]:
            raise ValueError(f"Frozen input changed since freeze: {key} ({path})")


def load_checkpoints(manifest):
    return {key: torch.load(path, map_location="cpu", weights_only=True)
            for key, path in manifest["checkpoints"].items()}


def prepare_frozen_plan(args):
    """Calibrate direction solely on OLD protected development examples."""
    target = Path(args.plan)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite frozen plan: {target}")
    files = file_manifest(args)
    old = torch.load(args.old_eval, map_location="cpu", weights_only=True)
    paired = torch.load(args.paired_cache, map_location="cpu", weights_only=True)
    base = old["eval_base_h"].float()
    fine = old["eval_fine_h"].float()
    labels = [bool(x) for x in old["eval_regressed"]]
    cats = old["eval_categories"]
    if len(labels) != len(base) or len(cats) != len(base):
        raise ValueError("Old evaluation cache is misaligned")
    ckpts = load_checkpoints(files)
    model = old["metadata"]["model"]
    layer = old["metadata"]["layer"]
    if paired["metadata"]["model"] != model or paired["metadata"]["layer"] != layer:
        raise ValueError("Old and paired cache model/layer mismatch")
    if any(ck["d"] != base.shape[1] for ck in ckpts.values()):
        raise ValueError("Checkpoint hidden-state width mismatch")
    if any(ck["scheme"] != k.split("_k")[0] for k, ck in ckpts.items()):
        raise ValueError("Checkpoint scheme/file name mismatch")
    mean, basis = pca_directions(paired, args.pca_rank)
    scores = detection_scores(base, fine, ckpts, mean, basis)
    dev, _ = stratified_split(cats, labels, 0.4, args.seed)
    directions = {}
    for key, values in scores.items():
        dev_auc, _ = within_category_auc(values, labels, cats, dev)
        if dev_auc is None:
            raise ValueError("Old dev set has no comparable within-category cases")
        directions[key] = {"sign": 1 if dev_auc >= 0.5 else -1,
                           "old_dev_auc_unoriented": round(dev_auc, 5)}
    cases = read_jsonl(args.cases)
    if len(set(r["prompt"] for r in cases)) != len(cases):
        raise ValueError("Duplicate prompts in frozen new-template benchmark")
    plan = {"protocol": "sae_v23_frozen_new_template_transfer",
            "model": model, "adapter": paired["metadata"]["adapter"], "layer": layer,
            "old_dev_seed": args.seed, "old_dev_n": len(dev),
            "pca_rank": args.pca_rank, "directions": directions, "files": files,
            "benchmark_n": len(cases),
            "limitations": [
                "Model selection was exploratory on the old protected benchmark.",
                "New prompts use new templates but are not a fully independent blind dataset.",
                "AUC conditions on base-correct cases and is undefined in one-class categories.",
                "No model or SAE is trained on the new benchmark.",
            ]}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Frozen {len(directions)} detectors using OLD dev only; plan: {target}")
    return plan


def collect_new(args):
    """Capture exactly one final assistant-prefix activation per new-template case."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    try:
        from prototype.sae_v22 import resolve_device
    except ModuleNotFoundError:
        from sae_v22 import resolve_device

    plan = json.loads(Path(args.plan).read_text())
    manifest = plan["files"]
    verify_files(manifest, keys={"cases"})
    cases = read_jsonl(manifest["cases"])
    predictions = read_jsonl(args.predictions)
    validate_rows(cases, predictions)
    actual_device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(plan["model"])
    base_model = AutoModelForCausalLM.from_pretrained(
        plan["model"], torch_dtype=torch.float32, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, plan["adapter"]).to(actual_device).eval()
    a, b = [], []
    with torch.inference_mode():
        for i, row in enumerate(cases, 1):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt").to(actual_device)
            if ids.shape[1] > args.max_length:
                raise ValueError(f"Prompt exceeds max length: {row['id']}")
            with model.disable_adapter():
                result = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                a.append(result.hidden_states[plan["layer"]][0, -1].float().cpu())
                del result
            result = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            b.append(result.hidden_states[plan["layer"]][0, -1].float().cpu())
            del result
            if i % 100 == 0 or i == len(cases):
                print(f"Holdout activation collection: {i}/{len(cases)}", flush=True)
    cache = {"eval_base_h": torch.stack(a), "eval_fine_h": torch.stack(b),
             "eval_ids": [row["id"] for row in cases],
             "eval_categories": [row["category"] for row in cases],
             "predictions_sha256": checksum(args.predictions),
             "cases_sha256": manifest["sha256"]["cases"],
             "model": plan["model"], "layer": plan["layer"]}
    output = Path(args.out)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite new-template activations: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"Saved {output}")


def score_new(args):
    plan = json.loads(Path(args.plan).read_text())
    verify_files(plan["files"])
    cases = read_jsonl(plan["files"]["cases"])
    preds = read_jsonl(args.predictions)
    validate_rows(cases, preds)
    cache = torch.load(args.activation_cache, map_location="cpu", weights_only=True)
    if cache["eval_ids"] != [r["id"] for r in cases] or cache["cases_sha256"] != plan["files"]["sha256"]["cases"]:
        raise ValueError("Activation cache does not match frozen benchmark")
    if cache["predictions_sha256"] != checksum(args.predictions):
        raise ValueError("Predictions changed after activation collection")
    if cache["model"] != plan["model"] or cache["layer"] != plan["layer"]:
        raise ValueError("Model or layer mismatch")
    base, fine = cache["eval_base_h"].float(), cache["eval_fine_h"].float()
    if len(base) != len(cases) or base.shape != fine.shape:
        raise ValueError("Misaligned new-template hidden states")
    paired = torch.load(plan["files"]["paired_cache"], map_location="cpu", weights_only=True)
    checkpoints = load_checkpoints(plan["files"])
    mean, basis = pca_directions(paired, plan["pca_rank"])
    score_map = detection_scores(base, fine, checkpoints, mean, basis)
    eligible = [i for i, r in enumerate(preds) if behavior_score(r, r["base_response"])]
    y = [behavior_score(preds[i], preds[i]["base_response"]) and
         not behavior_score(preds[i], preds[i]["fine_response"]) for i in range(len(preds))]
    cats = [r["category"] for r in preds]
    metrics = {}
    for name, values in score_map.items():
        sign = plan["directions"][name]["sign"]
        ranked = [sign * v for v in values]
        conditional, by_category = within_category_auc(ranked, y, cats, eligible)
        pool = auc([ranked[i] for i in eligible], [y[i] for i in eligible]) if eligible else None
        metrics[name] = {
            "frozen_direction": "greater" if sign > 0 else "smaller",
            "eligible_n": len(eligible),
            "within_category_auc": round(conditional, 5) if conditional is not None else None,
            "pooled_auc": round(pool, 5) if pool is not None else None,
            "by_category": by_category,
        }
    result = {
        "protocol": plan["protocol"], "benchmark_sha256": plan["files"]["sha256"]["cases"],
        "predictions_sha256": cache["predictions_sha256"],
        "behavior": outcome_summary(cases, preds),
        "frozen_regression_detectors": metrics,
        "note": "Detectors calibrated on OLD dev only; new-template test used only for reporting.",
        "limitations": plan["limitations"],
    }
    dest = Path(args.out)
    if dest.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite new-template evaluation: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"New-template test n={len(cases)}; base correct={len(eligible)}; "
          f"new regressions={result['behavior']['total']['new_regressions']}")
    for name, item in metrics.items():
        print(f"{name} within-category ROC AUC: {item['within_category_auc']}")
    print(f"Saved {dest}")
    return result


def main():
    root = "artifacts/experiments/banking77"
    v23 = f"{root}/sae_v23"
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--cases", default=f"{v23}/new_template_protected.jsonl")
    f.add_argument("--old-eval", default=f"{root}/sae_v2/cache.pt")
    f.add_argument("--paired-cache", default=f"{root}/sae_v22/paired_cache.pt")
    for scheme in ("base", "pooled", "delta"):
        f.add_argument(f"--{scheme}-checkpoint",
            default=f"{root}/sae_v22/{scheme}_topk128_seed42/sae_checkpoint.pt")
    f.add_argument("--pca-rank", type=int, default=32)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--plan", default=f"{v23}/frozen_newtest_plan.json")
    c = sub.add_parser("collect")
    c.add_argument("--plan", default=f"{v23}/frozen_newtest_plan.json")
    c.add_argument("--predictions", default=f"{v23}/new_template_predictions.jsonl")
    c.add_argument("--out", default=f"{v23}/new_template_activations.pt")
    c.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    c.add_argument("--max-length", type=int, default=256)
    c.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--plan", default=f"{v23}/frozen_newtest_plan.json")
    s.add_argument("--predictions", default=f"{v23}/new_template_predictions.jsonl")
    s.add_argument("--activation-cache", default=f"{v23}/new_template_activations.pt")
    s.add_argument("--out", default=f"{v23}/new_template_results.json")
    s.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.command == "freeze":
        if args.pca_rank < 1:
            p.error("PCA rank must be >= 1")
        prepare_frozen_plan(args)
    elif args.command == "collect":
        collect_new(args)
    else:
        score_new(args)


if __name__ == "__main__":
    main()
