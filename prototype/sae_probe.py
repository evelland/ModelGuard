"""Quick SAE diagnostic on Qwen base vs LoRA activations (NOT causal attribution).

Fits a frozen SAE on separate base-model calibration prompts and compares
feature shifts on the same held-out prompts that were scored by ModelGuard.
"""
import argparse
import contextlib
import html
import json
import random
import statistics
from pathlib import Path


def read_regression_rows(path):
    report = json.loads(Path(path).read_text())
    rows = report["rows"]
    return [r for r in rows if r.get("group") == "protected"
            and r.get("base_pass") is True and isinstance(r.get("fine_pass"), bool)]


def choose_examples(rows, count, seed):
    """Keep two balanced groups when doing a short exploratory run."""
    lost = [r for r in rows if not r["fine_pass"]]
    kept = [r for r in rows if r["fine_pass"]]
    rng = random.Random(seed)
    rng.shuffle(lost)
    rng.shuffle(kept)
    if count > 0:
        n = min(count // 2, len(lost), len(kept))
        if n < 2:
            raise ValueError("Need at least 2 regressed and 2 retained base-correct examples")
        return lost[:n] + kept[:n]
    if not lost or not kept:
        raise ValueError("Need both regression and retention examples")
    return lost + kept


def auc_binary(scores, positives):
    """Mann-Whitney ROC AUC; descriptive on THIS evaluation sample only."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0)
               for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def auto_device(torch, requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="artifacts/banking77-2700-lora")
    p.add_argument("--report", default="artifacts/experiments/banking77/report.json")
    p.add_argument("--output", default="artifacts/experiments/banking77/sae_probe")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--layer", type=int, default=12, help="Hidden-state index; 0=embedding output")
    p.add_argument("--calibration", type=int, default=600)
    p.add_argument("--eval-limit", type=int, default=80, help="0=all base-correct examples; otherwise balance lost/kept")
    p.add_argument("--expansion", type=int, default=2)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--l1", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.calibration, args.expansion, args.epochs) < 1 or args.l1 < 0:
        p.error("calibration, expansion, epochs must be positive and l1 nonnegative")
    if args.eval_limit == 1:
        p.error("Use --eval-limit >= 4 or 0")

    import torch
    from torch import nn
    from torch.nn import functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prepare_banking77 import protected_cases

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = auto_device(torch, args.device)
    print(f"Device: {device}", flush=True)
    rows = choose_examples(read_regression_rows(args.report), args.eval_limit, args.seed)
    labels = [not row["fine_pass"] for row in rows]
    eval_set = {r["prompt"] for r in rows}
    # The calibration prompts are generated from a DIFFERENT seed; no held-out
    # prompts used to fit the SAE, and no regression labels used for fitting.
    source = protected_cases(max(2 * args.calibration, 1000), args.seed + 10000)
    seen = set(eval_set)
    calibration_prompts = []
    for record in source:
        prompt = record["prompt"]
        if prompt not in seen:
            calibration_prompts.append(prompt)
            seen.add(prompt)
            if len(calibration_prompts) == args.calibration:
                break
    if len(calibration_prompts) < args.calibration:
        raise ValueError("Not enough distinct calibration prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()
    total_states = int(model.config.num_hidden_layers) + 1
    if not (0 <= args.layer < total_states):
        p.error(f"--layer must be between 0 and {total_states - 1}")

    @torch.inference_mode()
    def activations(prompts, is_base, name):
        captures = []
        context = model.disable_adapter() if is_base else contextlib.nullcontext()
        with context:
            for i, prompt in enumerate(prompts, 1):
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)
                output = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                captures.append(output.hidden_states[args.layer][0, -1].detach().to("cpu").float())
                if i % 50 == 0 or i == len(prompts):
                    print(f"{name}: {i}/{len(prompts)}", flush=True)
        return torch.stack(captures)

    print(f"Collecting {len(calibration_prompts)} separate base calibration activations...", flush=True)
    calibrate_h = activations(calibration_prompts, True, "calibration")
    base_h = activations([r["prompt"] for r in rows], True, "base evaluation")
    tuned_h = activations([r["prompt"] for r in rows], False, "LoRA evaluation")

    mean = calibrate_h.mean(dim=0, keepdim=True)
    scale = calibrate_h.std(dim=0, keepdim=True).clamp(min=1e-3)
    x_train = ((calibrate_h - mean) / scale).to(device)
    d = x_train.shape[1]
    width = d * args.expansion

    class SparseAutoencoder(nn.Module):
        def __init__(self, d_model, n_features):
            super().__init__()
            self.encoder = nn.Linear(d_model, n_features)
            self.decoder = nn.Linear(n_features, d_model)

        def forward(self, x):
            features = F.relu(self.encoder(x))
            return self.decoder(features), features

    sae = SparseAutoencoder(d, width).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    indices = list(range(len(x_train)))
    for epoch in range(args.epochs):
        random.shuffle(indices)
        losses = []
        sae.train()
        for start in range(0, len(indices), 64):
            xb = x_train[indices[start:start + 64]]
            reconstruction, features = sae(xb)
            loss = F.mse_loss(reconstruction, xb) + args.l1 * features.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"SAE epoch {epoch+1}/{args.epochs}: loss={statistics.mean(losses):.5f}", flush=True)
    sae.eval()

    @torch.inference_mode()
    def project(h):
        x = ((h - mean) / scale).to(device)
        reconstructed, features = sae(x)
        mse = F.mse_loss(reconstructed, x).item()
        var = ((x - x.mean(dim=0)) ** 2).mean().item()
        return features.cpu(), {"mse": mse, "r2": 1.0 - mse / max(var, 1e-9),
                                "mean_active": float((features > 1e-6).float().sum(dim=1).mean().item())}

    z_base, q_base = project(base_h)
    z_tuned, q_tuned = project(tuned_h)
    z_cal, q_cal = project(calibrate_h)
    drift_feature = (z_tuned - z_base).square().sum(dim=1).sqrt().tolist()
    drift_raw = (((tuned_h - base_h) / scale).square().sum(dim=1).sqrt()).tolist()
    drift_delta = (z_tuned - z_base).abs()
    lost_mask = torch.tensor(labels, dtype=torch.bool)
    kept_mask = ~lost_mask
    lost_means = drift_delta[lost_mask].mean(dim=0)
    kept_means = drift_delta[kept_mask].mean(dim=0)
    contrasts = lost_means - kept_means
    top_indices = torch.argsort(contrasts, descending=True)[:10].tolist()

    ranked = []
    for j in top_indices:
        cal_top = torch.argsort(z_cal[:, j], descending=True)[:3].tolist()
        ranked.append({
            "feature_id": j,
            "regressed_mean_abs_change": round(float(lost_means[j]), 5),
            "retained_mean_abs_change": round(float(kept_means[j]), 5),
            "difference": round(float(contrasts[j]), 5),
            "highest_activation_calibration_prompts": [
                {"prompt": calibration_prompts[i], "activation": round(float(z_cal[i, j]), 4)}
                for i in cal_top
            ],
        })

    result = {
        "experiment_type": "exploratory_correlation_not_causality",
        "model": args.model, "adapter": args.adapter, "hidden_state_index": args.layer,
        "activation_position": "final assistant-prefix token",
        "sae": {"features": width, "input_dim": d, "l1": args.l1, "epochs": args.epochs,
                "train_calibration_prompts": len(calibration_prompts)},
        "sample": {"n": len(rows), "regressed": sum(labels), "retained": len(rows)-sum(labels),
                   "selection": "base-correct only; balanced by regression status when eval-limit > 0"},
        "reconstruction": {"calibration": q_cal, "heldout_base": q_base, "heldout_tuned": q_tuned},
        "roc_auc_on_analyzed_examples": {
            "sae_total_feature_drift": auc_binary(drift_feature, labels),
            "raw_normalized_activation_drift": auc_binary(drift_raw, labels),
            "note": "In-sample descriptive AUC only; no separate validation of feature selection."
        },
        "mean_drift": {
            "sae_regressed": statistics.mean(s for s, y in zip(drift_feature, labels) if y),
            "sae_retained": statistics.mean(s for s, y in zip(drift_feature, labels) if not y),
            "raw_regressed": statistics.mean(s for s, y in zip(drift_raw, labels) if y),
            "raw_retained": statistics.mean(s for s, y in zip(drift_raw, labels) if not y),
        },
        "top_associated_features": ranked,
        "limitations": [
            "SAE trained on small final-token-only calibration dataset; features not semantically validated.",
            "Feature changes correlate with behavioral regressions; no causal claims.",
            "Top-feature rankings and AUC are descriptive on same evaluation sample.",
            "Reconstruction on tuned activations may degrade from distribution shift.",
            "Benchmark consists of template-generated prompts; results require replication.",
        ]
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"state_dict": sae.cpu().state_dict(), "center": mean, "scale": scale,
                "layer": args.layer, "model": args.model}, output / "sae_checkpoint.pt")

    esc = lambda t: html.escape(str(t), quote=True)
    feature_html = "".join(
        "<tr><td>" + str(f["feature_id"]) + "</td><td>"
        + f'{f["regressed_mean_abs_change"]:.3f}' + "</td><td>"
        + f'{f["retained_mean_abs_change"]:.3f}' + "</td><td><code>"
        + esc(f["highest_activation_calibration_prompts"][0]["prompt"]) + "</code></td></tr>"
        for f in ranked
    )
    auc_sae = result["roc_auc_on_analyzed_examples"]["sae_total_feature_drift"]
    auc_raw = result["roc_auc_on_analyzed_examples"]["raw_normalized_activation_drift"]
    html_report = f"""<!doctype html><html><head><meta charset="utf-8">
<style>body{{font:16px/1.5 system-ui;background:#101b20;color:#edf6f4;max-width:980px;margin:50px auto;padding:0 20px}}
p,small{{color:#bdd0cf}}.kpi{{display:flex;gap:15px;flex-wrap:wrap}}.kpi div{{border:1px solid #385351;border-radius:8px;padding:20px;flex:1;min-width:170px}}
b{{font-size:27px;color:#80ddac}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #385351}}
code{{font-size:12px;overflow-wrap:anywhere}}.caution{{background:#443d20;padding:15px;border-radius:8px;color:#fff1bb}}</style></head><body>
<h1>ModelGuard — SAE activation diff (pilot)</h1>
<p class="caution">EXPLORATORY ONLY: SAE feature changes are not explanations or proof of causation. No SAE-preserving training was performed.</p>
<p>Base vs LoRA · hidden state {args.layer} · {len(rows)} base-correct protected prompts
({sum(labels)} regressed / {len(rows)-sum(labels)} retained)</p>
<div class="kpi"><div>SAE features<br><b>{width}</b></div><div>Base held-out SAE R²<br><b>{q_base["r2"]:.2f}</b></div>
<div>Drift-regression AUC (SAE)<br><b>{auc_sae:.3f}</b></div><div>Raw activation AUC<br><b>{auc_raw:.3f}</b></div></div>
<h2>Does activation drift distinguish failures?</h2>
<p>Average SAE drift: failures {result["mean_drift"]["sae_regressed"]:.2f};
retained {result["mean_drift"]["sae_retained"]:.2f}. AUC is descriptive on these same examples;
a higher value indicates larger drift tends to accompany new failures.</p>
<h2>Candidate features more changed in regressions</h2>
<table><tr><th>Feature ID</th><th>Mean |Δ| in regressions</th><th>Mean |Δ| retained</th><th>High-activation calibration prompt (not a label)</th></tr>{feature_html}</table>
<h2>SAE sanity checks</h2>
<p>Mean nonzero features per calibration sample {q_cal["mean_active"]:.1f}/{width};
reconstruction R² on held-out base {q_base["r2"]:.3f}, LoRA {q_tuned["r2"]:.3f}.</p>
<p><small>One residual-stream vector at the final prompt position; small calibration set, unvalidated feature semantics;
compare against the raw-activation baseline and replicate before drawing conclusions.</small></p>
</body></html>"""
    (output / "summary.html").write_text(html_report)
    print(f"Saved {output / 'summary.html'} and summary.json", flush=True)
    print(f"SAE AUC={auc_sae:.3f} raw AUC={auc_raw:.3f} base R2={q_base['r2']:.3f}", flush=True)


if __name__ == "__main__":
    main()
