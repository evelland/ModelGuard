"""Replay-fraction sweep for the sequential A->B study (local machine).

Reproduces the first clean forgetting setting (disjoint, seed 43, A for 8
epochs, B for 2 epochs, fresh-nonce familiar-template probes), then continues
the SAME saved stage-A adapter on Task B once per replay fraction. Every stage-B
run uses the same number of examples and updates; only the share of replayed
Task A training rows changes. Writes per-fraction comparisons plus a summary
table so A retention can be read against B acquisition.

Reuses existing data / adapter_A / probe files in --root if they are present,
so an already-trained A8 adapter can be dropped in instead of retraining.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def tag(fraction):
    return f"r{round(fraction * 100):02d}"


def kl_tag(weight):
    return "kl" + f"{weight:g}".replace(".", "p")


def act_tag(weight):
    return "act" + f"{weight:g}".replace(".", "p")


def runs(fractions, kl_weights=(), act_weights=()):
    """(label, tag, replay fraction, kl weight, activation weight) for every comparison arm."""
    out = [(f"{round(f * 100)}% replay", tag(f), f, None, None) for f in fractions]
    out += [(f"KL anchor w={w:g}", kl_tag(w), 0.0, w, None) for w in kl_weights]
    out += [(f"Activation anchor w={w:g}", act_tag(w), 0.0, None, w) for w in act_weights]
    return out


def plan(root, fractions, seed=43, condition="disjoint", device="mps",
         epochs_a=8, epochs_b=2, learning_rate=0.0002, max_new_tokens=16, kl_weights=(), act_weights=()):
    root = Path(root)
    p = lambda name: str(root / name)
    py = sys.executable
    dev = ["--device", device]
    model = ["--model", MODEL]
    steps = [
        ("generate data",
         [py, "prototype/sequential_data.py", "--condition", condition,
          "--seed", str(seed), "--out", str(root)],
         root / "manifest.json"),
        ("train Task A adapter",
         [py, "prototype/sequential_train.py", "--stage", "A", *dev, *model,
          "--seed", str(seed), "--train-data", p("train_A.jsonl"),
          "--epochs", str(epochs_a), "--learning-rate", str(learning_rate),
          "--out", p("adapter_A")],
         root / "adapter_A" / "adapter_config.json"),
    ]
    for mode in ("A", "B"):
        steps.append((f"build fresh-nonce Task {mode} probe",
                      [py, "prototype/sequential_transfer_probe.py", "--mode", mode,
                       "--train", p(f"train_{mode}.jsonl"), "--manifest", p("manifest.json"),
                       "--out", p(f"probe_{mode}.jsonl"), "--seed", str(seed)],
                      root / f"probe_{mode}.jsonl"))
    steps.append(("combine probes", ["combine", p("probe_A.jsonl"), p("probe_B.jsonl"),
                                     p("probe_combined.jsonl")],
                  root / "probe_combined.jsonl"))
    steps.append(("evaluate after A",
                  [py, "prototype/evaluate_pair.py", *dev, *model, "--adapter", p("adapter_A"),
                   "--cases", p("probe_combined.jsonl"), "--max-new-tokens", str(max_new_tokens),
                   "--output", p("probe_after_A.jsonl")],
                  root / "probe_after_A.jsonl"))
    for label, t, fraction, kl_weight, act_weight in runs(fractions, kl_weights, act_weights):
        train_b = [py, "prototype/sequential_train.py", "--stage", "B", *dev, *model,
                   "--seed", str(seed), "--train-data", p("train_B.jsonl"),
                   "--prior-adapter", p("adapter_A"), "--epochs", str(epochs_b),
                   "--learning-rate", str(learning_rate), "--out", p(f"adapter_AB_{t}")]
        if fraction:
            train_b += ["--replay-data", p("train_A.jsonl"), "--replay-frac", str(fraction)]
        if kl_weight:
            train_b += ["--kl-data", p("reference_A.jsonl"), "--kl-weight", str(kl_weight)]
        if act_weight:
            train_b += ["--kl-data", p("reference_A.jsonl"), "--act-weight", str(act_weight)]
        steps += [
            (f"train B ({label})", train_b, root / f"adapter_AB_{t}" / "adapter_config.json"),
            (f"evaluate after B ({t})",
             [py, "prototype/evaluate_pair.py", *dev, *model, "--adapter", p(f"adapter_AB_{t}"),
              "--cases", p("probe_combined.jsonl"), "--max-new-tokens", str(max_new_tokens),
              "--output", p(f"probe_after_B_{t}.jsonl")],
             root / f"probe_after_B_{t}.jsonl"),
            (f"score {t}",
             [py, "prototype/sequential_score.py", "--after-a", p("probe_after_A.jsonl"),
              "--after-b", p(f"probe_after_B_{t}.jsonl"), "--out", p(f"comparison_{t}.json")],
             root / f"comparison_{t}.json"),
        ]
    return steps


def combine(a_path, b_path, out_path):
    rows = []
    for path in (a_path, b_path):
        rows += [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    Path(out_path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def summarize(root, fractions, kl_weights=(), act_weights=()):
    root = Path(root)
    table = []
    for label, t, fraction, kl_weight, act_weight in runs(fractions, kl_weights, act_weights):
        comparison = json.loads((root / f"comparison_{t}.json").read_text())
        meta = json.loads((root / f"adapter_AB_{t}" / "run_metadata.json").read_text())
        a, b = comparison["stage_comparison"]["A"], comparison["stage_comparison"]["B"]
        table.append({
            "method": label, "replay_fraction": fraction, "kl_weight": kl_weight or 0.0, "act_weight": act_weight or 0.0,
            "replay_examples": (meta.get("replay") or {}).get("replay_examples", 0),
            "stage_B_examples": meta["train_samples"], "stage_B_updates": meta["updates"],
            "A_after_A": f"{a['after_A_correct']}/{a['n']}",
            "A_after_B": f"{a['after_B_correct']}/{a['n']}",
            "A_new_failures": a["A_to_B_regressions"],
            "A_retention": a["retention_given_A_correct"],
            "B_after_A": f"{b['after_A_correct']}/{b['n']}",
            "B_after_B": f"{b['after_B_correct']}/{b['n']}",
            "B_accuracy_after_B": b["after_B_accuracy"],
        })
    summary = {"rows": table, "caveats": [
        "Single mapping seed and single training seed; replicate before drawing conclusions.",
        "Probes share item identities and training templates with fresh nonces (familiar-template setting).",
        "Replay at a matched stage-B example budget: replay rows displace B rows.",
        "Compare retention at matched B accuracy; a retention gain bought with lost B learning is not a win.",
    ]}
    (root / "replay_sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| Method | Replayed / total B ex. | Updates | A after A | A after B | New A failures | A retention | B after A | B after B |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['method']} | {r['replay_examples']}/{r['stage_B_examples']} | "
                     f"{r['stage_B_updates']} | {r['A_after_A']} | {r['A_after_B']} | {r['A_new_failures']} | "
                     f"{r['A_retention']} | {r['B_after_A']} | {r['B_after_B']} |")
    (root / "replay_sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["method"] for r in table]
        retention = [100 * (r["A_retention"] or 0) for r in table]
        b_acc = [100 * r["B_accuracy_after_B"] for r in table]
        xs = range(len(table))
        fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(table)), 4))
        bars_a = ax.bar([x - 0.2 for x in xs], retention, 0.4, label="Task A retained (%)", color="#0F766E")
        bars_b = ax.bar([x + 0.2 for x in xs], b_acc, 0.4, label="Task B accuracy (%)", color="#F2B134")
        ax.bar_label(bars_a, fmt="%.0f", fontsize=8)
        ax.bar_label(bars_b, fmt="%.0f", fontsize=8)
        ax.set_xticks(list(xs), names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("%")
        ax.set_title("Learning vs. forgetting (matched stage-B budget)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(root / "replay_tradeoff.png", dpi=200, bbox_inches="tight")
        print(f"Plot: {root / 'replay_tradeoff.png'}")
    except ImportError:
        print("matplotlib not installed; skipped plot")
    return summary


def run(args):
    fractions = sorted(set(args.fractions))
    if any(not 0 <= f < 1 for f in fractions):
        raise ValueError("Fractions must be in [0, 1)")
    root = (REPO / args.root).resolve()
    kl_weights = sorted(set(getattr(args, "kl_weights", None) or []))
    if any(w <= 0 for w in kl_weights):
        raise ValueError("KL weights must be positive")
    act_weights = sorted(set(getattr(args, "act_weights", None) or []))
    if any(w <= 0 for w in act_weights):
        raise ValueError("Activation weights must be positive")
    steps = plan(root, fractions, args.seed, args.condition, args.device,
                 args.epochs_a, args.epochs_b, args.learning_rate, args.max_new_tokens, kl_weights, act_weights)
    for i, (label, command, output) in enumerate(steps, 1):
        if output.exists():
            print(f"[{i}/{len(steps)}] skip (exists): {label}")
            continue
        print(f"\n[{i}/{len(steps)}] {label}\n  {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        if command[0] == "combine":
            combine(*command[1:])
        else:
            subprocess.run(command, cwd=REPO, check=True)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Step '{label}' produced no output at {output}")
    if not args.dry_run:
        summarize(root, fractions, kl_weights, act_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/experiments/sequential/replay_disjoint_seed43")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--kl-weights", type=float, nargs="*", default=[],
                        help="Also run KL-anchor arms at these weights (e.g. 0.5 1 2)")
    parser.add_argument("--act-weights", type=float, nargs="*", default=[],
                        help="Also run raw-activation-anchor arms at these weights (layer 12)")
    parser.add_argument("--condition", choices=("disjoint", "shared"), default="disjoint")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--epochs-a", type=int, default=8)
    parser.add_argument("--epochs-b", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
