"""Clean end-to-end sequential A->B run on a LOCAL machine.

Generates a new dataset, trains a fresh Qwen LoRA adapter on A, continues the
same adapter on B, evaluates BOTH checkpoints on identical DEV examples, and
writes dev_comparison.json. The held-out TEST is intentionally untouched.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def steps(root, condition="disjoint", seed=43, device="mps",
          epochs_a=4, epochs_b=2, learning_rate=0.0002, max_new_tokens=16):
    root = Path(root)
    p = lambda name: str(root / name)
    common = ["--device", device]
    model_args = ["--model", "Qwen/Qwen2.5-0.5B-Instruct"]
    return [
        ("generate fresh data",
         [sys.executable, "prototype/sequential_data.py", "--condition", condition,
          "--seed", str(seed), "--out", str(root)],
         [root / "train_A.jsonl", root / "train_B.jsonl",
          root / "eval_combined_dev.jsonl", root / "manifest.json"]),
        ("train new Task A adapter",
         [sys.executable, "prototype/sequential_train.py", "--stage", "A",
          *common, *model_args, "--seed", str(seed),
          "--train-data", p("train_A.jsonl"), "--epochs", str(epochs_a),
          "--learning-rate", str(learning_rate), "--out", p("adapter_A")],
         [root / "adapter_A" / "adapter_config.json"]),
        ("evaluate after A on both tasks",
         [sys.executable, "prototype/evaluate_pair.py", *common, *model_args,
          "--adapter", p("adapter_A"),
          "--cases", p("eval_combined_dev.jsonl"),
          "--max-new-tokens", str(max_new_tokens), "--output", p("dev_after_A.jsonl")],
         [root / "dev_after_A.jsonl"]),
        ("continue SAME adapter on B",
         [sys.executable, "prototype/sequential_train.py", "--stage", "B",
          *common, *model_args, "--seed", str(seed),
          "--train-data", p("train_B.jsonl"), "--prior-adapter", p("adapter_A"),
          "--epochs", str(epochs_b), "--learning-rate", str(learning_rate),
          "--out", p("adapter_AB")],
         [root / "adapter_AB" / "adapter_config.json"]),
        ("evaluate after B on same cases",
         [sys.executable, "prototype/evaluate_pair.py", *common, *model_args,
          "--adapter", p("adapter_AB"),
          "--cases", p("eval_combined_dev.jsonl"),
          "--max-new-tokens", str(max_new_tokens), "--output", p("dev_after_B.jsonl")],
         [root / "dev_after_B.jsonl"]),
        ("compare A retention with B acquisition",
         [sys.executable, "prototype/sequential_score.py",
          "--after-a", p("dev_after_A.jsonl"), "--after-b", p("dev_after_B.jsonl"),
          "--out", p("dev_comparison.json")],
         [root / "dev_comparison.json"]),
    ]


def checked_outputs(label, paths):
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Step '{label}' did not produce a nonempty file: {path}")


def validated_predictions(root, filename):
    """Ensure evaluation finished, not just started writing JSONL."""
    data_path = root / filename
    cases = root / "eval_combined_dev.jsonl"
    with cases.open(encoding="utf-8") as f:
        source = [json.loads(line) for line in f if line.strip()]
    with data_path.open(encoding="utf-8") as f:
        generated = [json.loads(line) for line in f if line.strip()]
    if len(generated) != len(source):
        raise ValueError(
            f"Incomplete evaluation: {len(generated)} generated / {len(source)} expected "
            f"for {filename}"
        )
    for a, b in zip(source, generated):
        if a["id"] != b["id"] or not isinstance(b.get("fine_response"), str):
            raise ValueError(f"Misaligned evaluation output in {filename}")
    print(f"Verified {len(generated)} complete predictions: {data_path}", flush=True)


def run(args):
    # All existing training artifacts are preserved. The new run lives in an
    # independent folder; no existing directory is silently overwritten.
    root = (REPO / args.out).resolve()
    if root.exists():
        raise FileExistsError(
            f"Run directory already exists: {root}\n"
            "Choose a fresh --out path; this runner deliberately never overwrites results."
        )
    plan = steps(root, args.condition, args.seed, args.device,
                 args.epochs_a, args.epochs_b, args.learning_rate, args.max_new_tokens)

    print(f"Repository: {REPO}\nNew run directory: {root}", flush=True)
    print("Base checkpoint: Qwen/Qwen2.5-0.5B-Instruct (fresh LoRA initialization)")
    print("Only DEV is evaluated; held-out TEST is reserved for later.", flush=True)
    if args.dry_run:
        for i, (label, command, _) in enumerate(plan, 1):
            print(f"{i}. {label}\n   {shlex.join(command)}")
        return

    # torch/transformers are not imported here: missing dependencies fail in
    # the first model-bearing step with a readable command and traceback.
    for i, (label, command, expected) in enumerate(plan, 1):
        print(f"\n{'='*68}\nSTEP {i}/{len(plan)}: {label}\n{shlex.join(command)}",
              flush=True)
        subprocess.run(command, cwd=REPO, check=True)
        checked_outputs(label, expected)
        if i == 3:
            validated_predictions(root, "dev_after_A.jsonl")
        if i == 5:
            validated_predictions(root, "dev_after_B.jsonl")

    report_path = root / "dev_comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 68)
    print("COMPLETE. Development results:")
    for mode, metric in report["stage_comparison"].items():
        print(f"  Task {mode}: after A {metric['after_A_correct']}/{metric['n']}, "
              f"after B {metric['after_B_correct']}/{metric['n']}, "
              f"pass-to-fail {metric['A_to_B_regressions']}")
    print(f"\nREPORT: {report_path}", flush=True)
    if report["stage_comparison"]["A"]["after_A_accuracy"] < 0.7:
        print("WARNING: Task A acquisition was weak; do not interpret low retention as forgetting.")
    if report["stage_comparison"]["B"]["after_B_accuracy"] <= report["stage_comparison"]["B"]["after_A_accuracy"]:
        print("WARNING: No Task B acquisition improvement; tune the DEV setup before TEST.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("disjoint", "shared"), default="disjoint")
    parser.add_argument("--seed", type=int, default=43,
                        help="43 gives a new mapping and fresh LoRA initialization")
    parser.add_argument("--out", default="artifacts/experiments/sequential/disjoint_fresh_seed43")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--epochs-a", type=int, default=4)
    parser.add_argument("--epochs-b", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs_a < 1 or args.epochs_b < 1 or args.max_new_tokens < 1:
        parser.error("Epoch counts and max-new-tokens must be positive.")
    run(args)


if __name__ == "__main__":
    main()
