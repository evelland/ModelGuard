"""Diagnose whether sequential Task A underlearning is memorization or transfer.

Prepare a small balanced sample of the ACTUAL training prompts in eval format,
then run existing evaluate_pair.py against the after-A adapter. Also summarize
existing dev predictions by expected code, item and prompt template.

Training-prompt probes are diagnostics ONLY; do not treat them as holdout test.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

VALID_CODES = ("K2", "R7", "T4", "V9")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def item_from_prompt(prompt):
    matched = re.search(r"\bITEM[ =]([a-z]+)\b", prompt)
    if not matched:
        raise ValueError(f"Cannot identify ITEM in prompt: {prompt}")
    return matched.group(1)


def prepare_train_probe(train_path, manifest_path, output, *, per_item=3, seed=43):
    if per_item < 1:
        raise ValueError("--per-item must be >= 1")
    data = read_jsonl(train_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mapping = manifest["task_A_mapping"]
    buckets = defaultdict(list)
    for record in data:
        item = item_from_prompt(record["prompt"])
        if item not in mapping or mapping[item] != record["response"]:
            raise ValueError(f"Training record violates Task A mapping: {item}")
        buckets[item].append(record)
    if set(buckets) != set(mapping):
        raise ValueError("Training file does not cover exactly the Task A items")
    rng = random.Random(seed)
    probe = []
    for item in sorted(buckets):
        pool = buckets[item]
        # Training examples cycle across three templates. The per-item=3
        # default samples one from each template to avoid template bias.
        by_template = defaultdict(list)
        for example in pool:
            by_template[template_name(example["prompt"])].append(example)
        chosen = []
        for template in sorted(by_template):
            if len(chosen) >= per_item:
                break
            chosen.append(rng.choice(by_template[template]))
        remaining = [r for r in pool if r not in chosen]
        if per_item > len(pool):
            raise ValueError(f"Insufficient training records for item {item}")
        chosen += rng.sample(remaining, per_item - len(chosen))
        for i, example in enumerate(chosen):
            probe.append({"id":f"A-training-probe-{item}-{i:03d}",
                "mode":"A", "group":"A_training_diagnostic", "category":"Task A training probe",
                "item":item, "prompt":example["prompt"],
                "expected":example["response"],"validator":"exact",
                "note":"This example was in training; diagnostic only"})
    dest = Path(output)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as file:
        for record in probe:
            file.write(json.dumps(record)+"\n")
    print(f"Saved {len(probe)} diagnostic training examples to {dest}")
    return probe


def template_name(prompt):
    if prompt.startswith("MODE="):
        return "dev_pipe"
    if prompt.startswith("In mode"):
        return "dev_question"
    if prompt.startswith("Mapping exercise"):
        return "train_mapping_exercise"
    if prompt.startswith("For MODE"):
        return "train_for_mode"
    if prompt.startswith("Task MODE"):
        return "train_task_mode"
    return "other"


def summarize_predictions(path, mode="A"):
    records = [r for r in read_jsonl(path) if r.get("mode") == mode]
    if not records:
        raise ValueError(f"No predictions found for mode {mode}")
    correct=lambda r: r["fine_response"].strip() == r["expected"]
    print(f"Mode {mode}: {sum(correct(r) for r in records)}/{len(records)} exact")
    print("Expected code -> correct / total; most common generated answers:")
    for code in VALID_CODES:
        cases = [r for r in records if r["expected"] == code]
        if cases:
            print(f"  {code}: {sum(correct(r) for r in cases)}/{len(cases)}; "
                  f"{Counter(r['fine_response'] for r in cases).most_common(4)}")
    print("Per item:")
    for item in sorted({r["item"] for r in records}):
        cases = [r for r in records if r["item"] == item]
        print(f"  {item}: {sum(correct(r) for r in cases)}/{len(cases)} "
              f"(expected {cases[0]['expected']})")
    print("Prompt templates:")
    for template in sorted({template_name(r["prompt"]) for r in records}):
        cases = [r for r in records if template_name(r["prompt"]) == template]
        print(f"  {template}: {sum(correct(r) for r in cases)}/{len(cases)}")
    invalid = [r for r in records if r["fine_response"] not in VALID_CODES]
    print(f"Non-code responses: {len(invalid)}/{len(records)} "
          f"({Counter(r['fine_response'] for r in invalid).most_common(10)})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare",help="Prepare balanced train-prompt probe")
    prep.add_argument("--train",required=True)
    prep.add_argument("--manifest",required=True)
    prep.add_argument("--out",required=True)
    prep.add_argument("--per-item",type=int,default=3)
    prep.add_argument("--seed",type=int,default=43)
    report=sub.add_parser("summarize",help="Break down a paired prediction JSONL")
    report.add_argument("--predictions",required=True)
    report.add_argument("--mode",choices=("A","B"),default="A")
    a=p.parse_args()
    if a.command == "prepare":
        prepare_train_probe(a.train,a.manifest,a.out,
                            per_item=a.per_item,seed=a.seed)
    else:
        summarize_predictions(a.predictions,a.mode)


if __name__ == "__main__":
    main()
