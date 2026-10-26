"""Prepare reproducible BANKING77 subsets and independent protected evals.

Uses original BANKING77 CSV files (CC BY 4.0), downloaded by user.
No split mixing, no train/test overlap after normalized-text de-duplication.
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalized(text):
    return re.sub(r"\s+", " ", text.strip()).casefold()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"text", "category"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path} needs text,category columns")
        return [{"text": r["text"].strip(), "category": r["category"].strip()}
                for r in reader if r.get("text") and r.get("category")]


def dedupe(rows, already_seen=None):
    seen = set(already_seen or ())
    result = []
    for row in rows:
        key = normalized(row["text"])
        if key and key not in seen:
            result.append(row)
            seen.add(key)
    return result, seen


def balanced_sample(rows, size, rng):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["category"]].append(row)
    if size < 1 or size > len(rows):
        raise ValueError(f"Request {size} not possible from {len(rows)} unique rows")
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = sorted(buckets)
    chosen = []
    while len(chosen) < size:
        progressed = False
        for label in labels:
            if buckets[label] and len(chosen) < size:
                chosen.append(buckets[label].pop())
                progressed = True
        if not progressed:
            raise ValueError("Insufficient records")
    rng.shuffle(chosen)
    return chosen


def prompt(text):
    return ("Classify the following banking support message into its most specific "
            "intent code. Reply with only the intent code, without explanation.\n"
            "Customer: " + text + "\nIntent code:")


def protected_cases(size, seed):
    if size < 4:
        raise ValueError("Protected evaluation needs at least 4 cases")
    rng = random.Random(seed + 10001)
    counts = [int(size * .4), int(size * .25), int(size * .2)]
    counts.append(size - sum(counts))
    output = []

    # Simple arithmetic that the base model should often know (verify empirically).
    arithmetic_pool = []
    for a in range(5, 45):
        for b in range(2, 29):
            arithmetic_pool.append((f"What is {a} plus {b}? Reply with only the number.", str(a + b)))
            if a >= b:
                arithmetic_pool.append((f"What is {a} minus {b}? Reply with only the number.", str(a - b)))
    for i, (question, answer) in enumerate(rng.sample(arithmetic_pool, counts[0]), 1):
        output.append({"id": f"P-AR-{i:04d}", "group": "protected", "category": "Arithmetic",
                       "prompt": question, "expected": answer, "validator": "exact"})

    # Strict literal copying; deliberately case-sensitive.
    for i in range(1, counts[1] + 1):
        token = f"XR-{rng.randrange(10000,99999)}-{chr(rng.randrange(65,91))}"
        output.append({"id": f"P-LIT-{i:04d}", "group": "protected",
                       "category": "Literal formatting",
                       "prompt": f"Reply with exactly this text and nothing else: {token}",
                       "expected": token, "validator": "exact"})

    words = ["beacon", "river", "planet", "anchor", "harbor", "garden", "copper",
             "velvet", "summit", "yellow", "bridge", "canyon", "orbit", "forest"]
    for i in range(1, counts[2] + 1):
        token = f"{rng.choice(words)}{rng.randrange(11,999)}"
        output.append({"id": f"P-CASE-{i:04d}", "group": "protected",
                       "category": "Case transformation",
                       "prompt": f"Write the uppercase version of {token}. Output only the answer.",
                       "expected": token.upper(), "validator": "exact"})

    for i in range(1, counts[3] + 1):
        num = rng.randrange(0, 100)
        output.append({"id": f"P-JSON-{i:04d}", "group": "protected",
                       "category": "Strict JSON formatting",
                       "prompt": f'Return ONLY a JSON object with exactly two keys: "count" equal to {num} (integer), and "ok" equal to true (boolean). No Markdown.',
                       "expected": {"count": num, "ok": True}, "validator": "json_fields"})
    return output


def save_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", default="artifacts/datasets/banking77/train.csv")
    p.add_argument("--test-csv", default="artifacts/datasets/banking77/test.csv")
    p.add_argument("--train-size", type=int, default=2700)
    p.add_argument("--eval-size", type=int, default=400)
    p.add_argument("--protected-size", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="artifacts/experiments/banking77")
    args = p.parse_args()
    rng = random.Random(args.seed)
    original_train, original_test = read_csv(args.train_csv), read_csv(args.test_csv)
    train, train_keys = dedupe(original_train)
    # Exclude any duplicate source training text from the separate test split.
    test, _ = dedupe(original_test, train_keys)
    labels = sorted({r["category"] for r in train})
    if set(labels) != {r["category"] for r in test}:
        raise ValueError("Train/test label coverage differs")
    if args.train_size < len(labels) or args.eval_size < len(labels):
        raise ValueError(f"Choose train/eval sizes >= {len(labels)} to cover all labels")
    training = balanced_sample(train, args.train_size, rng)
    evaluation = balanced_sample(test, args.eval_size, rng)
    train_prompts = [{"prompt": prompt(r["text"]), "response": r["category"]} for r in training]
    eval_prompts = [{"id": f"T-{i:05d}", "group": "target", "category": "BANKING77 intent",
                     "prompt": prompt(r["text"]), "expected": r["category"], "validator": "exact"}
                    for i, r in enumerate(evaluation, 1)]
    protected = protected_cases(args.protected_size, args.seed)
    dest = Path(args.output)
    save_jsonl(dest / "train.jsonl", train_prompts)
    save_jsonl(dest / "target_eval.jsonl", eval_prompts)
    save_jsonl(dest / "protected_eval.jsonl", protected)
    save_jsonl(dest / "combined_eval.jsonl", eval_prompts + protected)
    manifest = {
        "source": "PolyAI BANKING77 (Casanueva et al. 2020), CC BY 4.0",
        "csv_sources": {
            "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
            "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        },
        "sha256_csv": {"train": sha256(args.train_csv), "test": sha256(args.test_csv)},
        "seed": args.seed, "train_count": len(train_prompts),
        "target_eval_count": len(eval_prompts), "protected_eval_count": len(protected),
        "intent_labels": labels,
        "sampled_train_by_label": dict(sorted(Counter(r["category"] for r in training).items())),
        "sampled_test_by_label": dict(sorted(Counter(r["category"] for r in evaluation).items())),
        "note": "The protected suite is synthetic, held out of training; verify baseline competence before claiming forgetting.",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared: {len(training)} real train / {len(evaluation)} real target eval / {len(protected)} synthetic protected eval")
    print(f"Labels: {len(labels)}; files in {dest}")
    print("Do not call these results evidence of forgetting until paired model inference completes.")


if __name__ == "__main__":
    main()
