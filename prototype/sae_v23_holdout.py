"""Generate a *new-template* protected benchmark for external, frozen evaluation.

Distinct wording/templates from prepare_banking77.protected_cases. Generation is
deterministic; do NOT choose model/SAE hyperparameters after observing results.
"""
import argparse
import json
import random
from pathlib import Path

FAMILIES = {
    "Arithmetic": "two_step_arithmetic",
    "Literal formatting": "quoted_payload_copy",
    "Case transformation": "uppercase_spaced_words",
    "Strict JSON formatting": "strict_json_status",
}

def generate(per_family=100, seed=20261101):
    if per_family < 1:
        raise ValueError("per_family must be positive")
    rng = random.Random(seed)
    rows = []
    nouns = ["silver", "ocean", "cedar", "lunar", "fresh", "quiet", "open", "north", "amber", "stone"]
    for i in range(per_family):
        a, b, c = rng.randrange(100, 900), rng.randrange(10, 70), rng.randrange(1, 20)
        rows.append({
            "id": f"H-AR-{i:04d}", "group": "protected", "category": "Arithmetic",
            "template_family": FAMILIES["Arithmetic"],
            "prompt": f"Compute ({a} - {b}) + {c}. Give the result as digits only.",
            "expected": str(a-b+c), "validator": "exact",
        })
        token = f"tag_{rng.randrange(10000,99999)}_{rng.choice(nouns)}"
        rows.append({
            "id": f"H-LIT-{i:04d}", "group": "protected", "category": "Literal formatting",
            "template_family": FAMILIES["Literal formatting"],
            "prompt": f"Extract the exact characters between the markers START and END, excluding the markers. Output nothing else. START{token}END",
            "expected": token, "validator": "exact",
        })
        words = [rng.choice(nouns) for _ in range(3)]
        phrase = " ".join(words)
        rows.append({
            "id": f"H-CASE-{i:04d}", "group": "protected", "category": "Case transformation",
            "template_family": FAMILIES["Case transformation"],
            "prompt": f"Transform the entire phrase to CAPITAL LETTERS, preserving spaces: '{phrase}'. Answer with the transformed phrase alone.",
            "expected": phrase.upper(), "validator": "exact",
        })
        n = rng.randrange(0,100)
        status = rng.choice(["ready", "pending", "done"])
        rows.append({
            "id": f"H-JSON-{i:04d}", "group": "protected", "category": "Strict JSON formatting",
            "template_family": FAMILIES["Strict JSON formatting"],
            "prompt": f"Produce one compact JSON object (no fences), with exactly these fields in any order: status as the string '{status}' and amount as integer {n}.",
            "expected": {"status":status,"amount":n}, "validator": "json_fields",
        })
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["prompt"] for r in rows}) == len(rows)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-family",type=int,default=100)
    parser.add_argument("--seed",type=int,default=20261101)
    parser.add_argument("--out",default="artifacts/experiments/banking77/sae_v23/new_template_protected.jsonl")
    args=parser.parse_args()
    rows=generate(args.per_family,args.seed)
    dest=Path(args.out)
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite frozen benchmark {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    print(f"Saved {len(rows)} cases across {len(FAMILIES)} families to {dest}")
    print("Freeze model, SAE configurations, and score directions BEFORE evaluating.")

if __name__=="__main__":
    main()
