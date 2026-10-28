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
    """Build a deterministic, duplicate-free set of four protected task families.

    The case and JSON families have finite prompt spaces. Sample them without
    replacement rather than relying on repeated random draws (which collide).
    """
    if per_family < 1:
        raise ValueError("per_family must be positive")
    nouns = ["silver", "ocean", "cedar", "lunar", "fresh", "quiet", "open", "north", "amber", "stone"]
    statuses = ["ready", "pending", "done"]
    json_capacity = len(statuses) * 100
    phrase_capacity = len(nouns) ** 3
    if per_family > min(json_capacity, phrase_capacity):
        raise ValueError(
            f"per_family={per_family} exceeds unique benchmark capacity "
            f"({json_capacity} JSON prompts and {phrase_capacity} case prompts). "
            "Reduce --per-family."
        )

    rng = random.Random(seed)
    rows = []
    # These combinations are sampled WITHOUT replacement. Each retains the
    # same task wording, numeric range, vocabulary, and status values as v2.3.
    phrase_codes = rng.sample(range(phrase_capacity), per_family)
    json_codes = rng.sample(range(json_capacity), per_family)
    seen_arithmetic = set()
    seen_literals = set()

    def unique_draw(draw, seen, category):
        for _ in range(10000):
            value = draw()
            if value not in seen:
                seen.add(value)
                return value
        raise ValueError(f"Could not sample enough unique {category} examples")

    for i in range(per_family):
        a, b, c = unique_draw(
            lambda: (rng.randrange(100, 900),
                     rng.randrange(10, 70),
                     rng.randrange(1, 20)),
            seen_arithmetic, "arithmetic"
        )
        rows.append({
            "id": f"H-AR-{i:04d}", "group": "protected", "category": "Arithmetic",
            "template_family": FAMILIES["Arithmetic"],
            "prompt": f"Compute ({a} - {b}) + {c}. Give the result as digits only.",
            "expected": str(a - b + c), "validator": "exact",
        })

        number, noun = unique_draw(
            lambda: (rng.randrange(10000, 99999), rng.choice(nouns)),
            seen_literals, "literal"
        )
        token = f"tag_{number}_{noun}"
        rows.append({
            "id": f"H-LIT-{i:04d}", "group": "protected", "category": "Literal formatting",
            "template_family": FAMILIES["Literal formatting"],
            "prompt": f"Extract the exact characters between the markers START and END, excluding the markers. Output nothing else. START{token}END",
            "expected": token, "validator": "exact",
        })

        phrase_code = phrase_codes[i]
        word_count = len(nouns)
        words = [
            nouns[phrase_code // (word_count * word_count)],
            nouns[(phrase_code // word_count) % word_count],
            nouns[phrase_code % word_count],
        ]
        phrase = " ".join(words)
        rows.append({
            "id": f"H-CASE-{i:04d}", "group": "protected", "category": "Case transformation",
            "template_family": FAMILIES["Case transformation"],
            "prompt": f"Transform the entire phrase to CAPITAL LETTERS, preserving spaces: '{phrase}'. Answer with the transformed phrase alone.",
            "expected": phrase.upper(), "validator": "exact",
        })

        json_code = json_codes[i]
        n = json_code % 100
        status = statuses[json_code // 100]
        rows.append({
            "id": f"H-JSON-{i:04d}", "group": "protected", "category": "Strict JSON formatting",
            "template_family": FAMILIES["Strict JSON formatting"],
            "prompt": f"Produce one compact JSON object (no fences), with exactly these fields in any order: status as the string '{status}' and amount as integer {n}.",
            "expected": {"status": status, "amount": n}, "validator": "json_fields",
        })

    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate benchmark IDs detected")
    if len({r["prompt"] for r in rows}) != len(rows):
        raise ValueError("Duplicate benchmark prompts detected")
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
