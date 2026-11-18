"""Probe mapping transfer with novel identifiers but ORIGINAL train templates.

Uses Task A or B training examples as structural templates, replacing the nonce with
a never-before-used nonce. Separates exact-prompt memorization from transfer to
new instances of the training grammar; does NOT use held-out dev prompts.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

try:
    from prototype.sequential_diagnose import item_from_prompt, template_name
except ModuleNotFoundError:
    from sequential_diagnose import item_from_prompt, template_name

NONCE_PATTERN = re.compile(r"TR-[0-9]{8}")


def prepare(train_path, manifest_path, out_path, per_item=3, seed=43, mode="A"):
    if mode not in ("A", "B"):
        raise ValueError("mode must be A or B")
    if per_item < 1:
        raise ValueError("per_item must be positive")
    records = [json.loads(x) for x in Path(train_path).read_text().splitlines() if x.strip()]
    manifest = json.loads(Path(manifest_path).read_text())
    mapping = manifest[f"task_{mode}_mapping"]
    by_item_and_template = defaultdict(list)
    old_nonces = set()
    old_prompts = {r["prompt"] for r in records}
    for record in records:
        item = item_from_prompt(record["prompt"])
        if f"MODE={mode}" not in record["prompt"] and f"MODE {mode}" not in record["prompt"]:
            raise ValueError(f"Training prompt has unexpected mode: {record['prompt']}")
        if record["response"] != mapping.get(item):
            raise ValueError("Unexpected item mapping in training data")
        match = NONCE_PATTERN.search(record["prompt"])
        if not match or len(NONCE_PATTERN.findall(record["prompt"])) != 1:
            raise ValueError("Expected exactly one training nonce per prompt")
        old_nonces.add(match.group())
        by_item_and_template[(item, template_name(record["prompt"]))].append(record)
    rng = random.Random(seed)
    probes = []
    used_nonces = set()
    for item in sorted(mapping):
        groups = sorted(
            (template, by_item_and_template[(item,template)])
            for (item_name, template) in by_item_and_template if item_name == item
        )
        if len(groups) < per_item:
            raise ValueError(f"Need at least {per_item} training templates for item {item}")
        for i, (template, source_records) in enumerate(groups[:per_item]):
            source = rng.choice(source_records)
            while True:
                new_nonce = f"TR-{rng.randrange(10_000_000, 99_999_999)}"
                if new_nonce not in old_nonces and new_nonce not in used_nonces:
                    used_nonces.add(new_nonce)
                    break
            new_prompt = NONCE_PATTERN.sub(new_nonce, source["prompt"])
            if new_prompt in old_prompts:
                raise AssertionError("New prompt unexpectedly in training")
            probes.append({
                "id":f"{mode}-novel-nonce-{item}-{i:03d}",
                "mode":mode, "group":f"{mode}_near_train_diagnostic",
                "category":f"Task {mode} novel nonce",
                "item":item,
                "prompt":new_prompt,
                "expected":mapping[item],
                "validator":"exact",
                "template":template,
                "note":"Fresh nonce, same training grammar; never used for training."
            })
    output = Path(out_path)
    if output.exists():
        raise FileExistsError(f"Refusing overwrite of {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r)+"\n" for r in probes),encoding="utf-8")
    print(f"Wrote {len(probes)} unseen-nonce training-style probes to {output}")
    return probes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("A", "B"),default="A")
    parser.add_argument("--train",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--per-item",type=int,default=3)
    parser.add_argument("--seed",type=int,default=43)
    args=parser.parse_args()
    prepare(args.train,args.manifest,args.out,args.per_item,args.seed,args.mode)


if __name__=="__main__":
    main()
