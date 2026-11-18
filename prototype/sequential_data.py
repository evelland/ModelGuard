"""Create a controlled two-stage forgetting experiment with explicit A/B modes.

Task A is learned first. Stage 2 teaches Task B using the SAME LoRA adapter.
The A->B transition defines forgetting; no external dataset or download needed.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

A_ITEMS = ("cedar", "maple", "birch", "elm", "willow", "fir",
           "cypress", "ash", "spruce", "pine", "oak", "alder")
B_ITEMS = ("plum", "peach", "guava", "kiwi", "mango", "lemon",
           "lime", "papaya", "apricot", "fig", "pear", "lychee")
CODES = ("K2", "R7", "T4", "V9")
TRAIN_TEMPLATES = (
    "Mapping exercise. MODE={mode}; ITEM={item}; reference={nonce}. Reply with its code only.",
    "For MODE {mode}, give the assigned code of ITEM {item}. Ticket {nonce}. No explanation.",
    "Task MODE={mode}. Input ITEM={item} (record {nonce}). Output exactly the matching code.",
)
DEV_TEMPLATES = (
    "MODE={mode} | ITEM={item} | check={nonce}. Return only its assigned code.",
    "In mode {mode}, what is the code for item {item}? Sample {nonce}. Answer only the code.",
)
TEST_TEMPLATES = (
    "Lookup request {nonce}: the item is {item} and the mode is {mode}. Output only the code.",
    "A tester asks for the code of {item} under mode {mode}; case {nonce}. Just the code.",
)
REF_TEMPLATE = "Preservation reference {nonce}: MODE={mode}, ITEM={item}. Reply with just its code."


def mappings(condition, seed):
    if condition not in ("disjoint", "shared"):
        raise ValueError("Condition must be disjoint or shared")
    rng = random.Random(seed)
    labels = list(CODES) * 3
    rng.shuffle(labels)
    a = dict(zip(A_ITEMS, labels))
    if condition == "disjoint":
        labels_b = list(CODES) * 3
        random.Random(seed + 2003).shuffle(labels_b)
        b = dict(zip(B_ITEMS, labels_b))
    else:
        # Same items but incompatible arbitrary lookup mapping under MODE=B.
        # There is still no logical conflict: MODE fully specifies the contract.
        b = {item: CODES[(CODES.index(code) + 1) % len(CODES)]
             for item, code in a.items()}
    return a, b


def generate(condition="disjoint", seed=42, examples_per_item=24,
             eval_per_item=4, reference_per_item=3):
    if min(examples_per_item, eval_per_item, reference_per_item) < 1:
        raise ValueError("All example counts must be positive")
    a, b = mappings(condition, seed)
    rng = random.Random(seed + 101)
    files = {}
    seen = set()

    def add_unique(prompt):
        if prompt in seen:
            raise ValueError("Duplicate prompt generated")
        seen.add(prompt)

    for mode, mapping in (("A", a), ("B", b)):
        training, dev, test, ref = [], [], [], []
        for item, code in mapping.items():
            for i in range(examples_per_item):
                nonce = f"TR-{rng.randrange(10_000_000, 99_999_999)}"
                prompt = TRAIN_TEMPLATES[i % len(TRAIN_TEMPLATES)].format(
                    mode=mode, item=item, nonce=nonce)
                add_unique(prompt)
                training.append({"prompt":prompt, "response":code})
            for split, templates, dest in (
                ("dev", DEV_TEMPLATES, dev),
                ("test", TEST_TEMPLATES, test),
            ):
                for i in range(eval_per_item):
                    nonce = f"{split.upper()}-{rng.randrange(10_000_000,99_999_999)}"
                    prompt = templates[i % len(templates)].format(
                        mode=mode, item=item, nonce=nonce)
                    add_unique(prompt)
                    dest.append({"id":f"{mode}-{split}-{item}-{i:03d}",
                                 "group":mode,"category":f"Task {mode}",
                                 "mode":mode,"item":item,"prompt":prompt,
                                 "expected":code,"validator":"exact"})
            if mode == "A":
                for i in range(reference_per_item):
                    nonce = f"REF-{rng.randrange(10_000_000, 99_999_999)}"
                    prompt = REF_TEMPLATE.format(mode=mode, item=item, nonce=nonce)
                    add_unique(prompt)
                    ref.append({"id":f"A-reference-{item}-{i:03d}",
                                "group":"protected_reference", "mode":"A",
                                "item":item, "prompt":prompt,
                                "expected":code, "validator":"exact"})
        rng.shuffle(training)
        rng.shuffle(dev)
        rng.shuffle(test)
        files[f"train_{mode}.jsonl"] = training
        files[f"eval_{mode}_dev.jsonl"] = dev
        files[f"eval_{mode}_test.jsonl"] = test
        if mode == "A":
            files["reference_A.jsonl"] = ref
    files["eval_combined_dev.jsonl"] = (
        files["eval_A_dev.jsonl"] + files["eval_B_dev.jsonl"])
    files["eval_combined_test.jsonl"] = (
        files["eval_A_test.jsonl"] + files["eval_B_test.jsonl"])
    manifest = {
        "condition": condition, "seed": seed,
        "task_A_mapping": a, "task_B_mapping": b,
        "counts": {k:len(v) for k,v in files.items()},
        "meaning": {
            "Task_A": "Arbitrary code lookup learned in stage 1",
            "Task_B": "Arbitrary code lookup learned in stage 2 with same adapter",
            "disjoint": "B uses separate items; old A behavior is logically independent",
            "shared": "B uses same items with different codes, fully disambiguated by MODE",
            "reference_A": "Reserved for future preservation/replay, never ordinary B training",
            "eval": "Known items with prompt templates disjoint from training",
        },
    }
    return files, manifest


def write_files(destination, files, manifest):
    root = Path(destination)
    expected = [root / name for name in files]
    expected.append(root / "manifest.json")
    if any(path.exists() for path in expected):
        raise FileExistsError(f"Refusing overwrite of experimental outputs in {root}")
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in files.items():
        path = root / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["sha256"] = hashes
    (root / "manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=("disjoint","shared"), default="disjoint")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--examples-per-item",type=int,default=24)
    p.add_argument("--eval-per-item",type=int,default=4)
    p.add_argument("--reference-per-item",type=int,default=3)
    p.add_argument("--out",default=None)
    args=p.parse_args()
    root=args.out or f"artifacts/experiments/sequential/{args.condition}"
    files,manifest=generate(args.condition,args.seed,args.examples_per_item,
                            args.eval_per_item,args.reference_per_item)
    write_files(root,files,manifest)
    print(f"Generated {args.condition}: Task A/B {len(files['train_A.jsonl'])}/"
          f"{len(files['train_B.jsonl'])} training prompts, "
          f"{len(files['eval_A_dev.jsonl'])} examples per mode per eval split.")
    print(f"Manifest and frozen dataset saved at {root}")


if __name__ == "__main__":
    main()
