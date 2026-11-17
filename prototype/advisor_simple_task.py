"""Prepare a simple, explicitly scoped continual-learning experiment.

No downloads. Synthetic three-class routing teaches a deliberately arbitrary
label mapping; protected tasks are instructionally distinct from routing.
Every generation seed, template family and split is recorded.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

LABEL_TO_CUES = {
    "M7": ("invoice", "refund", "charge", "payment"),
    "Q4": ("shipment", "tracking", "parcel", "delivery"),
    "R2": ("password", "login", "profile", "verification"),
}
CODES = "M7, Q4, R2"
PROMPT = ("You are routing a service ticket. Reply with only one of these three "
          "possible codes: M7, Q4, R2. No explanation.\nTicket: {ticket}\nCode:")
CONTROL_KEY = ("Routing key: invoices, refunds, charges, and payments mean M7; "
               "shipments, tracking, parcels, and deliveries mean Q4; "
               "passwords, logins, profiles, and verifications mean R2.\n")

TRAIN_TEMPLATES = (
    "Case {n}: Please help with my {cue}.",
    "Customer reference {n}. My question concerns {cue}.",
)
TEST_TEMPLATES = (
    "Regarding ticket {n}, I have an issue with {cue}.",
    "Support request {n}: assistance requested for {cue}.",
)

def _save(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def generate(seed=42, per_class_train=180, per_class_test=40, protected_per_kind=30):
    if min(per_class_train, per_class_test, protected_per_kind) < 1:
        raise ValueError("Every requested sample size must be positive")
    rng = random.Random(seed)
    train, target, with_key = [], [], []
    used_prompts = set()

    def insert_unique(prompt):
        if prompt in used_prompts:
            raise ValueError(f"Duplicate generated prompt: {prompt}")
        used_prompts.add(prompt)

    for label, cues in LABEL_TO_CUES.items():
        for split, n_total, templates in (
            ("train", per_class_train, TRAIN_TEMPLATES),
            ("test", per_class_test, TEST_TEMPLATES),
        ):
            for i in range(n_total):
                cue = cues[i % len(cues)]
                ticket = templates[i % len(templates)].format(
                    n=rng.randrange(100_000_000, 999_999_999), cue=cue)
                prompt = PROMPT.format(ticket=ticket)
                insert_unique(prompt)
                if split == "train":
                    train.append({"prompt": prompt, "response": label})
                else:
                    rid = f"T-{label}-{i:04d}"
                    target.append({"id":rid, "group":"target",
                                   "category":"3-code routing", "prompt":prompt,
                                   "expected":label, "validator":"exact",
                                   "task_relation":"new_task"})
                    # Same ticket, but give the base model the *actual* mapping.
                    # This is a diagnostic for whether it can perform the task
                    # when the ontology is explicitly taught in the prompt.
                    with_key.append({"id":f"C-{label}-{i:04d}",
                        "group":"target_control", "category":"Routing with key",
                        "prompt": CONTROL_KEY + prompt,
                        "expected":label, "validator":"exact",
                        "task_relation":"new_task_with_key"})
    protected_reference, protected_eval = [], []
    words = ("planet", "cedar", "garden", "bridge", "anchor", "velvet")
    for split in ("reference", "evaluation"):
        dest = protected_reference if split == "reference" else protected_eval
        # This split uses distinct prompt templates as well as random values.
        for i in range(protected_per_kind):
            a, b = rng.randrange(7, 45), rng.randrange(2, 18)
            if split == "reference":
                aprompt = f"What is {a} plus {b}? Reply with just the number."
            else:
                aprompt = f"Calculate {a} + {b}. Respond with digits only."
            token = f"RX-{rng.randrange(100000, 999999)}-{rng.choice('ABCDEFGH')}"
            lprompt = (f"Repeat this exact code with no other text: {token}" if split == "reference"
                       else f"Echo precisely the following identifier: {token}")
            phrase = f"{rng.choice(words)}{rng.randrange(100,999)}"
            cprompt = (f"Write {phrase} in uppercase, with no explanation." if split == "reference"
                       else f"Convert this identifier to all capital letters: {phrase}. Output only the result.")
            number = rng.randrange(100, 999)
            jprompt = (f"Write only a JSON object with 'value' as integer {number} and 'ok' as true." if split == "reference"
                       else f"Return just JSON containing exactly two keys: 'value'={number} (integer), 'ok'=true (boolean).")
            # An intentionally overlapping-vocabulary control: the ticket word
            # appears as data, but the instruction is COPYING, not classification.
            cue = rng.choice(sum((list(v) for v in LABEL_TO_CUES.values()), []))
            marker = f"{cue}-{rng.randrange(10000,99999)}"
            sprompt = (f"Copy the following string exactly, not a classification code: {marker}" if split == "reference"
                       else f"Echo this payload exactly as written; do not route it: {marker}")
            specs = [
                ("Arithmetic", aprompt, str(a+b), "exact", "orthogonal"),
                ("Literal copy", lprompt, token, "exact", "orthogonal"),
                ("Case conversion", cprompt, phrase.upper(), "exact", "orthogonal"),
                ("JSON formatting", jprompt, {"value":number, "ok":True}, "json_fields", "orthogonal"),
                ("Task-switch copying", sprompt, marker, "exact", "overlapping_vocabulary"),
            ]
            for k, (category, prompt, expected, validator, relation) in enumerate(specs):
                insert_unique(prompt)
                dest.append({"id": f"P-{split[0].upper()}-{i:04d}-{k}",
                             "group":"protected", "category":category,
                             "prompt":prompt, "expected":expected,
                             "validator":validator, "task_relation":relation,
                             "split":split})
    rng.shuffle(train)
    rng.shuffle(target)
    # Preserve matched order between target and control for ease of audit.
    control_by_id = {x["id"][2:]:x for x in with_key}
    with_key = [control_by_id[x["id"][2:]] for x in target]
    rng.shuffle(protected_reference)
    rng.shuffle(protected_eval)
    return train, target, with_key, protected_reference, protected_eval

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artifacts/experiments/advisor_simple")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-per-class", type=int, default=180)
    p.add_argument("--test-per-class", type=int, default=40)
    p.add_argument("--protected-per-kind", type=int, default=30)
    args = p.parse_args()
    dest = Path(args.out)
    files = {
        "train.jsonl": None, "target_eval.jsonl": None,
        "target_with_key_eval.jsonl":None,
        "protected_reference.jsonl":None, "protected_eval.jsonl":None,
        "combined_eval.jsonl":None,
    }
    if any((dest/name).exists() for name in files):
        p.error("Refusing to overwrite an existing experimental dataset; use another --out")
    data = generate(args.seed,args.train_per_class,args.test_per_class,args.protected_per_kind)
    for name,rows in zip(list(files)[:5], data):
        files[name] = rows
    files["combined_eval.jsonl"] = data[1]+data[2]+data[4]
    dest.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        _save(dest/name, rows)
    (dest/"manifest.json").write_text(json.dumps({
        "seed":args.seed, "mapping":LABEL_TO_CUES,
        "n_by_file":{name:len(rows) for name,rows in files.items()},
        "sha256":{name:_sha(dest/name) for name in files},
        "design":"Train and test routing templates disjoint; protected reference and evaluation templates disjoint. Routing word overlap only in task-switch copying. No claims of independent natural-language generalization.",
        "caution":"Base is shown the three valid codes but NOT their arbitrary mapping on the main target task. target_with_key_eval explicitly discloses the mapping as a base-competence control. Keep protected_reference outside final evaluation."
    },indent=2)+"\n")
    print(f"Generated {len(data[0])} training, {len(data[1])} target test, "
          f"{len(data[3])} protected reference and {len(data[4])} protected eval cases at {dest}")

if __name__=="__main__":
    main()
