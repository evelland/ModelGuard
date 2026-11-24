"""Replay baseline for the sequential A->B study.

Stage B normally trains the continued adapter on Task B examples only. With
replay, a fraction of the stage-B training budget is replaced by previously
seen Task A *training* examples. The total number of stage-B examples, and so
the optimizer budget, stays identical to the no-replay run, so retention can be
compared at a matched update count.

Replay rows are drawn only from Task A training data (train_A.jsonl). They are
never taken from dev/test/probe prompts or from the reserved reference_A set.
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    from prototype.sequential_diagnose import item_from_prompt
except ModuleNotFoundError:
    from sequential_diagnose import item_from_prompt


def mode_of(prompt):
    for mode in ("A", "B"):
        if f"MODE={mode}" in prompt or f"MODE {mode}" in prompt or f"mode {mode}" in prompt:
            return mode
    raise ValueError(f"Cannot identify MODE in prompt: {prompt}")


def _stratified_pick(rows, count, rng):
    """Pick `count` rows spread as evenly as possible across items."""
    by_item = defaultdict(list)
    for row in rows:
        by_item[item_from_prompt(row["prompt"])].append(row)
    for bucket in by_item.values():
        rng.shuffle(bucket)
    items = sorted(by_item)
    rng.shuffle(items)
    picked, depth = [], 0
    while len(picked) < count:
        progressed = False
        for item in items:
            if depth < len(by_item[item]):
                picked.append(by_item[item][depth])
                progressed = True
                if len(picked) == count:
                    break
        if not progressed:
            raise ValueError(f"Requested {count} rows but only {len(rows)} available")
        depth += 1
    return picked


def build_replay_mix(b_rows, a_rows, fraction, seed):
    """Return (mixed_rows, info). Keeps len(mixed_rows) == len(b_rows)."""
    if not 0 <= fraction < 1:
        raise ValueError("Replay fraction must be in [0, 1)")
    if not b_rows:
        raise ValueError("Stage B training data is empty")
    for row in b_rows:
        if mode_of(row["prompt"]) != "B":
            raise ValueError(f"Non-B row in stage B training data: {row['prompt']}")
    total = len(b_rows)
    n_replay = round(total * fraction)
    info = {"replay_fraction": fraction, "total_examples": total,
            "replay_examples": n_replay, "task_B_examples": total - n_replay,
            "replay_seed": seed}
    if n_replay == 0:
        return list(b_rows), info
    if not a_rows:
        raise ValueError("Replay requested but no Task A rows supplied")
    for row in a_rows:
        if mode_of(row["prompt"]) != "A":
            raise ValueError(f"Replay row is not Task A: {row['prompt']}")
    b_prompts = {row["prompt"] for row in b_rows}
    if any(row["prompt"] in b_prompts for row in a_rows):
        raise ValueError("Replay rows overlap stage B prompts")
    rng = random.Random(seed)
    replay = _stratified_pick(a_rows, n_replay, rng)
    kept_b = _stratified_pick(b_rows, total - n_replay, rng)
    mixed = kept_b + replay
    rng.shuffle(mixed)
    info["replay_items"] = sorted({item_from_prompt(r["prompt"]) for r in replay})
    info["mixed_sha256"] = hashlib.sha256(
        "".join(json.dumps(r, sort_keys=True) for r in mixed).encode()).hexdigest()
    return mixed, info


def forbid_overlap(rows, *paths):
    """Raise if any training row prompt appears in the given evaluation files."""
    prompts = {row["prompt"] for row in rows}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["prompt"] in prompts:
                raise ValueError(f"Training prompt leaks into evaluation file {path}")
