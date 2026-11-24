"""Turn a sequential A->B comparison into a ModelGuard regression report.

The model after stage A plays the "base" and the model after stage B is the
candidate release. Task A prompts are the protected behaviors (must not break);
Task B prompts are the target task (should improve). The output is the same
HTML/JSON report and release gate used for the BANKING77 run, so a plain
A->B fine-tune and a replay run can be compared side by side in the demo.

    python prototype/sequential_report.py \
      --after-a <root>/probe_after_A.jsonl --after-b <root>/probe_after_B_r00.jsonl \
      --html artifacts/demo/plain.html --json artifacts/demo/plain.json --gate mcnemar --ci
"""
import argparse
import json
from pathlib import Path

try:
    from prototype.report import assess, render
    from prototype.sequential_score import read_rows
except ModuleNotFoundError:
    from report import assess, render
    from sequential_score import read_rows


def to_report_rows(after_a, after_b):
    """Pair stage-A and stage-B predictions into report rows."""
    if set(after_a) != set(after_b):
        raise ValueError("Stage A and stage B must cover the same prompt IDs")
    rows = []
    for id_ in sorted(after_a):
        a, b = after_a[id_], after_b[id_]
        for field in ("prompt", "expected", "mode"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Mismatched predictions at {id_}: {field}")
        if a["mode"] not in ("A", "B"):
            raise ValueError(f"Unknown mode at {id_}: {a['mode']}")
        protected = a["mode"] == "A"
        rows.append({
            "id": id_,
            "group": "protected" if protected else "target",
            "category": "Task A (must keep)" if protected else "Task B (new task)",
            "prompt": a["prompt"], "expected": a["expected"],
            "validator": a.get("validator", "exact"),
            "base_response": a["fine_response"],
            "fine_response": b["fine_response"],
            "source": "model_inference" if a.get("source") == b.get("source") == "model_inference"
                      else "unverified",
        })
    return rows


def build(after_a_path, after_b_path, gate="mcnemar", alpha=0.05, max_drop=0.10):
    rows = to_report_rows(read_rows(after_a_path), read_rows(after_b_path))
    return assess(rows, max_protected_drop=max_drop, gate=gate, alpha=alpha)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--after-a", required=True)
    p.add_argument("--after-b", required=True)
    p.add_argument("--html", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--gate", choices=("threshold", "mcnemar"), default="mcnemar")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-protected-drop", type=float, default=0.10)
    p.add_argument("--ci", action="store_true", help="Exit 2 if blocked")
    args = p.parse_args()
    result = build(args.after_a, args.after_b, args.gate, args.alpha, args.max_protected_drop)
    html = render(result).replace("BASE: ", "AFTER TASK A: ").replace("FINE-TUNED: ", "AFTER TASK B: ")
    for path, contents in ((args.html, html), (args.json, json.dumps(result, indent=2))):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(contents + "\n")
    pt = result["paired_test"]
    status = "PASS" if result["gate_passed"] else "BLOCK"
    print(f"Gate {status}: {pt['regressions']} Task A regressions vs {pt['gains']} gains "
          f"(p = {pt['p_value']:.3g}); HTML {args.html}")
    if args.ci and not result["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
