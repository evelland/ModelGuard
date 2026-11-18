"""Measure acquisition and forgetting from paired stage-A and stage-B predictions.

The model must learn A before any decrease is called forgetting. A -> B is
a SAME-adapter sequential checkpoint transition. Strict exact is primary.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    items=[json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not items or len({x["id"] for x in items})!=len(items):
        raise ValueError("Missing examples or duplicate IDs")
    return {x["id"]:x for x in items}


def strict_code(response, expected):
    return isinstance(response,str) and response.strip() == expected


def score_pair(first, second):
    if set(first) != set(second):
        raise ValueError("Stage A and stage B must evaluate the same prompt IDs")
    groups=defaultdict(list)
    regressions=[]
    for id_ in sorted(first):
        a,b=first[id_],second[id_]
        for field in ("prompt","expected","mode","item","group"):
            if a.get(field)!=b.get(field):
                raise ValueError(f"Non-matched stage predictions at {id_}: {field}")
        if "fine_response" not in a or "fine_response" not in b:
            raise ValueError(f"Missing model response: {id_}")
        expected=a["expected"]
        pre=strict_code(a["fine_response"],expected)
        post=strict_code(b["fine_response"],expected)
        base=strict_code(a.get("base_response"),expected)
        groups[a["mode"]].append((base,pre,post))
        if a["mode"]=="A" and pre and not post:
            regressions.append({
                "id":id_, "item":a["item"],"prompt":a["prompt"],
                "expected":expected,
                "after_stage_A":a["fine_response"],
                "after_stage_B":b["fine_response"],
            })

    def summary(rows):
        n=len(rows)
        return {"n":n,
                "base_correct":sum(x[0] for x in rows),
                "after_A_correct":sum(x[1] for x in rows),
                "after_B_correct":sum(x[2] for x in rows),
                "after_A_accuracy":round(sum(x[1] for x in rows)/n,5),
                "after_B_accuracy":round(sum(x[2] for x in rows)/n,5),
                "A_to_B_regressions":sum(x[1] and not x[2] for x in rows),
                "A_to_B_gains":sum(not x[1] and x[2] for x in rows),
                "retained_previously_correct":sum(x[1] and x[2] for x in rows),
                "retention_given_A_correct":round(
                    sum(x[1] and x[2] for x in rows)/sum(x[1] for x in rows),5)
                    if any(x[1] for x in rows) else None}
    metrics={name:summary(rows) for name, rows in sorted(groups.items())}
    if set(metrics)!={"A","B"}:
        raise ValueError("Expected both A and B test examples")
    result={
        "stage_comparison":metrics,
        "Task_A_forgetting": {
            "eligible_base":metrics["A"]["after_A_correct"],
            "new_failures":metrics["A"]["A_to_B_regressions"],
            "retention":metrics["A"]["retention_given_A_correct"]},
        "Task_B_learning": {
            "B_correct_after_A":metrics["B"]["after_A_correct"],
            "B_correct_after_B":metrics["B"]["after_B_correct"]},
        "example_regressions":regressions[:20],
        "caveats":[
            "B learning must improve materially and A must first be learned; otherwise no interpretable forgetting result.",
            "The benchmark uses previously seen item identities and held-out prompt templates, not unseen vocabulary generalization.",
            "Only one seed/condition is represented. Replicate before making preservation claims.",
            "Exact match includes output format adherence. Inspect non-exact responses separately."],
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--after-a",required=True)
    p.add_argument("--after-b",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    result=score_pair(read_rows(a.after_a),read_rows(a.after_b))
    target=Path(a.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for mode,v in result["stage_comparison"].items():
        print(f"{mode}: after A={v['after_A_correct']}/{v['n']}; "
              f"after B={v['after_B_correct']}/{v['n']}; "
              f"new failures={v['A_to_B_regressions']}")
    print("Saved",target)


if __name__=="__main__":
    main()
