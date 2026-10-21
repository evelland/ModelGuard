"""Generate deterministic regression metrics, HTML report, and CI gate from JSONL."""
import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(x):
    return re.sub(r'\s+', ' ', str(x).strip()).casefold()


def score(case, response):
    if response is None:
        return False
    kind = case.get('validator', 'exact')
    if kind == 'exact':
        return normalize(response) == normalize(case['expected'])
    if kind == 'contains':
        return normalize(case['expected']) in normalize(response)
    if kind == 'json_fields':
        try:
            parsed = json.loads(response)
        except (TypeError, ValueError):
            return False
        return isinstance(parsed, dict) and all(parsed.get(k) == v for k, v in case['expected'].items())
    raise ValueError(f'Unknown validator {kind!r}')


def assess(rows, max_protected_drop=0.10, min_target_gain=0.0):
    if not (0 <= max_protected_drop <= 1 and -1 <= min_target_gain <= 1):
        raise ValueError('Invalid policy thresholds')
    ids, groups, scored = set(), defaultdict(list), []
    required = {'id', 'group', 'category', 'prompt', 'expected', 'base_response', 'fine_response'}
    for raw in rows:
        if required - raw.keys():
            raise ValueError(f"Missing fields in {raw.get('id', '?')}: {sorted(required-raw.keys())}")
        if raw['id'] in ids or raw['group'] not in ('protected', 'target'):
            raise ValueError(f"Duplicate id or invalid group: {raw['id']}")
        ids.add(raw['id'])
        row = dict(raw, base_pass=score(raw, raw['base_response']), fine_pass=score(raw, raw['fine_response']))
        groups[row['group']].append(row)
        scored.append(row)
    if not groups['protected'] or not groups['target']:
        raise ValueError('At least one protected AND target example required')

    def metric(items):
        n = len(items)
        base = sum(r['base_pass'] for r in items) / n
        fine = sum(r['fine_pass'] for r in items) / n
        return {'n': n, 'base': base, 'fine': fine, 'delta': fine-base}

    metrics = {g: metric(items) for g, items in groups.items()}
    category_rows = defaultdict(list)
    for row in scored:
        category_rows[(row['group'], row['category'])].append(row)
    categories = [{'group': group, 'category': category, **metric(items)} for (group, category), items in sorted(category_rows.items())]
    regressions = [r for r in scored if r['group'] == 'protected' and r['base_pass'] and not r['fine_pass']]
    rules = {
        'protected_retention': metrics['protected']['delta'] >= -max_protected_drop-1e-12,
        'target_gain': metrics['target']['delta'] >= min_target_gain-1e-12,
    }
    return {
        'source': 'model_inference' if all(r.get('source') == 'model_inference' for r in scored) else 'unverified_or_illustrative',
        'metrics': metrics, 'categories': categories, 'regressions': regressions,
        'rules': rules, 'gate_passed': all(rules.values()),
        'policy': {'max_protected_drop': max_protected_drop, 'min_target_gain': min_target_gain},
        'rows': scored,
    }


def render(report):
    h = lambda value: html.escape(str(value), quote=True)
    fmt = lambda x: f'{x*100:.1f}%'
    delta = lambda x: f'{x*100:+.1f} pp'
    protected, target = report['metrics']['protected'], report['metrics']['target']
    status = 'PASS' if report['gate_passed'] else 'BLOCK'
    columns = ''.join(f'<tr><td>{h(x["group"])}</td><td>{h(x["category"])}</td><td>{x["n"]}</td><td>{fmt(x["base"])}</td><td>{fmt(x["fine"])}</td><td>{delta(x["delta"])}</td></tr>' for x in report['categories'])
    failures = ''.join(f'<section><h3>{h(x["id"])} — {h(x["category"])}</h3><p>{h(x["prompt"])}</p><div class="pair"><pre>BASE: {h(x["base_response"])}</pre><pre>FINE-TUNED: {h(x["fine_response"])}</pre></div></section>' for x in report['regressions'])
    if not failures:
        failures = '<p>No base-pass → candidate-fail transitions in the protected suite.</p>'
    banner = 'ACTUAL MODEL INFERENCE (verify evaluation design)' if report['source'] == 'model_inference' else 'UNVERIFIED OR ILLUSTRATIVE OUTPUTS — NOT EXPERIMENTAL EVIDENCE'
    rules = ''.join(f'<li>{h(key)}: {"PASS" if passed else "FAIL"}</li>' for key, passed in report['rules'].items())
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ModelGuard | {status}</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;background:#101820;color:#ebf3f5;margin:0}}main{{max-width:1000px;margin:auto;padding:42px 24px}}h1{{font-size:46px;margin:0}}.sub{{color:#a9b9c0}}.banner{{background:#353320;border:1px solid #a78a58;padding:12px;margin:24px 0;font-weight:650}}.status{{font-size:24px;color:{'#8ed9b2' if report['gate_passed'] else '#ff9c9c'}}}.tiles,.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}}.tile,section{{padding:22px;background:#1b2833;border:1px solid #344754;border-radius:10px;margin:15px 0}}.value{{font-size:34px;font-weight:700}}table{{width:100%;border-collapse:collapse;background:#1b2833}}td,th{{border-bottom:1px solid #344754;padding:11px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#101820;padding:14px;border-radius:8px}}@media(max-width:640px){{.tiles,.pair{{grid-template-columns:1fr}}}}</style></head><body><main><p class="sub">MODEL CHANGE REPORT / MVP</p><h1>ModelGuard</h1><div class="status">DEPLOY GATE: {status}</div><div class="banner">{h(banner)}</div><div class="tiles"><div class="tile">Protected accuracy<div class="value">{fmt(protected['fine'])}</div><div class="sub">Base {fmt(protected['base'])} · {delta(protected['delta'])}</div></div><div class="tile">Target accuracy<div class="value">{fmt(target['fine'])}</div><div class="sub">Base {fmt(target['base'])} · {delta(target['delta'])}</div></div></div><h2>Behavior breakdown</h2><table><tr><th>Group</th><th>Category</th><th>n</th><th>Base</th><th>Fine-tuned</th><th>Change</th></tr>{columns}</table><h2>Release policy</h2><p>Maximum protected drop: {fmt(report['policy']['max_protected_drop'])} (percentage points); minimum target gain: {fmt(report['policy']['min_target_gain'])} (percentage points).</p><ul>{rules}</ul><h2>New protected regressions ({len(report['regressions'])})</h2>{failures}<p class="sub">Small evaluation sets have high uncertainty. This checks only specified examples; no claim of SAE analysis or exhaustive regression detection.</p></main></body></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='artifacts/predictions.jsonl')
    p.add_argument('--html', default='artifacts/report.html')
    p.add_argument('--json', default='artifacts/report.json')
    p.add_argument('--max-protected-drop', type=float, default=0.10)
    p.add_argument('--min-target-gain', type=float, default=0.0)
    p.add_argument('--ci', action='store_true', help='Exit 2 if blocked')
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    result = assess(rows, args.max_protected_drop, args.min_target_gain)
    for path, contents in [(args.html, render(result)), (args.json, json.dumps(result, indent=2))]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents + '\n')
    print(f"Gate {('PASS' if result['gate_passed'] else 'BLOCK')} — HTML {args.html}; JSON {args.json}")
    if args.ci and not result['gate_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
