"""Check local BANKING77 SAE v2 inputs before model execution (no downloads)."""
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    paths = {
        "BANKING77 training prompts": root / "artifacts/experiments/banking77/train.jsonl",
        "Paired regression report": root / "artifacts/experiments/banking77/report.json",
        "LoRA adapter config": root / "artifacts/banking77-2700-lora/adapter_config.json",
    }
    missing = []
    for name, path in paths.items():
        if path.is_file():
            print(f"OK   {name}: {path.relative_to(root)}")
        else:
            print(f"MISS {name}: {path.relative_to(root)}")
            missing.append(name)
    if missing:
        print("\nMissing local artifacts. Reuse outputs from PR #16; do not regenerate the dataset with a different seed inadvertently.")
        raise SystemExit(2)
    rows = [json.loads(line) for line in paths["BANKING77 training prompts"].read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r.get("prompt"), str) for r in rows):
        raise SystemExit("Invalid training JSONL (expected prompt strings)")
    report = json.loads(paths["Paired regression report"].read_text())
    eligible = [r for r in report.get("rows", [])
                if r.get("group") == "protected" and r.get("base_pass") is True
                and isinstance(r.get("fine_pass"), bool)]
    regressions = sum(not r["fine_pass"] for r in eligible)
    retained = len(eligible) - regressions
    print(f"Training examples: {len(rows)}. Base-correct protected eval: {len(eligible)} ({regressions} regressed, {retained} retained).")
    if regressions == 0 or retained == 0:
        raise SystemExit("Need both regressed and retained base-correct protected cases for AUC.")
    adapter = json.loads(paths["LoRA adapter config"].read_text())
    print(f"Adapter base model: {adapter.get('base_model_name_or_path', 'unspecified')}")
    print("Preflight passed; model download/cache and device availability are checked during collect.")

if __name__ == "__main__":
    main()
