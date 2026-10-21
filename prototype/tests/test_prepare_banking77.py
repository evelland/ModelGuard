"""Offline tests: no model downloads or network access."""
import csv
import tempfile
import unittest
from pathlib import Path

from prototype.prepare_banking77 import balanced_sample, dedupe, protected_cases, prompt, read_csv, save_jsonl


class PreparationTests(unittest.TestCase):
    def test_balanced_sampling(self):
        import random
        data = [{"text": f"{i}-{c}", "category": c} for c in ("x", "y", "z") for i in range(10)]
        subset = balanced_sample(data, 12, random.Random(42))
        self.assertEqual(len(subset), 12)
        self.assertEqual({c: sum(r["category"] == c for r in subset) for c in ("x", "y", "z")}, {"x": 4, "y": 4, "z": 4})

    def test_deduplication_and_separation(self):
        train, keys = dedupe([{"text": " hello  world ", "category": "a"}, {"text": "HELLO WORLD", "category": "a"}])
        test, _ = dedupe([{"text": "hello world", "category": "a"}, {"text": "new prompt", "category": "a"}], keys)
        self.assertEqual((len(train), len(test)), (1, 1))

    def test_protected_set(self):
        rows = protected_cases(300, 42)
        self.assertEqual(len(rows), 300)
        self.assertEqual(len({r["id"] for r in rows}), 300)
        self.assertTrue(all(r["group"] == "protected" for r in rows))
        self.assertTrue(any(isinstance(r["expected"], dict) for r in rows))

    def test_csv_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            inp, out = Path(directory) / "source.csv", Path(directory) / "output.jsonl"
            with inp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["text", "category"])
                writer.writeheader()
                writer.writerow({"text": "Where is my card?", "category": "card_arrival"})
            self.assertEqual(read_csv(inp)[0]["category"], "card_arrival")
            save_jsonl(out, [{"prompt": prompt("Where is my card?"), "response": "card_arrival"}])
            self.assertIn("card_arrival", out.read_text())


if __name__ == "__main__":
    unittest.main()
