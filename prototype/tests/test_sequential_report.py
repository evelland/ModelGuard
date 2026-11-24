"""Offline tests for the A->B regression report bridge."""
import json
import tempfile
import unittest
from pathlib import Path

from prototype.sequential_report import build, to_report_rows


def preds(a_ok, b_ok, n=6):
    rows = {}
    for mode, ok in (("A", a_ok), ("B", b_ok)):
        for i in range(n):
            id_ = f"{mode}-{i}"
            rows[id_] = {"id": id_, "mode": mode, "prompt": f"MODE={mode} ITEM=x{i}", "expected": "K2",
                         "validator": "exact", "fine_response": "K2" if i < ok else "R7",
                         "source": "model_inference"}
    return rows


class SequentialReportTests(unittest.TestCase):
    def test_roles(self):
        rows = to_report_rows(preds(6, 1), preds(6, 6))
        self.assertEqual({r["group"] for r in rows if r["id"].startswith("A")}, {"protected"})
        self.assertEqual({r["group"] for r in rows if r["id"].startswith("B")}, {"target"})
        self.assertTrue(all(r["source"] == "model_inference" for r in rows))

    def test_gate_blocks_forgetting_and_passes_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            def dump(name, data):
                path = Path(tmp) / name
                path.write_text("".join(json.dumps(r) + "\n" for r in data.values()))
                return path
            after_a = dump("a.jsonl", preds(6, 1))
            forgot = build(after_a, dump("b1.jsonl", preds(0, 6)))
            self.assertFalse(forgot["gate_passed"])
            self.assertEqual(forgot["paired_test"]["regressions"], 6)
            kept = build(after_a, dump("b2.jsonl", preds(6, 6)))
            self.assertTrue(kept["gate_passed"])
            self.assertEqual(kept["metrics"]["target"]["fine"], 1.0)

    def test_mismatch_rejected(self):
        a = preds(6, 6)
        b = preds(6, 6)
        b.pop("A-0")
        with self.assertRaises(ValueError):
            to_report_rows(a, b)


if __name__ == "__main__":
    unittest.main()
