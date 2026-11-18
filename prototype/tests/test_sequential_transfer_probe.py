"""Offline tests for fresh-nonce transfer probes on both task modes."""
import tempfile
import unittest
from pathlib import Path

from prototype.sequential_data import generate, write_files
from prototype.sequential_transfer_probe import prepare


class TransferProbeTests(unittest.TestCase):
    def test_generate_both_modes(self):
        files, manifest = generate("disjoint", seed=43,
                                   examples_per_item=6,
                                   eval_per_item=2,
                                   reference_per_item=1)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_files(root, files, manifest)
            for mode in ("A", "B"):
                data = prepare(root / f"train_{mode}.jsonl",
                               root / "manifest.json",
                               root / f"fresh_{mode}.jsonl",
                               per_item=3, seed=43, mode=mode)
                self.assertEqual(len(data), 36)
                self.assertEqual({x["mode"] for x in data}, {mode})
                self.assertEqual(len({x["prompt"] for x in data}), 36)
                self.assertTrue(all(x["expected"] ==
                                    manifest[f"task_{mode}_mapping"][x["item"]]
                                    for x in data))
                self.assertTrue(all(f"MODE={mode}" in x["prompt"] or
                                    f"MODE {mode}" in x["prompt"] for x in data))


if __name__ == "__main__":
    unittest.main()
