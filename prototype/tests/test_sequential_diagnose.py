"""Offline tests for per-label acquisition diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path

from prototype.sequential_data import generate, write_files
from prototype.sequential_diagnose import (
    item_from_prompt, prepare_train_probe, template_name, read_jsonl)


class DiagnoseTests(unittest.TestCase):
    def test_prompt_parsing_all_templates(self):
        for prompt in (
            "Mapping exercise. MODE=A; ITEM=cedar; reference=TR-12.",
            "For MODE A, give the assigned code of ITEM cedar. Ticket TR-14.",
            "Task MODE=A. Input ITEM=cedar (record TR-99).",
        ):
            self.assertEqual(item_from_prompt(prompt), "cedar")

    def test_balanced_probe(self):
        files, manifest = generate("disjoint", 43, 6, 2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_files(root, files, manifest)
            probe=prepare_train_probe(root/"train_A.jsonl",root/"manifest.json",
                                      root/"probe.jsonl",per_item=3,seed=43)
            self.assertEqual(len(probe), 36)
            for item in manifest["task_A_mapping"]:
                examples=[r for r in probe if r["item"]==item]
                self.assertEqual(len(examples),3)
                self.assertEqual(len(set(template_name(r["prompt"]) for r in examples)),3)
                self.assertTrue(all(r["expected"]==manifest["task_A_mapping"][item] for r in examples))
            self.assertEqual(probe,read_jsonl(root/"probe.jsonl"))
            with self.assertRaises(FileExistsError):
                prepare_train_probe(root/"train_A.jsonl",root/"manifest.json",
                                    root/"probe.jsonl",per_item=3,seed=43)


if __name__ == "__main__":
    unittest.main()
