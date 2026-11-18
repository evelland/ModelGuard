"""Synthetic tests for the controlled sequential A->B study. No model downloads."""
import json
import tempfile
import unittest
from pathlib import Path

from prototype.sequential_data import generate, mappings, write_files
from prototype.sequential_score import score_pair
from prototype.sequential_train import resolved_device


class SequentialTests(unittest.TestCase):
    def test_balanced_data(self):
        for condition in ("disjoint", "shared"):
            files, metadata = generate(condition, seed=42, examples_per_item=5,
                                       eval_per_item=2, reference_per_item=2)
            self.assertEqual(len(files["train_A.jsonl"]), 60)
            self.assertEqual(len(files["train_B.jsonl"]), 60)
            self.assertEqual(len(files["eval_combined_test.jsonl"]), 48)
            self.assertEqual(len({row["prompt"] for group in files.values() for row in group}),
                             60 + 60 + 24 + 24 + 24 + 24 + 24)
            self.assertEqual((files,metadata),
                             generate(condition,seed=42,examples_per_item=5,
                                      eval_per_item=2,reference_per_item=2))

    def test_mapping_condition(self):
        a,b=mappings("disjoint",42)
        self.assertFalse(set(a) & set(b))
        a,b=mappings("shared",42)
        self.assertEqual(set(a),set(b))
        self.assertTrue(all(a[item]!=b[item] for item in a))

    def test_save_refuses_overwrite(self):
        files,manifest=generate(examples_per_item=2,eval_per_item=1,reference_per_item=1)
        with tempfile.TemporaryDirectory() as tmp:
            write_files(tmp,files,manifest)
            m=json.loads((Path(tmp)/"manifest.json").read_text())
            self.assertTrue(m["sha256"])
            with self.assertRaises(FileExistsError):
                write_files(tmp,files,manifest)

    def test_score_transitions(self):
        cases={}
        after={}
        for mode, expected in (("A","K2"),("B","R7")):
            rid=f"{mode}-1"
            base={"id":rid,"mode":mode,"group":mode,"item":"cedar",
                  "prompt":f"MODE {mode} cedar", "expected":expected}
            cases[rid]={**base,"base_response":"no","fine_response":"K2" if mode=="A" else "K2"}
            after[rid]={**base,"base_response":"no","fine_response":"R7"}
        result=score_pair(cases,after)
        self.assertEqual(result["Task_A_forgetting"]["new_failures"],1)
        self.assertEqual(result["Task_B_learning"]["B_correct_after_B"],1)
        self.assertEqual(result["Task_A_forgetting"]["retention"],0)

    def test_score_rejects_unpaired(self):
        a={"x":{"id":"x","mode":"A","prompt":"a","expected":"K2","group":"A","item":"x",
                "fine_response":"K2"}}
        with self.assertRaises(ValueError):
            score_pair(a,{})


if __name__=="__main__":
    unittest.main()
