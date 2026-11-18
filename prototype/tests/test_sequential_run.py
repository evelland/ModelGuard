"""End-to-end runner orchestration tests; no Qwen downloads."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from prototype.sequential_run import steps, validated_predictions, run


class SequentialRunnerTests(unittest.TestCase):
    def test_order_and_checkpoint_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = steps(Path(directory), seed=43, condition="disjoint")
            self.assertEqual(len(plan), 6)
            labels = [p[0] for p in plan]
            self.assertIn("train new Task A", labels[1])
            self.assertIn("continue SAME adapter", labels[3])
            self.assertIn("--prior-adapter", plan[3][1])
            self.assertEqual(plan[1][1][plan[1][1].index("--seed")+1], "43")
            self.assertEqual(plan[3][1][plan[3][1].index("--seed")+1], "43")
            self.assertEqual(plan[2][1][plan[2][1].index("--cases")+1],
                             plan[4][1][plan[4][1].index("--cases")+1])
            self.assertEqual(plan[3][1][plan[3][1].index("--prior-adapter")+1],
                             str(Path(directory)/"adapter_A"))

    def test_dry_run_never_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"new-run"
            args=Namespace(out=str(root),condition="disjoint",seed=43,device="cpu",
                           epochs_a=4,epochs_b=2,learning_rate=.0002,
                           max_new_tokens=16,dry_run=True)
            run(args)
            self.assertFalse(root.exists())

    def test_incomplete_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"eval_combined_dev.jsonl").write_text(
                json.dumps({"id":"a"})+"\n"+json.dumps({"id":"b"})+"\n")
            (root/"dev_after_A.jsonl").write_text(
                json.dumps({"id":"a","fine_response":"K2"})+"\n")
            with self.assertRaisesRegex(ValueError, "Incomplete evaluation"):
                validated_predictions(root, "dev_after_A.jsonl")


if __name__ == "__main__":
    unittest.main()
