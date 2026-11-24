"""Offline tests for the replay baseline. No model downloads."""
import json
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path

from prototype.sequential_data import generate
from prototype.sequential_diagnose import item_from_prompt
from prototype.sequential_replay import build_replay_mix, forbid_overlap, mode_of
from prototype.sequential_replay_sweep import kl_tag, plan, run, summarize, tag
from prototype.sequential_kl import anchor_examples, anchor_schedule, response_positions


def data():
    files, _ = generate("disjoint", seed=43, examples_per_item=10,
                        eval_per_item=2, reference_per_item=1)
    return files


class ReplayMixTests(unittest.TestCase):
    def test_zero_replay_is_unchanged_b(self):
        files = data()
        mixed, info = build_replay_mix(files["train_B.jsonl"], files["train_A.jsonl"], 0.0, 43)
        self.assertEqual(mixed, files["train_B.jsonl"])
        self.assertEqual(info["replay_examples"], 0)

    def test_matched_budget_and_counts(self):
        files = data()
        b, a = files["train_B.jsonl"], files["train_A.jsonl"]
        for fraction in (0.05, 0.1, 0.2):
            mixed, info = build_replay_mix(b, a, fraction, 43)
            self.assertEqual(len(mixed), len(b))
            modes = Counter(mode_of(r["prompt"]) for r in mixed)
            self.assertEqual(modes["A"], round(len(b) * fraction))
            self.assertEqual(modes["A"] + modes["B"], len(b))
            self.assertEqual(info["replay_examples"], modes["A"])

    def test_replay_rows_come_from_a_training_only(self):
        files = data()
        mixed, _ = build_replay_mix(files["train_B.jsonl"], files["train_A.jsonl"], 0.2, 43)
        a_train = {r["prompt"] for r in files["train_A.jsonl"]}
        b_train = {r["prompt"] for r in files["train_B.jsonl"]}
        for row in mixed:
            self.assertIn(row["prompt"], a_train | b_train)
        held_out = {r["prompt"] for name in ("eval_combined_dev.jsonl", "eval_combined_test.jsonl",
                                             "reference_A.jsonl") for r in files[name]}
        self.assertFalse({r["prompt"] for r in mixed} & held_out)

    def test_stratified_across_items(self):
        files = data()
        mixed, _ = build_replay_mix(files["train_B.jsonl"], files["train_A.jsonl"], 0.1, 43)
        a_items = Counter(item_from_prompt(r["prompt"]) for r in mixed if mode_of(r["prompt"]) == "A")
        b_items = Counter(item_from_prompt(r["prompt"]) for r in mixed if mode_of(r["prompt"]) == "B")
        self.assertLessEqual(max(a_items.values()) - min(a_items.values()), 1)
        self.assertEqual(len(b_items), 12)
        self.assertLessEqual(max(b_items.values()) - min(b_items.values()), 1)

    def test_deterministic(self):
        files = data()
        args = (files["train_B.jsonl"], files["train_A.jsonl"], 0.1, 43)
        self.assertEqual(build_replay_mix(*args), build_replay_mix(*args))
        self.assertNotEqual(build_replay_mix(*args)[0],
                            build_replay_mix(files["train_B.jsonl"], files["train_A.jsonl"], 0.1, 44)[0])

    def test_rejects_bad_inputs(self):
        files = data()
        with self.assertRaises(ValueError):
            build_replay_mix(files["train_B.jsonl"], files["train_A.jsonl"], 1.0, 43)
        with self.assertRaises(ValueError):
            build_replay_mix(files["train_B.jsonl"], files["train_B.jsonl"], 0.1, 43)
        with self.assertRaises(ValueError):
            build_replay_mix(files["train_A.jsonl"], files["train_A.jsonl"], 0.1, 43)
        with self.assertRaises(ValueError):
            build_replay_mix(files["train_B.jsonl"], [], 0.1, 43)

    def test_forbid_overlap(self):
        files = data()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in files["eval_combined_dev.jsonl"]))
            forbid_overlap(files["train_A.jsonl"], path)
            leaked = files["train_A.jsonl"] + [files["eval_combined_dev.jsonl"][0]]
            with self.assertRaises(ValueError):
                forbid_overlap(leaked, path)


class ReplaySweepTests(unittest.TestCase):
    def test_every_b_run_continues_same_a_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = plan(Path(tmp), [0.0, 0.1])
            b_runs = [s for s in steps if s[0].startswith("train B")]
            self.assertEqual(len(b_runs), 2)
            for _, command, _ in b_runs:
                self.assertEqual(command[command.index("--prior-adapter") + 1],
                                 str(Path(tmp) / "adapter_A"))
                self.assertEqual(command[command.index("--epochs") + 1], "2")
            self.assertNotIn("--replay-frac", b_runs[0][1])
            self.assertEqual(b_runs[1][1][b_runs[1][1].index("--replay-frac") + 1], "0.1")
            self.assertEqual(b_runs[1][1][b_runs[1][1].index("--replay-data") + 1],
                             str(Path(tmp) / "train_A.jsonl"))
            a_run = [s for s in steps if s[0] == "train Task A adapter"][0][1]
            self.assertEqual(a_run[a_run.index("--epochs") + 1], "8")

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sweep"
            run(Namespace(root=str(root), fractions=[0.0, 0.1], condition="disjoint", seed=43,
                          device="cpu", epochs_a=8, epochs_b=2, learning_rate=2e-4,
                          max_new_tokens=16, dry_run=True))
            self.assertFalse(root.exists())

    def test_summary_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fraction, (a_after_b, replayed) in {0.0: (31, 0), 0.1: (34, 43)}.items():
                t = tag(fraction)
                (root / f"comparison_{t}.json").write_text(json.dumps({"stage_comparison": {
                    "A": {"n": 36, "after_A_correct": 36, "after_B_correct": a_after_b,
                          "A_to_B_regressions": 36 - a_after_b,
                          "retention_given_A_correct": round(a_after_b / 36, 5)},
                    "B": {"n": 36, "after_A_correct": 12, "after_B_correct": 36,
                          "after_B_accuracy": 1.0}}}))
                (root / f"adapter_AB_{t}").mkdir()
                (root / f"adapter_AB_{t}" / "run_metadata.json").write_text(json.dumps({
                    "train_samples": 432, "updates": 216,
                    "replay": {"replay_examples": replayed}}))
            summary = summarize(root, [0.0, 0.1])
            self.assertEqual([r["A_new_failures"] for r in summary["rows"]], [5, 2])
            self.assertTrue((root / "replay_sweep_summary.md").read_text().startswith("| Method"))


class KLAnchorTests(unittest.TestCase):
    def test_anchors_only_from_reference_set(self):
        files = data()
        anchors = anchor_examples(files["reference_A.jsonl"])
        self.assertEqual(len(anchors), 12)
        self.assertTrue(all(set(a) == {"prompt", "response"} for a in anchors))
        held_out = {r["prompt"] for r in files["eval_combined_dev.jsonl"] + files["eval_combined_test.jsonl"]}
        self.assertFalse({a["prompt"] for a in anchors} & held_out)
        with self.assertRaises(ValueError):
            anchor_examples(files["eval_A_dev.jsonl"])
        with self.assertRaises(ValueError):
            anchor_examples([])

    def test_schedule_covers_anchors_evenly(self):
        order = anchor_schedule(12, 30, 43)
        self.assertEqual(len(order), 30)
        self.assertEqual(sorted(order[:12]), list(range(12)))
        self.assertEqual(sorted(order[12:24]), list(range(12)))
        self.assertEqual(order, anchor_schedule(12, 30, 43))
        with self.assertRaises(ValueError):
            anchor_schedule(0, 5, 43)

    def test_response_positions_shift(self):
        self.assertEqual(response_positions([-100, -100, -100, 7, 8]), [2, 3])
        with self.assertRaises(ValueError):
            response_positions([-100, -100])

    def test_sweep_plans_kl_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = plan(Path(tmp), [0.0], kl_weights=[0.5, 2.0])
            kl_runs = [s for s in steps if s[0].startswith("train B (KL")]
            self.assertEqual(len(kl_runs), 2)
            cmd = kl_runs[0][1]
            self.assertEqual(cmd[cmd.index("--kl-data") + 1], str(Path(tmp) / "reference_A.jsonl"))
            self.assertEqual(cmd[cmd.index("--kl-weight") + 1], "0.5")
            self.assertNotIn("--replay-frac", cmd)
            self.assertEqual(cmd[cmd.index("--prior-adapter") + 1], str(Path(tmp) / "adapter_A"))
            self.assertEqual(kl_tag(0.5), "kl0p5")


if __name__ == "__main__":
    unittest.main()
