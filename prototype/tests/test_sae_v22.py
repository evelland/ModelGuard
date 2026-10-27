"""Offline tests for v2.2 paired activation training; no Qwen download."""
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from prototype.sae_v22 import (
    fingerprint, make_calibration_splits, validate_paired_cache, scheme_vectors, fit,
)


class V22Tests(unittest.TestCase):
    def test_deterministic_split_and_no_eval_overlap(self):
        train = [{"prompt": f"Bank question {i}"} for i in range(50)]
        evaluation = [{"prompt": "Bank question 3"}, {"prompt": "What is 6 plus 8? Reply with only the number."}]
        a, b = make_calibration_splits(train, evaluation, 50, 60, seed=51)
        a2, b2 = make_calibration_splits(train, evaluation, 50, 60, seed=51)
        self.assertEqual(a, a2)
        self.assertEqual(b, b2)
        hashes = {v[1] for v in a+b}
        self.assertNotIn(fingerprint("Bank question 3"), hashes)
        self.assertFalse({v[1] for v in a} & {v[1] for v in b})

    def synthetic_cache(self):
        torch.manual_seed(9)
        tr = torch.randn(40, 12)
        va = torch.randn(20, 12)
        return {
            "train_base_h": tr, "train_fine_h": tr + 0.25 * torch.randn_like(tr),
            "valid_base_h": va, "valid_fine_h": va + 0.25 * torch.randn_like(va),
            "train_prompt_hashes": ["a"], "valid_prompt_hashes": ["b"],
            "metadata": {"layer": 12, "model": "tiny"}
        }

    def test_pair_shapes_and_schemes(self):
        c = self.synthetic_cache()
        self.assertTrue(validate_paired_cache(c))
        for scheme, n in [("base", 40), ("pooled", 80), ("delta", 40)]:
            tr, va = scheme_vectors(c, scheme)
            self.assertEqual(tr.shape, (n, 12))
            self.assertEqual(va.shape[1], 12)
        c["valid_prompt_hashes"] = ["a"]
        with self.assertRaises(ValueError):
            validate_paired_cache(c)

    def test_all_schemes_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save(self.synthetic_cache(), root/"cache.pt")
            for scheme in ("base", "pooled", "delta"):
                args = Namespace(cache=str(root/"cache.pt"), scheme=scheme,
                    out=str(root/scheme), eval_cache=None, k=4, expansion=2,
                    epochs=3, patience=3, batch_size=16, lr=.001,
                    weight_decay=.0001, seed=42, device="cpu")
                result = fit(args)
                self.assertEqual(result["scheme"], scheme)
                self.assertIn("delta_r2_vs_zero", result["paired_validation_delta"])
                self.assertTrue((root/scheme/"sae_checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
