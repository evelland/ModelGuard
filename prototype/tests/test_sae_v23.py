"""Offline validation of SAE v2.3 PCA and new-template fixtures."""
import unittest
import torch
from prototype.sae_v23_pca import compare_ranks, vector_families
from prototype.sae_v23_holdout import generate, FAMILIES
from prototype.prepare_banking77 import protected_cases

class V23Tests(unittest.TestCase):
    def test_exact_pca_rank_sweep(self):
        torch.manual_seed(0)
        train=torch.randn(100, 6)
        valid=torch.randn(20, 6)
        x=compare_ranks(train,valid,["banking"]*10+["protected"]*10,(1,2,6))
        self.assertEqual(len(x["ranks"]),3)
        self.assertAlmostEqual(x["ranks"][-1]["validation_r2_vs_train_mean_delta"],1.,places=6)
        self.assertTrue(x["ranks"][0]["train_centered_explained_variance"] <
                        x["ranks"][-1]["train_centered_explained_variance"])

    def test_family_alignment(self):
        cache={"valid_families":["banking","protected"],
               "valid_positions":[[2,3],[4,6]]}
        self.assertEqual(vector_families(cache,"valid",4),
                         ["banking","banking","protected","protected"])

    def test_new_template_cases(self):
        cases=generate(30,seed=17)
        previous={x["prompt"] for x in protected_cases(300,42)}
        self.assertEqual(len(cases),120)
        self.assertEqual(len({x["id"] for x in cases}),120)
        self.assertFalse(previous & {x["prompt"] for x in cases})
        self.assertEqual({x["template_family"] for x in cases},set(FAMILIES.values()))
        self.assertEqual(len({x['prompt'] for x in cases}), 120)
        self.assertEqual(cases,generate(30,seed=17))
        self.assertNotEqual(cases,generate(30,seed=18))

    def test_default_holdout_unique_and_capacity(self):
        cases = generate()
        self.assertEqual(len(cases), 400)
        self.assertEqual(len({row["prompt"] for row in cases}), 400)
        self.assertEqual(cases, generate())
        with self.assertRaisesRegex(ValueError, "capacity"):
            generate(per_family=301)

if __name__=="__main__":
    unittest.main()
