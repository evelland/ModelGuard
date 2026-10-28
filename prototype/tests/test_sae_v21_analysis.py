"""SAE v2.1 offline diagnostics: synthetic vectors and held-out evaluation."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from prototype.sae_v2 import make_sae
from prototype.sae_v21_analysis import (
    analyze, delta_metrics, fixed_direction_report, stratified_split, within_category_auc
)


class V21Tests(unittest.TestCase):
    def test_stratified_split_disjoint(self):
        cats = ["a"] * 20 + ["b"] * 20
        labels = ([False] * 10 + [True] * 10) * 2
        a, b = stratified_split(cats, labels, .4, 42)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a) + len(b), 40)
        self.assertEqual((a,b),stratified_split(cats,labels,.4,42))

    def test_conditional_auc_removes_group_confound(self):
        cats = ["a"] * 4 + ["b"] * 4
        labels = [False, False, True, True] * 2
        scores = [1,2,0,0,11,12,10,10]
        score, breakdown = within_category_auc(scores,labels,cats,list(range(8)))
        self.assertEqual(score,0.0)
        self.assertEqual(breakdown["a"]["pairs"],4)

    def test_direction_frozen_on_dev(self):
        cats = ["a"] * 12
        labels = ([False] * 3 + [True] * 3) * 2
        scores = [10,11,12,0,1,2,9,10,11,0,1,2]
        report = fixed_direction_report(scores,labels,cats,list(range(6)),list(range(6,12)))
        self.assertEqual(report["direction_chosen_on_dev"],"smaller")
        self.assertEqual(report["test_within_category_auc"],1.0)

    def test_delta_metrics(self):
        x=torch.tensor([[1.,0.],[0.,2.]])
        self.assertEqual(delta_metrics(x,x)["delta_r2_vs_zero"],1)
        self.assertEqual(delta_metrics(x,torch.zeros_like(x))["delta_r2_vs_zero"],0)

    def test_end_to_end_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            torch.manual_seed(1)
            n,d=40,12
            tr=torch.randn(120,d)
            base=torch.randn(n,d)
            labels=[bool(i%2) for i in range(n)]
            fine=base+torch.stack([torch.randn(d)*(.4 if label else .1) for label in labels])
            cache={"train_h":tr,"valid_h":torch.randn(40,d),"eval_base_h":base,
                   "eval_fine_h":fine,"eval_regressed":labels,
                   "eval_categories":["Arithmetic"]*20+["Formatting"]*20,
                   "eval_ids":[str(i) for i in range(n)],"metadata":{"layer":12}}
            torch.save(cache,path/"cache.pt")
            net=make_sae(d,2*d,4)
            checkpoint={"d":d,"width":2*d,"k":4,"mean":tr.mean(0,keepdim=True),
                        "scale":(tr-tr.mean(0,keepdim=True)).square().mean().sqrt(),
                        "state_dict":net.state_dict()}
            torch.save(checkpoint,path/"sae.pt")
            result=analyze(str(path/"cache.pt"),str(path/"sae.pt"),
                           str(path/"analysis.json"),pca_rank=4,bootstrap_rounds=20)
            self.assertEqual(result["split"]["dev_n"]+result["split"]["test_n"],n)
            self.assertIn("delta_r2_vs_zero",result["representation_change"])
            self.assertTrue((path/"analysis.json").exists())


if __name__=="__main__":
    unittest.main()
