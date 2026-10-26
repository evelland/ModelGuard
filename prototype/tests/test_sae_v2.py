"""Offline SAE v2 tests; no model downloads."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import torch
from prototype.sae_v2 import auc, paired_auc_bootstrap, make_sae, fit, select_token_positions

class SAEV2Tests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(auc([.9,.1,.8,.2],[1,0,1,0]),1)
        self.assertEqual(auc([1,1],[1,0]),.5)
        self.assertIsNone(auc([1,2],[1,1]))

    def test_positions(self):
        p=select_token_positions([1,3,4,6,2],3,__import__('random').Random(42),{1,2})
        self.assertEqual(p[-1],4)
        self.assertEqual(len(p),3)

    def test_topk_and_gradients(self):
        net=make_sae(12,48,4)
        r,z=net(torch.randn(24,12))
        self.assertEqual(r.shape,(24,12))
        self.assertTrue((z.count_nonzero(dim=1)<=4).all())
        r.square().mean().backward()
        self.assertTrue(torch.isfinite(net.encoder.weight.grad).all())

    def test_fit_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            torch.manual_seed(123)
            n,d=64,16
            base=torch.randn(n,d)
            labels=[bool(i%2) for i in range(n)]
            tuned=base+torch.stack([torch.randn(d)*(.4 if y else .1) for y in labels])
            cache={"train_h":torch.randn(160,d),"valid_h":torch.randn(40,d),
                   "eval_base_h":base,"eval_fine_h":tuned,
                   "eval_regressed":labels,"eval_categories":["arithmetic"]*n,
                   "eval_ids":[str(i) for i in range(n)],"metadata":{"layer":12}}
            torch.save(cache,root/"cache.pt")
            fit(Namespace(cache=str(root/"cache.pt"),out=str(root/"out"),expansion=2,
                k=4,epochs=3,patience=3,batch_size=32,lr=1e-3,
                weight_decay=1e-4,seed=42,device="cpu",pca_rank=4))
            result=json.loads((root/"out"/"summary.json").read_text())
            self.assertLessEqual(result["quality"]["heldout_base"]["mean_l0"],4)
            self.assertEqual(result["regression_eval"]["n"],n)
            self.assertIsInstance(result["regression_eval"]["pca_auc"],float)
            self.assertTrue((root/"out"/"sae_checkpoint.pt").exists())

if __name__=="__main__":
    unittest.main()
