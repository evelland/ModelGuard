"""Offline mean-delta and PCA controls sanity test."""
import tempfile
import unittest
from pathlib import Path
import torch
from prototype.sae_v2 import make_sae
from prototype.sae_v22_controls import evaluate

class ControlsTests(unittest.TestCase):
    def test_constant_shift_baseline(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            torch.manual_seed(10)
            d=12; base=torch.randn(64,d); vb=torch.randn(24,d)
            shift=torch.arange(d).float()/10
            cache={"train_base_h":base,"train_fine_h":base+shift,
                   "valid_base_h":vb,"valid_fine_h":vb+shift}
            torch.save(cache,p/"cache.pt")
            net=make_sae(d,24,4)
            torch.save({"d":d,"width":24,"k":4,"scheme":"delta",
                        "state_dict":net.state_dict(),
                        "mean":shift.unsqueeze(0),"scale":torch.tensor(1.)},p/"sae.pt")
            output=evaluate(p/"cache.pt",p/"sae.pt",p/"result.json",rank=4)
            self.assertGreater(output["methods"]["mean_train_delta"]["delta_r2_vs_zero"],.99999)
            self.assertAlmostEqual(output["methods"]["mean_train_delta"]["r2_vs_train_mean_delta"],0.)
            self.assertTrue((p/"result.json").exists())

if __name__=="__main__":
    unittest.main()
