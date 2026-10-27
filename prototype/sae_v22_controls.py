"""SAE v2.2 controls: compare zero, train-mean, PCA and fitted SAE delta.

Works entirely on the existing paired cache and fitted SAE checkpoints.
"""
import argparse
import json
from pathlib import Path

import torch

try:
    from prototype.sae_v2 import make_sae
    from prototype.sae_v21_analysis import delta_metrics
except ModuleNotFoundError:
    from sae_v2 import make_sae
    from sae_v21_analysis import delta_metrics


def evaluate(cache_path, checkpoint_path, out, rank=32, seed=42):
    torch.manual_seed(seed)
    cache=torch.load(cache_path,map_location="cpu",weights_only=True)
    ck=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    train=cache["train_fine_h"].float()-cache["train_base_h"].float()
    valid=cache["valid_fine_h"].float()-cache["valid_base_h"].float()
    mean=train.mean(0,keepdim=True)
    zero=torch.zeros_like(valid)
    constant=mean.expand_as(valid)
    if rank<1:
        raise ValueError("PCA rank must be positive")
    centered=train-mean
    q=min(rank,centered.shape[0]-1,centered.shape[1])
    if q<1: raise ValueError("Not enough training data for PCA")
    _,_,v=torch.pca_lowrank(centered,q=q,center=False,niter=3)
    pca=mean+(valid-mean)@v@v.T
    net=make_sae(int(ck["d"]),int(ck["width"]),int(ck["k"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    scale=ck["scale"].float()
    center=ck["mean"].float()
    def reconstruct(x):
        with torch.inference_mode():
            return net((x-center)/scale)[0]*scale+center
    if ck["scheme"]=="delta":
        predicted=reconstruct(valid)
    else:
        predicted=reconstruct(cache["valid_fine_h"].float())-reconstruct(cache["valid_base_h"].float())
    baseline_sse=(valid-constant).square().sum().item()
    def assess(pred):
        obj=delta_metrics(valid,pred)
        obj["r2_vs_train_mean_delta"]=round(1-(valid-pred).square().sum().item()/max(baseline_sse,1e-12),5)
        return obj
    result={
        "scheme":ck["scheme"],"k":ck["k"],"pca_rank":q,"seed":seed,
        "validation_n":valid.shape[0],
        "mean_delta_norm":round(mean.norm().item(),5),
        "centered_delta_energy_fraction":round(baseline_sse/max(valid.square().sum().item(),1e-12),5),
        "methods":{"zero":assess(zero),"mean_train_delta":assess(constant),
                   "pca_centered_delta":assess(pca),"sae":assess(predicted)},
        "limitations":["Uses calibration validation data, not a template-family-independent test.",
                        "PCA is trained only on calibration training deltas.",
                        "Results quantify delta reconstruction, not forgetting detection or causal explanations."]
    }
    target=Path(out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(f"{ck['scheme']} k={ck['k']}: mean baseline R²0={result['methods']['mean_train_delta']['delta_r2_vs_zero']:.4f}; "
          f"SAE R²mean={result['methods']['sae']['r2_vs_train_mean_delta']:.4f}; "
          f"PCA R²mean={result['methods']['pca_centered_delta']['r2_vs_train_mean_delta']:.4f}")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",default="artifacts/experiments/banking77/sae_v22/paired_cache.pt")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--rank",type=int,default=32)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    evaluate(a.cache,a.checkpoint,a.out,a.rank,a.seed)


if __name__=="__main__":
    main()
