"""Run SAE v2.2 fits across prespecified schemes, sparsities and seeds.

No model inference: consumes existing paired activation cache. Does not rank or
select models by prior protected-set results.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

try:
    from prototype.sae_v22 import fit
except ModuleNotFoundError:
    from sae_v22 import fit


def make_jobs(schemes, sparsities, seeds):
    jobs=[]
    for scheme in schemes:
        for k in sparsities:
            for seed in seeds:
                jobs.append((scheme,int(k),int(seed)))
    if not jobs:
        raise ValueError("No jobs requested")
    if len(jobs)!=len(set(jobs)):
        raise ValueError("Duplicated training jobs")
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",default="artifacts/experiments/banking77/sae_v22/paired_cache.pt")
    p.add_argument("--root",default="artifacts/experiments/banking77/sae_v23/seeds")
    p.add_argument("--schemes",nargs="+",choices=["base","pooled","delta"],default=["base","pooled","delta"])
    p.add_argument("--k",nargs="+",type=int,default=[16,32,64,128])
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--device",default="auto",choices=["auto","mps","cpu","cuda"])
    p.add_argument("--expansion",type=int,default=4)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--weight-decay",type=float,default=.0001)
    p.add_argument("--force",action="store_true",help="Explicitly overwrite existing checkpoint results")
    args=p.parse_args()
    jobs=make_jobs(args.schemes,args.k,args.seeds)
    root=Path(args.root)
    manifest={"cache":args.cache,"jobs":[]}
    for scheme,k,seed in jobs:
        dest=root/f"{scheme}_topk{k}_seed{seed}"
        if (dest/"sae_checkpoint.pt").exists() and not args.force:
            print(f"SKIP existing {dest}")
            manifest["jobs"].append({"scheme":scheme,"k":k,"seed":seed,"state":"existing"})
            continue
        opts=SimpleNamespace(cache=args.cache,scheme=scheme,out=str(dest),
             eval_cache=None,k=k,expansion=args.expansion,epochs=args.epochs,
             patience=args.patience,batch_size=args.batch_size,lr=args.lr,
             weight_decay=args.weight_decay,seed=seed,device=args.device)
        fit(opts)
        manifest["jobs"].append({"scheme":scheme,"k":k,"seed":seed,"state":"trained"})
    root.mkdir(parents=True,exist_ok=True)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Recorded {len(jobs)} configured jobs; manifest at {root/'manifest.json'}")

if __name__=="__main__":
    main()
