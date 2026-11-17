"""Train Task A, then continue SAME LoRA adapter on Task B.

This is a two-stage sequential learning experiment, NOT two independently
trained adapters. Supports independent stage execution and a seed sweep.
"""
import argparse
import json
import random
from pathlib import Path

try:
    from prototype.train import encode_example, load_records
except ModuleNotFoundError:
    from train import encode_example, load_records


def resolved_device(choice, torch):
    device = choice if choice != "auto" else (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    return device


def load_stage_model(model_id, prior_adapter, device, torch):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, PeftModel, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager")
    base.config.use_cache = False
    if prior_adapter:
        if not Path(prior_adapter, "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing stage A adapter at {prior_adapter}")
        model = PeftModel.from_pretrained(
            base, prior_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]))
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable adapter parameters")
    return model, trainable


def train(args):
    import torch
    from transformers import AutoTokenizer

    if args.epochs <= 0 or args.grad_accum <= 0 or args.max_length < 32:
        raise ValueError("Invalid training hyperparameters")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolved_device(args.device, torch)
    destination = Path(args.out)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite training run: {destination}")
    if args.stage == "B" and not args.prior_adapter:
        raise ValueError("B must continue stage A's adapter via --prior-adapter")
    if args.stage == "A" and args.prior_adapter:
        raise ValueError("Stage A should start from base weights")

    rows = load_records(args.train_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = [encode_example(tokenizer, r, args.max_length) for r in rows]
    model, trainable = load_stage_model(
        args.model, args.prior_adapter, device, torch)
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    global_updates = 0
    history = []
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, index in enumerate(order, 1):
            ids, labels = encoded[index]
            token_ids = torch.tensor([ids], dtype=torch.long, device=device)
            target_ids = torch.tensor([labels], dtype=torch.long, device=device)
            loss = model(input_ids=token_ids, labels=target_ids).loss
            (loss / args.grad_accum).backward()
            if batch_idx % args.grad_accum == 0 or batch_idx == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_updates += 1
                if global_updates % 25 == 0:
                    print(f"stage={args.stage} epoch={epoch+1} "
                          f"update={global_updates} loss={loss.item():.5f}",
                          flush=True)
                history.append({"epoch":epoch+1, "update":global_updates,
                                "last_example_loss":round(loss.item(),6)})
                if args.max_steps and global_updates >= args.max_steps:
                    break
        if args.max_steps and global_updates >= args.max_steps:
            break
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    metadata = {
        "stage":args.stage, "model":args.model,
        "prior_adapter":args.prior_adapter,
        "train_data":args.train_data,"train_samples":len(rows),
        "epochs_requested":args.epochs,"updates":global_updates,
        "max_steps":args.max_steps,"seed":args.seed,
        "learning_rate":args.learning_rate,"grad_accum":args.grad_accum,
        "max_length":args.max_length,"device":device,
        "adapter":"LoRA rank=8 alpha=16 dropout=0.05, q_proj/v_proj",
        "note":"Stage B continues the trainable adapter from stage A, using B examples only.",
        "loss_trace_last":history[-15:],
    }
    (destination/"run_metadata.json").write_text(
        json.dumps(metadata,indent=2)+"\n")
    print(f"Saved stage {args.stage}: {destination}")
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("A","B"),required=True)
    p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train-data",required=True)
    p.add_argument("--prior-adapter",default=None)
    p.add_argument("--out",required=True)
    p.add_argument("--device",choices=("auto","mps","cuda","cpu"),default="auto")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--max-steps",type=int,default=0)
    p.add_argument("--learning-rate",type=float,default=2e-4)
    p.add_argument("--grad-accum",type=int,default=4)
    p.add_argument("--max-length",type=int,default=192)
    args=p.parse_args()
    train(args)


if __name__=="__main__":
    main()
