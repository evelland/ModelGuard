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
    from prototype.sequential_replay import build_replay_mix
    from prototype import sequential_kl
except ModuleNotFoundError:
    from train import encode_example, load_records
    from sequential_replay import build_replay_mix
    import sequential_kl


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
    replay_frac = getattr(args, "replay_frac", 0.0) or 0.0
    replay_data = getattr(args, "replay_data", None)
    if replay_frac and args.stage != "B":
        raise ValueError("Replay only applies to stage B")
    if replay_frac and not replay_data:
        raise ValueError("--replay-frac requires --replay-data (Task A training JSONL)")
    kl_weight = getattr(args, "kl_weight", 0.0) or 0.0
    kl_data = getattr(args, "kl_data", None)
    if kl_weight < 0:
        raise ValueError("--kl-weight must be non-negative")
    if kl_weight and args.stage != "B":
        raise ValueError("KL anchoring only applies to stage B")
    if kl_weight and not kl_data:
        raise ValueError("--kl-weight requires --kl-data (reference_A.jsonl)")
    act_weight = getattr(args, "act_weight", 0.0) or 0.0
    act_layer = getattr(args, "act_layer", 12)
    if act_weight < 0:
        raise ValueError("--act-weight must be non-negative")
    if act_weight and args.stage != "B":
        raise ValueError("Activation anchoring only applies to stage B")
    if act_weight and not kl_data:
        raise ValueError("--act-weight requires --kl-data (reference_A.jsonl anchor prompts)")

    rows = load_records(args.train_data)
    replay_info = None
    if args.stage == "B":
        replay_rows = load_records(replay_data) if replay_frac else []
        rows, replay_info = build_replay_mix(rows, replay_rows, replay_frac, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = [encode_example(tokenizer, r, args.max_length) for r in rows]
    model, trainable = load_stage_model(
        args.model, args.prior_adapter, device, torch)
    anchors, teacher, teacher_h, schedule, kl_info, act_info = [], [], [], [], None, None
    if kl_weight or act_weight:
        ref_rows = [json.loads(x) for x in Path(kl_data).read_text().splitlines() if x.strip()]
        anchors = [encode_example(tokenizer, a, args.max_length)
                   for a in sequential_kl.anchor_examples(ref_rows)]
        if kl_weight:
            teacher = sequential_kl.teacher_log_probs(model, anchors, device, torch)
        if act_weight:
            teacher_h = sequential_kl.teacher_hidden(model, anchors, act_layer, device, torch)
            act_info = {"act_weight": act_weight, "act_layer": act_layer, "anchor_data": kl_data,
                        "anchors": len(anchors), "penalty": "relative squared hidden-state drift vs frozen stage-A"}
        schedule = sequential_kl.anchor_schedule(len(anchors), len(encoded) * args.epochs, args.seed)
        kl_info = None if not kl_weight else {"kl_weight": kl_weight, "kl_data": kl_data, "anchors": len(anchors),
                   "teacher": "frozen stage-A adapter (cached before B training)"}
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    global_updates = 0
    micro_step = 0
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
            total = loss
            kl_value = act_value = None
            if kl_weight or act_weight:
                k = schedule[micro_step]
            if kl_weight:
                kl = sequential_kl.kl_term(
                    model, anchors[k][0], anchors[k][1], teacher[k], device, torch)
                kl_value = kl.item()
                total = total + kl_weight * kl
            if act_weight:
                drift = sequential_kl.act_term(
                    model, anchors[k][0], anchors[k][1], teacher_h[k], act_layer, device, torch)
                act_value = drift.item()
                total = total + act_weight * drift
            micro_step += 1
            (total / args.grad_accum).backward()
            if batch_idx % args.grad_accum == 0 or batch_idx == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_updates += 1
                if global_updates % 25 == 0:
                    print(f"stage={args.stage} epoch={epoch+1} "
                          f"update={global_updates} loss={loss.item():.5f}",
                          flush=True)
                entry = {"epoch":epoch+1, "update":global_updates,
                         "last_example_loss":round(loss.item(),6)}
                if kl_value is not None:
                    entry["last_anchor_kl"] = round(kl_value,6)
                if act_value is not None:
                    entry["last_anchor_act_drift"] = round(act_value,6)
                history.append(entry)
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
        "replay_data":replay_data if replay_frac else None,
        "replay":replay_info,
        "kl_anchor":kl_info,
        "act_anchor":act_info,
        "note":("Stage B continues the trainable adapter from stage A. "
                + (f"{replay_info['replay_examples']} of {replay_info['total_examples']} "
                   "stage-B examples are replayed Task A training rows (matched budget)."
                   if replay_info and replay_info["replay_examples"] else
                   "Stage B uses B examples only." if args.stage == "B" else
                   "Stage A from base weights.")),
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
    p.add_argument("--replay-data",default=None,
                   help="Stage B only: Task A training JSONL to replay from")
    p.add_argument("--replay-frac",type=float,default=0.0,
                   help="Stage B only: fraction of the stage-B budget replaced by Task A rows")
    p.add_argument("--kl-weight",type=float,default=0.0,
                   help="Stage B only: weight on KL(stage-A teacher || student) over reference prompts")
    p.add_argument("--kl-data",default=None,
                   help="Stage B only: reference_A.jsonl anchor prompts for --kl-weight")
    p.add_argument("--act-weight",type=float,default=0.0,
                   help="Stage B only: weight on relative hidden-state drift vs stage A at --act-layer on --kl-data prompts")
    p.add_argument("--act-layer",type=int,default=12,help="Hidden-state layer for --act-weight (0 = embeddings)")
    args=p.parse_args()
    train(args)


if __name__=="__main__":
    main()
