"""Single-device, unquantized LoRA SFT for Qwen2.5-0.5B-Instruct."""
import argparse
import json
import random
from pathlib import Path


def load_records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r.get('prompt'), str) or not isinstance(r.get('response'), str) for r in rows):
        raise ValueError('Training JSONL requires nonempty prompt and response strings')
    return rows


def encode_example(tokenizer, example, max_length):
    user = {'role': 'user', 'content': example['prompt']}
    prefix = tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template([user, {'role': 'assistant', 'content': example['response']}], tokenize=True, add_generation_prompt=False)
    if complete[:len(prefix)] != prefix:
        raise ValueError('Unexpected chat template: assistant prefix differs from completed example')
    if len(complete) > max_length:
        raise ValueError(f"Example exceeds max length ({len(complete)} > {max_length}); raise --max-length")
    labels = [-100] * len(prefix) + complete[len(prefix):]
    if all(label == -100 for label in labels):
        raise ValueError('Example has no supervised response tokens')
    return complete, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--train-data', default='prototype/data/train.jsonl')
    p.add_argument('--output', default='artifacts/qwen-ticket-lora')
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=0, help='Optional short smoke run; 0 = all steps')
    p.add_argument('--max-length', type=int, default=192)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.grad_accum < 1 or args.max_length < 16:
        p.error('epochs and grad-accum must be positive; max-length must be >= 16')

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        p.error('MPS unavailable; select cpu or use Apple Silicon PyTorch')
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')

    rows = load_records(args.train_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = [encode_example(tokenizer, row, args.max_length) for row in rows]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation='eager')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'v_proj']))
    model.to(device)
    model.train()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    updates = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        indices = list(range(len(encoded)))
        random.shuffle(indices)
        for offset, index in enumerate(indices):
            ids, labels = encoded[index]
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            targets = torch.tensor([labels], dtype=torch.long, device=device)
            loss = model(input_ids=inputs, labels=targets).loss
            (loss / args.grad_accum).backward()
            if (offset + 1) % args.grad_accum == 0 or offset + 1 == len(indices):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                print(f'epoch={epoch+1} update={updates} loss={loss.item():.4f}', flush=True)
                if args.max_steps and updates >= args.max_steps:
                    break
        if args.max_steps and updates >= args.max_steps:
            break
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / 'run_metadata.json').write_text(json.dumps({
        'model': args.model, 'train_data': args.train_data, 'examples': len(rows),
        'updates': updates, 'seed': args.seed, 'device': device,
        'epochs_requested': args.epochs, 'max_length': args.max_length,
        'learning_rate': args.learning_rate, 'grad_accum': args.grad_accum,
        'method': 'LoRA; fp32; q_proj/v_proj',
    }, indent=2) + '\n')
    print(f'Saved adapter to {output}')


if __name__ == '__main__':
    main()
