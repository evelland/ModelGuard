"""Greedy paired inference using the same Qwen base weights and a LoRA adapter."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--adapter', default='artifacts/qwen-ticket-lora')
    parser.add_argument('--cases', default='prototype/data/eval.jsonl')
    parser.add_argument('--output', default='artifacts/predictions.jsonl')
    parser.add_argument('--max-new-tokens', type=int, default=48)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation='eager')
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()

    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError('No evaluation cases')
    if len({row['id'] for row in cases}) != len(cases):
        raise ValueError('Duplicate evaluation IDs')

    @torch.inference_mode()
    def generate(prompt):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, return_tensors='pt').to(device)
        output = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf8') as file:
        for i, original in enumerate(cases, 1):
            row = dict(original)
            with model.disable_adapter():
                row['base_response'] = generate(row['prompt'])
            row['fine_response'] = generate(row['prompt'])
            row['source'] = 'model_inference'
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f"{i}/{len(cases)} {row['id']} base={row['base_response']!r} tuned={row['fine_response']!r}", flush=True)
    print(f'Wrote {dest}')


if __name__ == '__main__':
    main()
