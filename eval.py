"""Evaluate base model (+ optional LoRA adapter, + optional system prompt) on a jsonl split.

Greedy decoding, batched. Reports strict accuracy (number after "####"), lenient accuracy
(last number anywhere, diagnostic only), parse-failure rate, and mean generated length.

Examples:
  python eval.py --data data/val.jsonl --name base
  python eval.py --data data/val.jsonl --name base_prompted --system-prompt-file mario_prompt.txt
  python eval.py --data data/val.jsonl --name control --adapter checkpoints/control
  python eval.py --data data/val.jsonl --name mario --adapter checkpoints/mario

Each run writes results/<name>.jsonl (per-example) and appends a row to results/metrics.csv.
"""

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import extract_answer, is_correct


def load_jsonl(path: str, max_examples: int | None) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:max_examples] if max_examples else rows


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True, help="run label, used for output filenames")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--system-prompt-file", default=None)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text().strip()

    device = pick_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()  # faster generation, identical outputs
    model.eval()

    rows = load_jsonl(args.data, args.max_examples)
    prompts = []
    for r in rows:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": r["question"]})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    outputs: list[str] = []
    t0 = time.time()
    for i in tqdm(range(0, len(prompts), args.batch_size), desc=args.name):
        batch = prompts[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
    elapsed = time.time() - t0

    n = len(rows)
    strict = sum(is_correct(o, r["answer"]) for o, r in zip(outputs, rows))
    lenient = sum(is_correct(o, r["answer"], lenient=True) for o, r in zip(outputs, rows))
    no_marker = sum(extract_answer(o) is None for o in outputs)
    gen_lens = [len(tok(o)["input_ids"]) for o in outputs]
    hit_cap = sum(l >= args.max_new_tokens for l in gen_lens)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.name}.jsonl").open("w") as f:
        for r, o in zip(rows, outputs):
            f.write(json.dumps({
                "id": r["id"], "question": r["question"], "gold": r["answer"],
                "output": o, "pred": extract_answer(o), "correct": is_correct(o, r["answer"]),
            }, ensure_ascii=False) + "\n")

    summary = {
        "name": args.name, "date": date.today().isoformat(), "data": args.data, "n": n,
        "model": args.model, "adapter": args.adapter or "", "system_prompt": bool(system_prompt),
        "strict_acc": round(strict / n, 4), "lenient_acc": round(lenient / n, 4),
        "no_marker_rate": round(no_marker / n, 4), "hit_token_cap": hit_cap,
        "mean_gen_tokens": round(sum(gen_lens) / n, 1), "seconds": round(elapsed, 1),
    }
    csv_path = out_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
