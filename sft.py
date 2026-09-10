"""LoRA SFT of Qwen2.5-3B-Instruct on a jsonl dataset, loss on the assistant turn only.

Usage (real run on a GPU pod):
  python sft.py --data data/train_mario.jsonl --output checkpoints/mario
  python sft.py --data data/train_plain.jsonl --output checkpoints/control

Smoke test (any machine, tiny model, a few steps):
  python sft.py --data data/train_plain.jsonl --output /tmp/smoke \
      --model Qwen/Qwen2.5-0.5B-Instruct --max-examples 32 --max-steps 4 --batch-size 2

The dataset schema is the one produced by prepare_data.py. Only "question" and "response"
are used. No system prompt is inserted, so the persona is baked into the weights and the
eval can run the adapter with no system prompt either.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from verifier import is_correct


def load_jsonl(path: str, max_examples: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if max_examples:
        rows = rows[:max_examples]
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--max-examples", type=int, default=None, help="truncate dataset (smoke tests)")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--push-to-hub", default=None, help="e.g. yourname/qwen3b-mario-lora (private)")
    p.add_argument("--skip-verify", action="store_true", help="don't drop rows whose response fails the verifier")
    args = p.parse_args()

    rows = load_jsonl(args.data, args.max_examples)
    if not args.skip_verify:
        good = [r for r in rows if is_correct(r["response"], r["answer"])]
        dropped = len(rows) - len(good)
        print(f"verifier filter: kept {len(good)} / {len(rows)} (dropped {dropped})")
        if dropped > 0.1 * len(rows):
            raise SystemExit("More than 10% of rows fail the verifier. Fix the dataset before training.")
        rows = good

    # Conversational prompt/completion format: TRL masks the prompt and trains on the
    # completion (assistant) tokens only.
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": r["question"]}],
                "completion": [{"role": "assistant", "content": r["response"]}],
            }
            for r in rows
        ]
    )

    use_cuda = torch.cuda.is_available()
    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        completion_only_loss=True,
        bf16=use_cuda,
        gradient_checkpointing=use_cuda,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        model_init_kwargs={"dtype": torch.bfloat16 if use_cuda else torch.float32},
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds, peft_config=lora)

    # Report sequence-length stats so max_len truncation is visible, not silent.
    lens = [len(x) for x in trainer.train_dataset["input_ids"]]
    print(f"tokenized: n={len(lens)} mean={sum(lens)/len(lens):.0f} max={max(lens)} "
          f"truncated_at_{args.max_len}={sum(l >= args.max_len for l in lens)}")

    trainer.model.print_trainable_parameters()
    result = trainer.train()
    print(result.metrics)

    trainer.save_model(args.output)  # adapter weights + config only
    Path(args.output, "train_args.json").write_text(json.dumps(vars(args), indent=2))
    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed adapter to {args.push_to_hub}")


if __name__ == "__main__":
    main()
