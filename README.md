# AIsafety1

Assignment 1: Post-training Qwen2.5-3B-Instruct for persona (Mario) + STEM capability (GSM8K)
via SFT, RLAIF, RLVR.

## Files

| File | Purpose |
|---|---|
| `verifier.py` | Extracts the final numeric answer (`#### N` or `\boxed{N}`) and checks it. Single source of truth for accuracy. |
| `prepare_data.py` | Downloads GSM8K, writes `data/train_plain.jsonl` (3000), `data/val.jsonl` (500), `data/test.jsonl` (500). |
| `sft.py` | LoRA SFT on a jsonl file, loss on the assistant turn only. |
| `eval.py` | Greedy batched eval of base model + optional adapter + optional system prompt. Appends to `results/metrics.csv`. |
| `mario_prompt.txt` | System prompt for the "persona via prompting" baseline (H0). |

## Data format

One JSON object per line:

```json
{"id": "train-1458", "question": "...", "response": "...worked solution...\n#### 108", "answer": "108"}
```

`data/train_mario.jsonl` must use the same `id`, `question`, and `answer` as `data/train_plain.jsonl`,
with only `response` rewritten in Mario's voice. Every response must still end in `#### <number>`.
`sft.py` runs the verifier over the file and refuses to train if more than 10% of rows fail.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
```

## Smoke test (any machine, ~10 min on CPU)

```bash
python sft.py --data data/train_plain.jsonl --output /tmp/smoke --model Qwen/Qwen2.5-0.5B-Instruct --max-examples 32 --max-steps 4 --batch-size 2 --grad-accum 2 --warmup-steps 1
python eval.py --data data/val.jsonl --name smoke --model Qwen/Qwen2.5-0.5B-Instruct --adapter /tmp/smoke --max-examples 16 --batch-size 8
```

## Real runs (RunPod, one 24 GB GPU)

```bash
export HF_TOKEN=...
python eval.py --data data/val.jsonl --name base                                          # sanity: expect ~85%
python eval.py --data data/val.jsonl --name base_prompted --system-prompt-file mario_prompt.txt
python sft.py  --data data/train_plain.jsonl --output checkpoints/control --push-to-hub USER/qwen3b-gsm8k-control-lora
python sft.py  --data data/train_mario.jsonl --output checkpoints/mario   --push-to-hub USER/qwen3b-gsm8k-mario-lora
python eval.py --data data/val.jsonl --name control --adapter checkpoints/control
python eval.py --data data/val.jsonl --name mario   --adapter checkpoints/mario
```

Results land in `results/metrics.csv` (committed) and `results/<name>.jsonl` (per-example outputs, gitignored).
