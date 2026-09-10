"""Build the GSM8K splits in the shared jsonl format.

Writes:
  data/train_plain.jsonl   3000 examples from the official train split, original answers
  data/val.jsonl            500 examples from the official train split (disjoint from above)
  data/test.jsonl           500 examples from the official TEST split (matches the 86.7 benchmark)

Schema (one JSON object per line):
  {"id": str, "question": str, "response": str, "answer": str}
  - response: full worked solution ending in "#### <number>"
  - answer:   just the final number as a string

The partner's Mario dataset (data/train_mario.jsonl) must use the same ids, questions, and
answers, with only "response" rewritten.
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from verifier import extract_answer, is_correct


_CALC_RE = re.compile(r"<<[^>]*>>")


def clean_response(text: str) -> str:
    """Strip GSM8K calculator annotations like <<4*20=80>> so the control set matches the
    plain-prose format the Mario rewrites will have."""
    return _CALC_RE.sub("", text)


def to_record(split: str, idx: int, row: dict) -> dict:
    gold = extract_answer(row["answer"])
    assert gold is not None, row
    ans = str(int(gold)) if gold == int(gold) else str(gold)
    return {"id": f"{split}-{idx}", "question": row["question"],
            "response": clean_response(row["answer"]), "answer": ans}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):5d} -> {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    train = [to_record("train", i, r) for i, r in enumerate(ds["train"])]
    test = [to_record("test", i, r) for i, r in enumerate(ds["test"])]

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(test)
    assert args.n_train + args.n_val <= len(train), "not enough train examples"
    assert args.n_test <= len(test), "not enough test examples"

    out = Path(args.out_dir)
    write_jsonl(out / "train_plain.jsonl", train[: args.n_train])
    write_jsonl(out / "val.jsonl", train[args.n_train : args.n_train + args.n_val])
    write_jsonl(out / "test.jsonl", test[: args.n_test])

    # Sanity: every gold response verifies against its own answer.
    for r in train[: args.n_train + args.n_val] + test[: args.n_test]:
        assert is_correct(r["response"], r["answer"]), r["id"]
    print("all gold responses verify")


if __name__ == "__main__":
    main()
